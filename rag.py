"""
rag.py
Indian Legal Research Assistant - retrieval + generation
"""

from typing import Dict, Iterator, List

import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from pathlib import Path

from config import (
    VECTORSTORE_PATH,
    DATA_PATH,
    METADATA_PATH,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    SEARCH_TYPE,
    TOP_K,
    FETCH_K,
    MMR_LAMBDA,
    MAX_CONTEXT_CHARS,
    MAX_CASES,
)
from prompts import RAG_PROMPT
from case_lookup import law_info, find_related_cases


# ============================================================
# Cached resources (loaded once per Streamlit server)
# ============================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(
        str(VECTORSTORE_PATH),
        load_embeddings(),
        allow_dangerous_deserialization=True,
    )


@st.cache_resource
def load_retriever():
    db = load_vectorstore()

    kwargs = {"k": TOP_K}
    if SEARCH_TYPE == "mmr":
        kwargs.update({"fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA})

    return db.as_retriever(search_type=SEARCH_TYPE, search_kwargs=kwargs)


@st.cache_resource
def load_llm():
    # Prefer the configured Nemotron model, then other lightweight chat models.
    # NVIDIA's catalog changes over time, so intersect this preference list
    # with the models currently exposed to the user's API key at app startup.
    preferred_models = [
        LLM_MODEL,
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-nano-30b-a3b",
        "nvidia/nvidia-nemotron-nano-9b-v2",
        "meta/llama-3.1-8b-instruct",
    ]
    try:
        available_models = ChatNVIDIA.get_available_models()
    except Exception:
        # Keep offline startup behavior useful; inference will return a clear
        # connection/model error if the configured model cannot be reached.
        available_models = []

    available_by_id = {model.id: model for model in available_models}
    selected_model = next(
        (model_id for model_id in dict.fromkeys(preferred_models) if model_id in available_by_id),
        None,
    )
    if selected_model is None and available_models:
        selected_model = available_models[0].id
    selected_model = selected_model or LLM_MODEL

    model_kwargs = {}
    if selected_model.startswith("nvidia/nemotron"):
        model_kwargs = {"chat_template_kwargs": {"enable_thinking": False}}

    return ChatNVIDIA(
        model=selected_model,
        temperature=1,
        max_tokens=LLM_MAX_TOKENS,
        model_kwargs=model_kwargs,
    )


def active_llm_model() -> str:
    """Return the model selected from NVIDIA's current available-model list."""
    return getattr(load_llm(), "model", None) or LLM_MODEL


# ============================================================
# Helpers
# ============================================================

def build_context(docs) -> tuple[str, List[Dict], List]:
    """Build model context and matching page links from exactly the same chunks."""
    parts, sources, source_ids, included_docs = [], [], {}, []
    used = 0

    for doc in docs:
        source = str(doc.metadata.get("source", ""))
        raw_page = doc.metadata.get("page")
        page = raw_page + 1 if isinstance(raw_page, int) and raw_page >= 0 else None
        filename = Path(source).name if source else ""
        key = (filename.lower(), page)

        _, law_name = law_info(source or "Unknown", METADATA_PATH)
        source_id = source_ids.get(key, f"S{len(sources) + 1}")
        page_label = f"Page {page}" if page is not None else ""
        chunk = f"[{source_id}] {law_name} {page_label}\n{doc.page_content.strip()}".strip()

        # Keep each passage whole so the model never cites a page for a claim
        # supported only by an accidentally cut-off sentence.
        separator_size = 2 if parts else 0
        if used + separator_size + len(chunk) > MAX_CONTEXT_CHARS:
            break

        if key not in source_ids:
            source_ids[key] = source_id
            can_open_pdf = bool(
                filename.lower().endswith(".pdf")
                and (DATA_PATH / "laws" / filename).is_file()
            )
            sources.append({
                "id": source_id,
                "label": f"{law_name} {page_label}".strip(),
                "filename": filename if can_open_pdf else None,
                "page": page if can_open_pdf else None,
            })

        parts.append(chunk)
        included_docs.append(doc)
        used += separator_size + len(chunk)

    return "\n\n".join(parts), sources, included_docs


def friendly_error(e: Exception) -> str:
    error = str(e)

    if "410" in error or "404" in error or "end of life" in error:
        return (
            "⚠️ NVIDIA could not serve the selected model. Restart the app to "
            "refresh NVIDIA's available-model list; the app will choose an "
            "available chat model automatically. If the problem continues, "
            "check the API key or NVIDIA service status."
        )
    if "Read timed out" in error or "timed out" in error:
        return "⚠️ NVIDIA API timed out. Please try again in a few seconds."
    if "401" in error or "403" in error:
        return "⚠️ NVIDIA API key is invalid or not authorised."
    if "429" in error:
        return "⚠️ NVIDIA API rate limit exceeded. Please wait a minute and try again."

    return f"Unexpected Error:\n\n{error}"


def _strip_thinking(chunks: Iterator[str]) -> Iterator[str]:
    """Hide <think>...</think> reasoning that some models put in the text."""
    buffer, state = "", "start"   # start -> thinking -> answer

    for text in chunks:
        if state == "answer":
            yield text
            continue

        buffer += text

        if state == "start":
            head = buffer.lstrip()
            if head.startswith("<think>"):
                state = "thinking"
            elif len(head) >= 7 or not "<think>".startswith(head):
                state = "answer"
                yield buffer
                buffer = ""
                continue
            else:
                continue

        if state == "thinking" and "</think>" in buffer:
            state = "answer"
            rest = buffer.split("</think>", 1)[1].lstrip()
            buffer = ""
            if rest:
                yield rest

    if state == "start" and buffer:
        yield buffer


# ============================================================
# Main RAG function
# ============================================================

def ask_legal_question_stream(question: str, history=None) -> Dict:
    """
    Returns {"stream": <generator of text>, "sources": [...], "cases": [...]}.
    Retrieval happens immediately; the answer streams as the model writes it.
    """
    try:
        retriever = load_retriever()
        llm = load_llm()

        history = history or []
        prior_user_questions = [
            str(message.get("content", "")).strip()
            for message in history
            if message.get("role") == "user" and str(message.get("content", "")).strip()
        ][-2:]
        follow_up_markers = (
            "that", "it", "this", "those", "these", "above", "you mentioned",
            "what does", "why is", "who was", "which case", "explain more",
            "tell me more", "what about", "what are its", "what is its",
            "what are the consequences", "what is the effect", "and the penalty",
        )
        is_follow_up = bool(prior_user_questions) and any(
            marker in question.lower() for marker in follow_up_markers
        )
        retrieval_query = question
        if is_follow_up:
            retrieval_query = "\n".join(
                [f"Earlier topic: {prior_user_questions[-1]}", f"Follow-up: {question}"]
            )
        docs = retriever.invoke(retrieval_query)

        if not docs:
            return {
                "stream": iter(
                    ["I could not find any relevant legal documents for your question."]
                ),
                "sources": [],
                "cases": [],
            }

        context, sources, context_docs = build_context(docs)
        if not context_docs:
            return {
                "stream": iter(
                    ["I could not find sufficient information in the provided legal documents."]
                ),
                "sources": [],
                "cases": [],
            }

        law_ids = {
            law_info(d.metadata.get("source", ""), METADATA_PATH)[0]
            for d in context_docs
        } - {None}
        cases = find_related_cases(retrieval_query, law_ids, METADATA_PATH, MAX_CASES)

        messages = RAG_PROMPT.format_messages(
            context=context,
            question=question,
            history=[
                (message["role"], str(message.get("content", "")))
                for message in (history[-6:] if is_follow_up else [])
                if message.get("role") in {"user", "assistant"}
            ],
        )

        def generate():
            try:
                def content_chunks():
                    for chunk in llm.stream(messages):
                        content = chunk.content
                        if isinstance(content, str) and content:
                            yield content
                        elif isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and isinstance(part.get("text"), str):
                                    yield part["text"]

                answer_text = "".join(_strip_thinking(content_chunks())).strip()
                if not answer_text:
                    yield (
                        "⚠️ NVIDIA returned an empty answer. Check that the selected "
                        f"model (`{active_llm_model()}`) is available to your API key, then retry."
                    )
                else:
                    yield answer_text
            except Exception as e:
                yield friendly_error(e)

        return {
            "stream": generate(),
            "sources": sources,
            "cases": [f"{c['title']} ({c['year']}) - {c['citation']}" for c in cases],
        }

    except Exception as e:
        return {"stream": iter([friendly_error(e)]), "sources": [], "cases": []}


def ask_legal_question(question: str) -> Dict:
    """Non-streaming version (CLI / notebooks)."""
    result = ask_legal_question_stream(question)
    result["answer"] = "".join(result.pop("stream")).strip()
    return result


# ============================================================
# Local testing
# ============================================================

if __name__ == "__main__":

    while True:
        query = input("\nAsk a legal question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        result = ask_legal_question(query)

        print("\n" + "=" * 80)
        print(result["answer"])
        print("=" * 80)

        print("\nSources:\n")
        for src in result["sources"]:
            print("-", src)
