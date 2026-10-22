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
    return ChatNVIDIA(
        model=LLM_MODEL,
        # This app needs a short direct answer; the API otherwise defaults to
        # spending tokens on a reasoning trace before returning the answer.
        temperature=1,
        max_tokens=LLM_MAX_TOKENS,
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )


# ============================================================
# Helpers
# ============================================================

def _page_label(doc) -> str:
    page = doc.metadata.get("page")
    return f"Page {page + 1}" if isinstance(page, int) else ""


def build_context(docs) -> str:
    """Label each chunk with its law + page so the model can cite correctly."""
    parts, used = [], 0

    for i, doc in enumerate(docs, 1):
        _, law_name = law_info(doc.metadata.get("source", "Unknown"), METADATA_PATH)
        header = f"[{i}] {law_name} {_page_label(doc)}".strip()
        chunk = f"{header}\n{doc.page_content.strip()}"

        remaining = MAX_CONTEXT_CHARS - used
        if remaining < 300:
            break
        chunk = chunk[:remaining]

        parts.append(chunk)
        used += len(chunk) + 2

    return "\n\n".join(parts)


def extract_sources(docs) -> List[Dict]:
    """Build page-level source links in the order documents were retrieved."""
    grouped: Dict[str, Dict] = {}

    for doc in docs:
        source = doc.metadata.get("source", "")
        _, law_name = law_info(source or "Unknown", METADATA_PATH)
        page = doc.metadata.get("page")
        group = grouped.setdefault(law_name, {"pages": set(), "filename": None})
        filename = Path(str(source)).name
        if filename.lower().endswith(".pdf") and (DATA_PATH / "laws" / filename).is_file():
            group["filename"] = filename
        if isinstance(page, int) and page >= 0:
            group["pages"].add(page + 1)

    sources = []
    for law_name, info in grouped.items():
        for page in sorted(info["pages"]):
            sources.append({
                "label": f"{law_name} — Page {page}",
                "filename": info["filename"],
                "page": page,
            })
        if not info["pages"]:
            sources.append({"label": law_name, "filename": info["filename"], "page": None})
    return sources


def friendly_error(e: Exception) -> str:
    error = str(e)

    if "410" in error or "404" in error or "end of life" in error:
        return (
            "⚠️ The configured NVIDIA model is no longer available.\n\n"
            f"Current model: `{LLM_MODEL}`\n\n"
            "Run `python list_models.py`, pick a live model, and set it in "
            "`config.py` (or `LLM_MODEL=` in `.env`), then restart the app."
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

        law_ids = {
            law_info(d.metadata.get("source", ""), METADATA_PATH)[0] for d in docs
        } - {None}
        cases = find_related_cases(retrieval_query, law_ids, METADATA_PATH, MAX_CASES)

        messages = RAG_PROMPT.format_messages(
            context=build_context(docs),
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
                        "⚠️ NVIDIA returned an empty answer. Check that the configured "
                        f"model (`{LLM_MODEL}`) is available to your API key, then retry."
                    )
                else:
                    yield answer_text
            except Exception as e:
                yield friendly_error(e)

        return {
            "stream": generate(),
            "sources": extract_sources(docs),
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
