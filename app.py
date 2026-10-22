import csv
import hashlib
from html import escape

import streamlit as st

from case_summary import (
    answer_case_question,
    build_case_qa_index,
    extract_case_pages,
    summarize_case_pdf,
)
from config import DATA_PATH, METADATA_PATH
from pdf_server import pdf_page_url, register_uploaded_pdf, uploaded_pdf_page_url
from rag import active_llm_model, ask_legal_question_stream


st.set_page_config(
    page_title="Indian Legal Research Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp { background: var(--st-background-color); color: var(--st-text-color); }
[data-testid="stSidebar"] {
    background: var(--st-secondary-background-color);
    border-right: 1px solid var(--st-border-color);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--st-text-color); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--st-text-color); opacity: .82; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap: 8px; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
    background: var(--st-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 12px;
    padding: 11px 13px;
    transition: background .18s ease, border-color .18s ease, transform .18s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
    background: color-mix(in srgb, var(--st-primary-color) 10%, var(--st-background-color));
    border-color: var(--st-primary-color);
    transform: translateX(2px);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
    background: color-mix(in srgb, var(--st-primary-color) 16%, var(--st-background-color));
    border-color: var(--st-primary-color);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p {
    color: var(--st-text-color);
    font-weight: 650;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color: var(--st-primary-color); }
.page-hero {
    position: relative; overflow: hidden;
    padding: 28px 32px; margin: 4px 0 22px;
    border: 1px solid var(--st-border-color); border-radius: 20px;
    background: linear-gradient(115deg, var(--st-secondary-background-color) 0%, var(--st-background-color) 70%, color-mix(in srgb, var(--st-primary-color) 8%, var(--st-background-color)) 100%);
    box-shadow: 0 12px 32px rgba(22, 38, 58, .07);
}
.page-hero:after {
    content: ''; position: absolute; width: 230px; height: 230px; right: -70px; top: -95px;
    border-radius: 50%; border: 1px solid color-mix(in srgb, var(--st-primary-color) 38%, transparent);
    box-shadow: 0 0 0 22px color-mix(in srgb, var(--st-primary-color) 5%, transparent), 0 0 0 46px color-mix(in srgb, var(--st-primary-color) 3%, transparent);
}
.hero-kicker { color: var(--st-primary-color); text-transform: uppercase; letter-spacing: .16em; font-size: 11px; font-weight: 750; }
.hero-title { color: var(--st-text-color); font-size: clamp(30px, 3.4vw, 42px); line-height: 1.1; font-weight: 760; margin: 7px 0 8px; }
.hero-copy { color: var(--st-text-color); opacity: .78; font-size: 15px; margin: 0; max-width: 760px; }
[data-testid="stChatMessage"] { background: var(--st-secondary-background-color); color: var(--st-text-color); border: 1px solid var(--st-border-color); border-radius: 16px; box-shadow: 0 5px 16px rgba(22,38,58,.035); }
[data-testid="stExpander"] { background: var(--st-secondary-background-color); border: 1px solid var(--st-border-color); border-radius: 13px; }
[data-testid="stFileUploaderDropzone"] { background: var(--st-secondary-background-color); border: 1px dashed var(--st-border-color); border-radius: 16px; }
div.stButton > button[kind="primary"] { background: var(--st-primary-color); border-color: var(--st-primary-color); }
div.stButton > button[kind="primary"]:hover { filter: brightness(.92); }
</style>
    """,
    unsafe_allow_html=True,
)

NAV_ITEMS = ["Ask AI", "Data Store", "Case File Summary"]
NAV_LABELS = {
    "Ask AI": "💬  Ask AI",
    "Data Store": "📚  Data Store",
    "Case File Summary": "⚖️  Case File Summary",
}
EXAMPLES = [
    "What is the right to education?",
    "What are the grounds for divorce under the Hindu Marriage Act?",
    "What is the right to privacy?",
    "What is sexual harassment at the workplace?",
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_view" not in st.session_state:
    st.session_state.active_view = NAV_ITEMS[0]
elif st.session_state.active_view not in NAV_ITEMS:
    st.session_state.active_view = NAV_ITEMS[0]


def page_header(kicker, title, description):
    st.markdown(
        f'<section class="page-hero"><div class="hero-kicker">{escape(kicker)}</div>'
        f'<div class="hero-title">{escape(title)}</div>'
        f'<p class="hero-copy">{escape(description)}</p></section>',
        unsafe_allow_html=True,
    )


def show_sources_and_cases(sources, cases):
    if cases:
        with st.expander("⚖️ Related landmark cases"):
            for case in cases:
                st.write(f"• {case}")

    if sources:
        with st.expander("📚 Sources"):
            for source in sources:
                label = escape(source["label"])
                source_id = escape(source.get("id", ""))
                linked_label = f"[{source_id}] {label}" if source_id else label
                if source.get("filename") and source.get("page"):
                    url = escape(pdf_page_url(source["filename"], source["page"]), quote=True)
                    st.markdown(
                        f'<a href="{url}" target="_blank" rel="noopener noreferrer">📄 {linked_label} ↗</a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.write(f"• {linked_label}")


def render_ask_ai():
    page_header(
        "Legal research workspace",
        "Ask AI",
        "Ask a question and get a concise answer grounded in the laws in your data store, with relevant cases and page-linked sources.",
    )

    pending_question = None
    with st.sidebar:
        st.markdown("**Try an example**")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True, key=f"example-{example}"):
                pending_question = example
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources_and_cases(message.get("sources", []), message.get("cases", []))

    question = st.chat_input("Ask your legal question...") or pending_question
    if question:
        # Keep recent turns available for follow-ups such as “what does that mean?”
        history = st.session_state.messages[-6:]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching legal database and generating your answer..."):
                result = ask_legal_question_stream(question, history=history)
            with st.spinner("Writing a concise, source-grounded answer..."):
                answer = st.write_stream(result["stream"])
            show_sources_and_cases(result["sources"], result["cases"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": result["sources"],
                "cases": result["cases"],
            }
        )


def load_law_files():
    metadata_by_file = {}
    laws_csv = METADATA_PATH / "laws.csv"
    if laws_csv.is_file():
        with open(laws_csv, encoding="utf-8-sig", newline="") as law_file:
            for row in csv.DictReader(law_file):
                metadata_by_file[row.get("file_name", "").strip().lower()] = row

    records = []
    for path in sorted((DATA_PATH / "laws").glob("*.pdf"), key=lambda item: item.name.lower()):
        metadata = metadata_by_file.get(path.name.lower(), {})
        records.append(
            {
                "name": metadata.get("law_name", "").strip() or path.stem.replace("_", " ").title(),
                "year": metadata.get("year", "").strip(),
                "category": metadata.get("category", "").strip(),
                "filename": path.name,
                "size_mb": path.stat().st_size / (1024 * 1024),
            }
        )
    return records


def render_data_store():
    page_header(
        "Your research library",
        "Data Store",
        "Browse the law PDFs available to the assistant and open any source document directly.",
    )

    laws = load_law_files()
    search = st.text_input("Search laws", placeholder="Search by name, year, or category")
    query = search.strip().lower()
    visible_laws = [
        law for law in laws
        if not query or query in " ".join(str(value) for value in law.values()).lower()
    ]
    st.caption(f"Showing {len(visible_laws)} of {len(laws)} law files")

    if not visible_laws:
        st.info("No law files match that search.")
        return

    card_columns = st.columns(2, gap="medium")
    for index, law in enumerate(visible_laws):
        url = escape(pdf_page_url(law["filename"], 1), quote=True)
        with card_columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {escape(law['name'])}")
                details = " · ".join(value for value in [law["year"], law["category"]] if value)
                st.caption(details or "Legal document")
                left, right = st.columns([2, 1])
                left.caption(f"PDF · {law['size_mb']:.1f} MB")
                right.markdown(
                    f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open PDF ↗</a>',
                    unsafe_allow_html=True,
                )


def render_case_summary():
    page_header(
        "Case document analysis",
        "Case File Summary",
        "Upload a case PDF to extract the facts, issues, decision, and outcome into a concise, page-referenced summary.",
    )

    uploaded_file = st.file_uploader(
        "Choose a case PDF",
        type=["pdf"],
        key="case-pdf-upload",
        help="Long documents are processed in sections. Scanned image-only PDFs need OCR first.",
    )
    if uploaded_file is None:
        st.info("Upload a PDF to summarize its facts, issues, decision, and outcome.")
        return

    file_hash = hashlib.sha256()
    uploaded_file.seek(0)
    while True:
        block = uploaded_file.read(1024 * 1024)
        if not block:
            break
        file_hash.update(block)
    uploaded_file.seek(0)
    file_hash = file_hash.hexdigest()
    if st.session_state.get("case_qa_file_hash") != file_hash:
        st.session_state.case_qa_file_hash = file_hash
        st.session_state.case_qa_index = None
        st.session_state.case_qa_messages = []
    st.caption(f"Selected file: {uploaded_file.name} · {uploaded_file.size / (1024 * 1024):.1f} MB")
    st.caption("Long PDFs are split into sections, summarized, then combined. Large files take longer and use more model requests.")

    if st.button("Generate case summary", type="primary", key="generate-case-summary"):
        progress = st.progress(0, text="Preparing case PDF")
        try:
            summary = summarize_case_pdf(
                uploaded_file,
                on_progress=lambda fraction, label: progress.progress(
                    min(1.0, max(0.0, fraction)), text=label
                ),
            )
            st.session_state.case_summary_result = {
                "file_hash": file_hash,
                "file_name": uploaded_file.name,
                "summary": summary,
            }
        except Exception as error:
            st.error(str(error))
        finally:
            progress.empty()

    saved_summary = st.session_state.get("case_summary_result")
    if saved_summary and saved_summary["file_hash"] == file_hash:
        st.subheader(f"Summary: {saved_summary['file_name']}")
        st.markdown(saved_summary["summary"])

    st.divider()
    st.subheader("Ask about this case")
    st.caption("Ask a question about the uploaded judgment. Answers use only this case and cite its page numbers.")

    for message in st.session_state.case_qa_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("pages"):
                with st.expander("📄 Cited pages in this case"):
                    token = register_uploaded_pdf(uploaded_file.getvalue(), uploaded_file.name)
                    for page in message["pages"]:
                        url = escape(uploaded_pdf_page_url(token, page), quote=True)
                        st.markdown(
                            f'<a href="{url}" target="_blank" rel="noopener noreferrer">📄 Page {page} ↗</a>',
                            unsafe_allow_html=True,
                        )

    case_question = st.chat_input("Ask a question about this uploaded case...", key="case-qa-input")
    if case_question:
        history = st.session_state.case_qa_messages[-6:]
        st.session_state.case_qa_messages.append({"role": "user", "content": case_question})
        with st.chat_message("user"):
            st.markdown(case_question)

        with st.chat_message("assistant"):
            try:
                if st.session_state.case_qa_index is None:
                    with st.spinner("Reading the case and preparing page-level search…"):
                        pages = extract_case_pages(uploaded_file.getvalue())
                        st.session_state.case_qa_index = build_case_qa_index(pages, uploaded_file.name)
                with st.spinner("Searching this case and writing a cited answer…"):
                    result = answer_case_question(
                        case_question,
                        st.session_state.case_qa_index,
                        history=history,
                    )
                answer = result["answer"]
                st.markdown(answer)
                if result["pages"]:
                    with st.expander("📄 Cited pages in this case"):
                        token = register_uploaded_pdf(uploaded_file.getvalue(), uploaded_file.name)
                        for page in result["pages"]:
                            url = escape(uploaded_pdf_page_url(token, page), quote=True)
                            st.markdown(
                                f'<a href="{url}" target="_blank" rel="noopener noreferrer">📄 Page {page} ↗</a>',
                                unsafe_allow_html=True,
                            )
                st.session_state.case_qa_messages.append(
                    {"role": "assistant", "content": answer, "pages": result["pages"]}
                )
            except Exception as error:
                error_message = str(error)
                st.error(error_message)
                st.session_state.case_qa_messages.append(
                    {"role": "assistant", "content": f"I could not answer from this case: {error_message}", "pages": []}
                )


with st.sidebar:
    st.title("⚖️ Legal Research AI")
    st.caption("INDIAN LAW · RESEARCH ASSISTANT")
    st.markdown("**WORKSPACE**")
    st.radio(
        "Project sections",
        NAV_ITEMS,
        format_func=lambda item: NAV_LABELS[item],
        key="active_view",
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"LLM: `{active_llm_model()}`")

if st.session_state.active_view == "Ask AI":
    render_ask_ai()
elif st.session_state.active_view == "Data Store":
    render_data_store()
else:
    render_case_summary()
