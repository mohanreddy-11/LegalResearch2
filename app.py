import streamlit as st

from config import LLM_MODEL
from rag import ask_legal_question_stream

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Indian Legal Research Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ------------------------------------------
# Session State
# ------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

EXAMPLES = [
    "What is the right to education?",
    "What are the grounds for divorce under the Hindu Marriage Act?",
    "What is the right to privacy?",
    "What is sexual harassment at the workplace?",
]

pending = None

# ------------------------------------------
# Sidebar
# ------------------------------------------

with st.sidebar:
    st.title("⚖️ Legal Research AI")

    st.markdown("---")

    st.markdown(
        f"""
### Features

✅ Ask Questions on Indian Laws

🚧 Analyze Case Documents (Coming Soon)

📚 Uses Retrieval Augmented Generation (RAG)

🤖 Model: `{LLM_MODEL}`
"""
    )

    st.markdown("---")
    st.markdown("**Try an example**")

    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            pending = example

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------
# Main Title
# ------------------------------------------

st.title("⚖️ Indian Legal Research Assistant")

st.caption(
    "Ask questions on the Constitution, BNS, BSA, IT Act, Companies Act, "
    "Hindu Marriage & Succession Acts, Consumer Protection, IBC, RERA, DPDP and more."
)

st.divider()

# ------------------------------------------
# Display Previous Messages
# ------------------------------------------

def show_extras(sources, cases):
    if cases:
        with st.expander("⚖️ Related landmark cases"):
            for c in cases:
                st.write(f"• {c}")
    if sources:
        with st.expander("📚 Sources"):
            for s in sources:
                st.write(f"• {s}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_extras(msg.get("sources"), msg.get("cases"))

# ------------------------------------------
# Chat Input
# ------------------------------------------

question = st.chat_input("Ask your legal question...") or pending

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching legal database..."):
            result = ask_legal_question_stream(question)

        answer = st.write_stream(result["stream"])

        show_extras(result["sources"], result["cases"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result["sources"],
            "cases": result["cases"],
        }
    )
