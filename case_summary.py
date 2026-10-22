"""Text extraction and grounded summaries for uploaded case PDFs."""

from typing import Callable, List, Optional

from rag import load_llm


CASE_SUMMARY_SYSTEM = """You are a careful legal case summarizer. Use only the uploaded document text. Do not add outside law or assume facts that are not stated. Distinguish the parties' arguments from the court's findings and final order. Cite page numbers using the [Page N] markers supplied in the text. If a detail is absent, say it is not stated in the provided text.

Return a concise summary with these headings:
**Case and court**
**Facts and procedural history**
**Issues and decision**
**Outcome and important provisions**

Keep the full summary to about 200-300 words.
"""

CHUNK_SYSTEM = """Extract only material facts from this excerpt of a legal case. Preserve page references. In 3-5 compact bullets, capture the parties, procedural posture, issues, arguments, findings, cited provisions, and any order that appear in this excerpt. Clearly label arguments versus findings. Do not infer or add facts. Keep this excerpt summary under 120 words."""

REDUCE_SYSTEM = """Condense the supplied excerpt summaries into 3-5 compact bullets. Preserve page references and distinguish party arguments from the court's findings and final order. Do not add facts or legal conclusions. Keep the result under 180 words."""


def _extract_pdf_pages(pdf_file, on_progress: Optional[Callable] = None) -> List[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF summaries need pypdf. Install project requirements and restart the app.") from exc

    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    pages = []
    total_pages = len(reader.pages)
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[Page {page_number}]\n{text}")
        if on_progress and total_pages:
            on_progress(0.2 * page_number / total_pages, f"Reading PDF page {page_number} of {total_pages}")
    if not pages:
        raise ValueError("No selectable text was found in this PDF. Scanned PDFs need OCR before summarizing.")
    return pages


def _chunks(pages: List[str], max_chars: int = 24000) -> List[str]:
    chunks = []
    current = ""
    for page_text in pages:
        page_marker, page_body = page_text.split("\n", 1)
        body_limit = max_chars - len(page_marker) - 1
        segments = [
            f"{page_marker}\n{page_body[i:i + body_limit]}"
            for i in range(0, len(page_body), body_limit)
        ]
        for segment in segments:
            if current and len(current) + len(segment) + 2 > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current}\n\n{segment}".strip()
    if current:
        chunks.append(current)
    return chunks


def _invoke_text(llm, system_prompt: str, user_text: str) -> str:
    response = llm.invoke([("system", system_prompt), ("human", user_text)])
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            part["text"] for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ).strip()
    return ""


def _group_summaries(summaries: List[str], max_chars: int = 12000) -> List[str]:
    groups = []
    current = []
    current_size = 0
    for summary in summaries:
        if current and current_size + len(summary) > max_chars:
            groups.append("\n\n".join(current))
            current, current_size = [], 0
        current.append(summary)
        current_size += len(summary)
    if current:
        groups.append("\n\n".join(current))
    return groups


def summarize_case_pdf(pdf_file, on_progress: Optional[Callable] = None) -> str:
    """Summarize a seekable PDF stream with map/reduce for long documents."""
    pages = _extract_pdf_pages(pdf_file, on_progress)
    text_chunks = _chunks(pages)
    llm = load_llm()

    if len(text_chunks) == 1:
        document_text = text_chunks[0]
    else:
        chunk_summaries = []
        for index, text in enumerate(text_chunks, start=1):
            if on_progress:
                fraction = 0.2 + 0.5 * index / len(text_chunks)
                on_progress(fraction, f"Summarizing section {index} of {len(text_chunks)}")
            excerpt_summary = _invoke_text(
                llm,
                CHUNK_SYSTEM,
                f"Excerpt {index} of {len(text_chunks)}:\n\n{text}",
            )
            if excerpt_summary:
                chunk_summaries.append(excerpt_summary)

        if not chunk_summaries:
            raise RuntimeError("NVIDIA returned no section summaries. Check the model/API settings and try again.")

        reduction_round = 0
        while sum(map(len, chunk_summaries)) > 24000 and len(chunk_summaries) > 1:
            reduction_round += 1
            groups = _group_summaries(chunk_summaries)
            reduced = []
            for index, group in enumerate(groups, start=1):
                if on_progress:
                    fraction = min(0.88, 0.72 + 0.04 * reduction_round)
                    on_progress(fraction, f"Condensing long-document summaries ({index} of {len(groups)})")
                condensed = _invoke_text(llm, REDUCE_SYSTEM, group)
                if condensed:
                    reduced.append(condensed)
            if not reduced or len(reduced) >= len(chunk_summaries):
                break
            chunk_summaries = reduced
        document_text = "\n\n".join(chunk_summaries)

    if on_progress:
        on_progress(0.92, "Writing the final case summary")
    summary = _invoke_text(
        llm,
        CASE_SUMMARY_SYSTEM,
        f"Summarize this uploaded case document:\n\n{document_text}",
    )
    if summary:
        if on_progress:
            on_progress(1.0, "Summary complete")
        return summary
    raise RuntimeError("The model returned an empty case summary. Please retry or check the NVIDIA model/API settings.")
