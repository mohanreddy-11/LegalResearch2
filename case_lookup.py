"""
case_lookup.py
Pure-python helpers (no LangChain / Streamlit) that use the CSV metadata to
  1. turn a PDF filename into a readable law name, and
  2. find landmark cases related to the question.
"""

import csv
import re
from functools import lru_cache
from pathlib import Path

# laws.csv labels contract_labour_act.pdf as the "Indian Contract Act", but the
# PDF is actually the Contract Labour (Regulation and Abolition) Act, 1970.
SOURCE_OVERRIDES = {
    "contract_labour_act.pdf": (
        None,
        "Contract Labour (Regulation and Abolition) Act, 1970",
    ),
}

# Words too generic to prove a case is relevant to a question.
GENERIC = {
    "right", "rights", "law", "laws", "legal", "india", "indian", "union",
    "state", "court", "supreme", "under", "what", "which", "when", "where",
    "about", "explain", "section", "article", "provision", "provisions",
    "does", "that", "this", "with", "from", "have", "other", "their",
    "there", "between", "difference", "meaning", "define", "definition",
    "ltd", "private", "company", "home", "secretary", "delhi", "administration",
    "government", "act", "acts", "case", "cases", "mean", "means",
}


def _read_csv(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def load_metadata(metadata_dir: str):
    base = Path(metadata_dir)
    laws = _read_csv(base / "laws.csv")
    cases = _read_csv(base / "cases.csv")
    mapping = _read_csv(base / "law_case_mapping.csv")

    files = {}
    for row in laws:
        files[row["file_name"].strip().lower()] = (
            row["law_id"].strip(),
            row["law_name"].strip(),
        )
    files.update(SOURCE_OVERRIDES)

    cases_by_id = {row["id"].strip(): row for row in cases}

    law_cases = {}
    for row in mapping:
        law_cases.setdefault(row["law_id"].strip(), []).append(
            (row["case_id"].strip(), row["legal_topic"].strip())
        )

    return files, cases_by_id, law_cases


def law_info(source: str, metadata_dir) -> tuple:
    """('L001', 'Constitution of India') for 'constitution.pdf'."""
    files, _, _ = load_metadata(str(metadata_dir))
    name = Path(source).name.lower()
    if name in files:
        return files[name]
    return (None, Path(source).stem.replace("_", " ").title())


def _stems(text: str) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    return {w[:5] for w in words if len(w) >= 4 and w not in GENERIC}


def find_related_cases(question: str, law_ids, metadata_dir, limit: int = 3):
    """
    Landmark cases that (a) are mapped to the retrieved laws and
    (b) share a meaningful keyword with the question.
    Returns [] when nothing matches, so the model is never pushed to
    invent case law.
    """
    _, cases_by_id, law_cases = load_metadata(str(metadata_dir))
    q = _stems(question)
    if not q:
        return []

    found = {}
    for law_id in law_ids:
        for case_id, legal_topic in law_cases.get(law_id, []):
            case = cases_by_id.get(case_id)
            if not case or case_id in found:
                continue
            text = f"{case['title']} {case['topic']} {legal_topic}"
            overlap = q & _stems(text)
            if overlap:
                found[case_id] = (len(overlap), case)

    if not found:
        return []

    # keep only the strongest keyword matches
    best = max(score for score, _ in found.values())
    ranked = sorted(
        (v for v in found.values() if v[0] == best), key=lambda x: x[1]["id"]
    )[:limit]
    return [
        {
            "title": c["title"],
            "year": c["year"],
            "court": c["court"],
            "citation": c["citation"],
            "topic": c["topic"],
        }
        for _, c in ranked
    ]


def format_cases(cases) -> str:
    if not cases:
        return "None"
    return "\n".join(
        f"- {c['title']} ({c['year']}), {c['court']}, {c['citation']} - topic: {c['topic']}"
        for c in cases
    )
