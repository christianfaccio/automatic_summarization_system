import os
# Force HuggingFace Hub to use only locally cached files — no network calls.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import re
from collections import defaultdict
from io import StringIO
from pathlib import Path

from bert_score import score as _bert_score
from langchain_community.document_loaders import PyPDFDirectoryLoader

from ir import clean_text

DOCS_DIR      = Path("../docs")
IR_JSON       = Path("../output/info_regex.json")
NIR_JSON      = Path("../output/info_llm.json")
SUMMARIES_DIR = Path("../output/generated_summaries")
REPORT_PATH   = Path("../output/evaluation_report.txt")

FIELDS = [
    "academic_year",
    "total_budget_general_millions_eur",
    "income_thresholds_defined_by",
    "eligible_programs",
    "scholarship_components",
    "academic_requirements",
    "application_period",
]


# Scoring helpers

def _token_f1(pred: str, ref: str) -> float:
    """Bag-of-words F1 between two strings."""
    p_toks = set(str(pred).lower().split())
    r_toks = set(str(ref).lower().split())
    if not r_toks:
        return 1.0 if not p_toks else 0.0
    tp   = len(p_toks & r_toks)
    prec = tp / len(p_toks) if p_toks else 0.0
    rec  = tp / len(r_toks)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _set_f1(pred: list, ref: list) -> float:
    """Set-overlap F1 between two lists (case-insensitive)."""
    p_set = {s.lower().strip() for s in (pred or [])}
    r_set = {s.lower().strip() for s in (ref  or [])}
    if not r_set:
        return 1.0 if not p_set else 0.0
    tp   = len(p_set & r_set)
    prec = tp / len(p_set) if p_set else 0.0
    rec  = tp / len(r_set)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _numeric_score(pred, ref) -> float:
    """1 − relative_error, clipped to [0, 1]."""
    try:
        pred, ref = float(pred), float(ref)
    except (TypeError, ValueError):
        return 0.0
    if ref == 0:
        return 1.0 if pred == 0 else 0.0
    return max(0.0, 1.0 - abs(pred - ref) / abs(ref))


def _dict_numeric_score(pred: dict, ref: dict) -> float:
    """Average numeric score over all numeric keys in ref."""
    scores = []
    for k, rv in ref.items():
        pv = pred.get(k) if isinstance(pred, dict) else None
        if pv is None:
            scores.append(0.0)
        elif isinstance(rv, dict):
            sub = [_numeric_score(pv.get(gk), av) for gk, av in rv.items()]
            scores.append(sum(sub) / len(sub) if sub else 0.0)
        elif isinstance(rv, (int, float)):
            scores.append(_numeric_score(pv, rv))
        else:
            scores.append(1.0 if str(pv).strip() == str(rv).strip() else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def score_field(pred, ref, field: str):
    if ref is None:
        return None, "regex did not capture"
    if pred is None:
        return 0.0, None

    if field == "academic_year":
        return (1.0 if str(pred).strip() == str(ref).strip() else 0.0), None

    if field == "total_budget_general_millions_eur":
        return _numeric_score(pred, ref), None

    if field == "income_thresholds_defined_by":
        return _token_f1(pred, ref), None

    if field == "eligible_programs":
        if not isinstance(ref, dict):
            return None, "unexpected format in regex output"
        scores = []
        for sub in ("university", "non_university"):
            rv = ref.get(sub)
            if rv is not None:
                pv = pred.get(sub) if isinstance(pred, dict) else []
                scores.append(_set_f1(pv or [], rv))
        return (sum(scores) / len(scores) if scores else None), None

    if field == "scholarship_components":
        if not isinstance(ref, dict):
            return None, "unexpected format in regex output"
        return _dict_numeric_score(pred if isinstance(pred, dict) else {}, ref), None

    if field == "academic_requirements":
        if not isinstance(ref, dict):
            return None, "unexpected format in regex output"
        scores = []
        for k, rv in ref.items():
            pv = pred.get(k) if isinstance(pred, dict) else None
            if pv is None:
                scores.append(0.0)
            elif k == "min_gpa":
                scores.append(1.0 if abs(float(pv) - float(rv)) < 0.05 else 0.0)
            else:
                scores.append(1.0 if int(pv) == int(rv) else 0.0)
        return (sum(scores) / len(scores) if scores else None), None

    if field == "application_period":
        if not isinstance(ref, dict):
            return None, "unexpected format in regex output"
        scores = [
            1.0 if (pred.get(k) if isinstance(pred, dict) else None) == rv else 0.0
            for k, rv in ref.items()
        ]
        return (sum(scores) / len(scores) if scores else None), None

    return (1.0 if pred == ref else 0.0), None


# Evaluation of structured data

def evaluate_structured(out: StringIO):
    def p(line=""):
        print(line)
        out.write(line + "\n")

    p("\n" + "=" * 60)
    p("Evaluating structured extracted data")
    p("=" * 60)

    if not IR_JSON.exists():
        p(f"  {IR_JSON} not found — run ir.py first.")
        return
    if not NIR_JSON.exists():
        p(f"  {NIR_JSON} not found — run nir.py first.")
        return

    with open(IR_JSON, encoding="utf-8") as f:
        ir_raw = json.load(f)
    with open(NIR_JSON, encoding="utf-8") as f:
        nir_raw = json.load(f)

    ir_by_year = {
        entry["academic_year"]: entry
        for entry in ir_raw.values()
        if entry.get("academic_year")
    }
    nir_by_year = {}
    for entry in (nir_raw if isinstance(nir_raw, list) else nir_raw.values()):
        yr = entry.get("academic_year")
        if yr:
            nir_by_year[yr] = entry

    common = sorted(set(ir_by_year) & set(nir_by_year))
    if not common:
        p("  No matching academic years found between the two outputs.")
        p(f"      ir.py years : {sorted(ir_by_year)}")
        p(f"      nir.py years: {sorted(nir_by_year)}")
        return

    field_scores  = defaultdict(list)
    field_skipped = defaultdict(int)

    for year in common:
        ir_entry  = ir_by_year[year]
        nir_entry = nir_by_year[year]
        for field in FIELDS:
            s, reason = score_field(nir_entry.get(field), ir_entry.get(field), field)
            if reason:
                field_skipped[field] += 1
            elif s is not None:
                field_scores[field].append(s)

    W = 42
    p(f"\n  {'Field':<{W}} {'Score':>7}  {'N docs':>6}  {'Skipped (regex=None)':>22}")
    p("  " + "─" * (W + 42))
    for field in FIELDS:
        scores  = field_scores[field]
        skipped = field_skipped[field]
        if scores:
            avg = sum(scores) / len(scores)
            p(f"  {field:<{W}} {avg:>7.3f}  {len(scores):>6}  {skipped:>22}")
        else:
            p(f"  {field:<{W}} {'—':>7}  {'0':>6}  {skipped:>22}")

    scored_fields = [f for f in FIELDS if field_scores[f]]
    if scored_fields:
        overall = sum(sum(field_scores[f]) / len(field_scores[f]) for f in scored_fields) / len(scored_fields)
        p(f"\n  Overall average (scored fields only): {overall:.3f}")


# Helpers

def _extract_pdf_texts(docs_dir: Path) -> dict:
    """Return {filename: body_text} using the same pipeline as ir.py."""
    loader = PyPDFDirectoryLoader(str(docs_dir))
    raw_docs = loader.load()

    source_pages = defaultdict(list)
    for doc in raw_docs:
        doc.page_content = clean_text(doc.page_content)
        source_pages[doc.metadata["source"]].append(doc)

    texts = {}
    for source_path, pages in source_pages.items():
        filename = os.path.basename(source_path)
        pages.sort(key=lambda d: d.metadata.get("page", 0))
        texts[filename] = "\n".join(p.page_content for p in pages)
    return texts


def _year_from_text(text: str):
    m = re.search(r'CURSO ACAD[EÉ]MICO\s+(\d{4}[-–]\d{2,4})', text, re.IGNORECASE)
    return m.group(1) if m else None


def _load_summaries(summaries_dir: Path) -> dict:
    """Return {(year, model_tag): text} from {year}_{model_tag}.txt files."""
    result = {}
    for path in sorted(summaries_dir.glob("*.txt")):
        stem  = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            year, model_tag = parts
            result[(year, model_tag)] = path.read_text(encoding="utf-8")
    return result


# Evaluating summaries against original documents

def evaluate_summaries(out: StringIO):
    def p(line=""):
        print(line)
        out.write(line + "\n")

    p("\n" + "=" * 60)
    p("Evaluating summaries against original documents.")
    p("  BERTScore: semantic overlap (XLM-RoBERTa, cross-lingual)")
    p("=" * 60)

    if not SUMMARIES_DIR.exists() or not any(SUMMARIES_DIR.glob("*.txt")):
        p(f"  No summaries found in {SUMMARIES_DIR} — run summary.py first.")
        return
    if not DOCS_DIR.exists():
        p(f"  {DOCS_DIR} not found.")
        return

    pdf_texts    = _extract_pdf_texts(DOCS_DIR)
    year_to_text = {}
    for filename, text in pdf_texts.items():
        yr = _year_from_text(text)
        if yr:
            year_to_text[yr] = text
        else:
            p(f"  Could not detect academic year in {filename} — skipped.")

    summaries  = _load_summaries(SUMMARIES_DIR)
    model_tags = sorted({mtag for _, mtag in summaries})

    for model_tag in model_tags:
        pairs = [
            (year, summaries[(year, model_tag)], year_to_text[year])
            for (year, mtag) in summaries
            if mtag == model_tag and year in year_to_text
        ]
        if not pairs:
            p(f"\n  [{model_tag}] No matching PDF texts found — skipped.")
            continue

        years = [pr[0] for pr in pairs]
        hyps  = [pr[1] for pr in pairs]
        refs  = [pr[2] for pr in pairs]

        _, _, bs_f1 = _bert_score(
            hyps, refs,
            model_type="xlm-roberta-base",
            verbose=False,
        )

        p(f"\n  Model: {model_tag}")
        p(f"  {'Year':<15} {'BERTScore-F1':>13}")
        p(f"  {'─' * 30}")
        for i, year in enumerate(years):
            p(f"  {year:<15} {bs_f1[i].item():>13.3f}")
        p(f"  {'AVERAGE':<15} {bs_f1.mean().item():>13.3f}")

if __name__ == "__main__":
    buf = StringIO()
    evaluate_structured(buf)
    evaluate_summaries(buf)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(buf.getvalue(), encoding="utf-8")
