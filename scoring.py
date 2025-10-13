"""
scoring.py — Milestone 2 refined
--------------------------------
- Partial matches (token overlap / fuzzy-ish)
- Missing steps detection
- Section-wise caps and weights
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import re

SECTION_MAX = {
    "trust_relationship": 30,
    "clarity_accuracy": 40,
    "value_upsell": 30,
}

# KPIs with indicative phrases (English side — we score on translated EN)
KPI_KEYWORDS = {
    "trust_relationship": {
        "Greeting": ["good morning", "good afternoon", "hello", "hi"],
        "Introduction": ["my name is", "this is", "speaking"],
        "Company Presentation": ["from", "company", "on behalf of"],
    },
    "clarity_accuracy": {
        "Product Mention": ["product", "item", "model", "chainsaw", "drill", "order"],
        "Address Confirmation": ["address", "street", "house number", "postcode", "zip", "city", "province"],
        "Recap": ["recap", "summary", "to confirm", "confirm once more"],
        "Tone of Voice": ["please", "thank you", "kindly", "appreciate"],
    },
    "value_upsell": {
        "Upsell Product": ["second product", "another model", "upgrade", "bundle"],
        "Warranty Offer": ["warranty", "guarantee", "extended", "protection"],
    },
}

# Points per KPI (sub-scores)
SUBPOINTS = {
    "trust_relationship": {"Greeting": 10, "Introduction": 10, "Company Presentation": 10},
    "clarity_accuracy": {"Product Mention": 10, "Address Confirmation": 15, "Recap": 10, "Tone of Voice": 5},
    "value_upsell": {"Upsell Product": 20, "Warranty Offer": 10},
}

def _normalize(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return f" {t} "

def _token_overlap(text: str, phrase: str) -> float:
    """
    Lightweight partial-match: compute overlap ratio between tokens in `phrase` and `text`.
    Returns [0..1].
    """
    text_tokens = set(re.findall(r"[a-z0-9]+", text))
    phrase_tokens = set(re.findall(r"[a-z0-9]+", phrase.lower()))
    if not phrase_tokens:
        return 0.0
    inter = len(text_tokens & phrase_tokens)
    return inter / len(phrase_tokens)

def _kpi_score_for(text: str, keywords: List[str], full_points: int) -> Tuple[int, float]:
    """
    Give full points if any keyword appears exactly.
    Else if partial token overlap >= 0.5 ⇒ 60% credit.
    Else if overlap >= 0.3 ⇒ 30% credit.
    """
    for kw in keywords:
        if f" {kw} " in text:
            return full_points, 1.0
    # partial matches
    best = 0.0
    for kw in keywords:
        best = max(best, _token_overlap(text, kw))
    if best >= 0.5:
        return int(round(full_points * 0.6)), best
    if best >= 0.3:
        return int(round(full_points * 0.3)), best
    return 0, best

def score_text(english_text: str) -> Tuple[int, Dict[str, int], List[str], str]:
    """
    Score translated-English transcript.
    Returns: total, by_section, missing_kpis, comment
    """
    text = _normalize(english_text)
    by_section = {s: 0 for s in SECTION_MAX}
    missing: List[str] = []
    debug_hits: Dict[str, float] = {}

    for section, kpis in KPI_KEYWORDS.items():
        for kpi_name, kw_list in kpis.items():
            full_pts = SUBPOINTS[section][kpi_name]
            pts, conf = _kpi_score_for(text, kw_list, full_pts)
            by_section[section] += pts
            debug_hits[f"{section}:{kpi_name}"] = conf
            if pts == 0:
                missing.append(kpi_name)

        # clamp to section max
        by_section[section] = min(by_section[section], SECTION_MAX[section])

    total = sum(by_section.values())

    # Comments
    comment_bits = []
    if "Upsell Product" in missing and "Warranty Offer" in missing:
        comment_bits.append("Upsell opportunities missed.")
    elif "Upsell Product" in missing:
        comment_bits.append("Upsell product not offered.")
    if "Address Confirmation" in missing:
        comment_bits.append("Address not confirmed.")
    if "Greeting" in missing or "Introduction" in missing:
        comment_bits.append("Opening could be stronger.")
    if "Recap" in missing:
        comment_bits.append("Recap/summary missing.")

    comment = " ".join(comment_bits) or "Good structure overall."
    return total, by_section, missing, comment
