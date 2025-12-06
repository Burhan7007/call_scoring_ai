from pathlib import Path
import re
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer, util

# Paths
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "hf" / "distiluse-base-multilingual-cased-v2"

print(f"🧠 Loading sentence transformer from: {MODEL_DIR}")
model = SentenceTransformer(str(MODEL_DIR))

# KPI definitions (you can tweak phrases anytime)
KPI_DEFINITIONS = {
    "Greeting": [
        "hello", "good morning", "good afternoon", "good evening",
        "hi, how are you", "good day"
    ],
    "Introduction": [
        "my name is", "this is", "you are speaking with"
    ],
    "Company Presentation": [
        "i am calling from", "we are calling from", "from company",
        "on behalf of", "our company", "presenting our company"
    ],
    "Product Mention": [
        "about your order", "about the product", "regarding the product",
        "you ordered", "regarding your purchase"
    ],
    "Address Confirmation": [
        "confirm your address", "your address is", "postal code",
        "zip code", "confirm your data", "confirm your details"
    ],
    "Recap": [
        "to recap", "just to confirm", "let me summarize", "so in summary"
    ],
    "Tone of Voice": [
        "thank you very much", "we appreciate", "thanks for your time",
        "have a nice day"
    ],
    "Upsell Product": [
        "we also have another offer", "second product", "another product",
        "additional product", "extra product"
    ],
    "Warranty Offer": [
        "extended warranty", "extra warranty", "additional guarantee",
        "protection plan", "2 year warranty", "3 year warranty"
    ],
    # Extra “script sections” the client mentioned
    "Insurance Upsell": [
        "insurance", "insure the product", "insurance plan",
        "coverage plan"
    ],
    "Product Upsell": [
        "upgrade your order", "special offer", "bundle offer",
        "add another item", "promotion for you"
    ],
}

KPI_SCORE = 10  # each KPI = 10 points


def _clean_text(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _is_noise_or_empty(t: str) -> Dict:
    """
    Detect obvious garbage / blank calls so we don't punish score logic:
    - too few words
    - repeated filler like 'yeah yeah yeah'
    """
    txt = _clean_text(t)
    words = txt.split()
    wc = len(words)

    if wc == 0:
        return {"is_noise": True, "reason": "empty", "word_count": 0}

    # Huge repetition of same token
    unique = set(words)
    rep_ratio = len(unique) / max(1, wc)

    if wc < 8:
        return {"is_noise": True, "reason": "too_short", "word_count": wc}

    if rep_ratio < 0.2:
        return {"is_noise": True, "reason": "high_repetition", "word_count": wc}

    return {"is_noise": False, "reason": "", "word_count": wc}


def ai_score(agent_english_text: str) -> Dict:
    """
    Main scoring function.
    Input: full Agent English text (already translated if needed).
    Output: dict with score, missing KPIs, comment, and flags.
    """
    text = _clean_text(agent_english_text or "")
    noise_info = _is_noise_or_empty(text)

    # If clear noise / too short → don't try to score, just mark as low-content
    if noise_info["is_noise"]:
        return {
            "score": 0,
            "missing": list(KPI_DEFINITIONS.keys()),
            "comment": f"Low-content / noisy call ({noise_info['reason']})",
            "flags": {
                "low_content": True,
                "noise_like": True,
                "word_count": noise_info["word_count"],
            },
        }

    # Encode whole call once
    call_emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)

    total = 0
    missing: List[str] = []
    details = []

    for kpi_name, phrases in KPI_DEFINITIONS.items():
        phrase_embs = model.encode(
            phrases, convert_to_tensor=True, normalize_embeddings=True
        )
        sims = util.cos_sim(call_emb, phrase_embs)[0].cpu().numpy()
        max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0

        # Threshold ~0.45 is a decent "similar phrase" cutoff
        if max_sim >= 0.45:
            total += KPI_SCORE
            details.append(f"{kpi_name} (match {max_sim:.2f})")
        else:
            missing.append(kpi_name)

    total = min(total, 100)

    if not missing:
        comment = "Strong call, all key steps detected."
    else:
        comment = "Missing / weak sections: " + ", ".join(missing)

    return {
        "score": total,
        "missing": missing,
        "comment": comment,
        "flags": {
            "low_content": False,
            "noise_like": False,
            "word_count": noise_info["word_count"],
            "details": details,
        },
    }
