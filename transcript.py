"""
transcript.py — Milestone 2.1
Advanced speaker labeling for Italian/Multilingual calls.
Handles mixed-dialogue segments, question-answer alternation, and embedded 'yes/no' replies.
"""

from typing import List, Dict
import re

SILENCE_FLIP_SEC = 0.8

AFFIRM_NEG = ["sì", "si", "no", "ok", "va bene", "perfetto", "grazie", "certo", "arrivederci", "salve"]
OPERATOR_CUES = [
    "sono", "mi chiamo", "la contattavamo", "le volevo comunicare", "pagamento", "corriere",
    "garanzia", "azienda", "signor", "offerta", "catalogo", "prodotto", "ordine", "spedizione", "indirizzo"
]

def _split_by_keywords(text: str) -> List[str]:
    """Split long segments by strong conversational boundaries (., ?, !, grazie, arrivederci)."""
    parts = re.split(r"(?<=[?.!])\s+|\b(grazie|arrivederci|salve)\b", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 2]

def _detect_speaker(chunk: str, current: str) -> str:
    """Guess if this chunk is Operator or Client based on phrases."""
    txt = chunk.lower()
    if any(kw in txt for kw in AFFIRM_NEG) and len(txt.split()) <= 6:
        return "Client"
    if any(kw in txt for kw in OPERATOR_CUES):
        return "Operator"
    if "?" in txt:
        # heuristic: if current was Operator asking, next likely Client
        return "Operator" if current == "Client" else "Client"
    return current

def assign_speakers(segments: List[Dict], lang_hint: str = "it") -> List[Dict]:
    """Label speakers in Whisper segments, with keyword & punctuation splitting."""
    labeled = []
    current = "Operator"

    for seg in segments:
        start, end, text = seg.get("start", 0.0), seg.get("end", 0.0), seg.get("text", "").strip()
        if not text:
            continue

        # split long continuous text into conversational sub-chunks
        chunks = _split_by_keywords(text)
        dur_per_chunk = (end - start) / max(1, len(chunks))

        for i, chunk in enumerate(chunks):
            spk = _detect_speaker(chunk, current)
            labeled.append({
                "speaker": spk,
                "start": round(start + i * dur_per_chunk, 2),
                "end": round(start + (i + 1) * dur_per_chunk, 2),
                "text": chunk.strip(),
            })
            current = spk

    # merge consecutive same-speaker lines
    merged: List[Dict] = []
    for seg in labeled:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)

    return merged

def format_transcript(labeled_segments: List[Dict]) -> str:
    lines = []
    for seg in labeled_segments:
        txt = re.sub(r"\s+([?.!,])", r"\1", (seg.get("text") or "").strip())
        if txt:
            lines.append(f"{seg.get('speaker','?')}: {txt}")
    return "\n".join(lines)

def build_transcript_output(segments: List[Dict], lang_hint: str = "it") -> Dict:
    labeled = assign_speakers(segments, lang_hint)
    formatted = format_transcript(labeled)
    return {
        "segments": labeled,
        "formatted_text": formatted,
    }
