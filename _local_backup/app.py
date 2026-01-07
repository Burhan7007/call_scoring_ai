import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# ====== MODEL & CACHE PATHS ======
ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
MODELS_DIR = ROOT / "models"
HF_CACHE = MODELS_DIR / "hf"
WHISPER_DIR = MODELS_DIR / "whisper"

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)
WHISPER_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE))

# ====== IMPORTS ======
import torch
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer

# ====== DEVICE / PRECISION ======
USE_GPU = torch.cuda.is_available()
WHISPER_COMPUTE = "float16" if USE_GPU else "int8"
WHISPER_DEVICE = "cuda" if USE_GPU else "cpu"

# ====== LOAD STT (Whisper) ======
print(f"Loading Whisper model (device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE})...")
whisper_model = WhisperModel(
    "large-v2", device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE,
    download_root=str(WHISPER_DIR)
)

# ====== TRANSLATION MODELS ======
TO_EN_MODELS = {
    "bg": "Helsinki-NLP/opus-mt-bg-en",          # Bulgarian -> English
    "ro": "Helsinki-NLP/opus-mt-ROMANCE-en",     # Romanian -> English (covers Romance langs)
}
EN_TO_IT = "Helsinki-NLP/opus-mt-en-it"          # English -> Italian

_TRANSLATORS = {}

def load_translator(model_name: str):
    if model_name in _TRANSLATORS:
        return _TRANSLATORS[model_name]
    print(f"Loading translator: {model_name}")
    tok = MarianTokenizer.from_pretrained(model_name, cache_dir=str(HF_CACHE))
    mdl = MarianMTModel.from_pretrained(model_name, cache_dir=str(HF_CACHE))
    _TRANSLATORS[model_name] = (tok, mdl)
    return tok, mdl

@torch.inference_mode()
def translate(text: str, model_name: str) -> str:
    tok, mdl = load_translator(model_name)
    batch = tok([text], return_tensors="pt", padding=True, truncation=True)
    out_ids = mdl.generate(**batch, max_new_tokens=1024)
    return tok.decode(out_ids[0], skip_special_tokens=True)

# ====== STT ======
def transcribe(file_path: Path):
    print(f"Transcribing: {file_path.name}")
    segments, info = whisper_model.transcribe(
        str(file_path),
        vad_filter=True,
        beam_size=5,
        condition_on_previous_text=True,
    )
    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    detected_lang = (info.language or "").lower()
    return transcript, detected_lang, dict(
        language=info.language, language_probability=getattr(info, "language_probability", None)
    )

# ====== SCORING RULES ======
SECTION_MAX = {
    "trust_relationship": 30,
    "clarity_accuracy": 40,
    "value_upsell": 30,
}

EN_KEYWORDS = {
    "trust_relationship": {
        "Greeting": ["good morning", "good afternoon", "hello", "hi"],
        "Introduction": ["my name is", "this is", "speaking"],
        "Company Presentation": ["from", "company", "on behalf of"],
    },
    "clarity_accuracy": {
        "Product Mention": ["chainsaw", "product", "order", "item"],
        "Address Confirmation": ["address", "street", "house number", "postcode", "zip"],
        "Recap": ["recap", "summary", "confirm once more", "to confirm"],
        "Tone of Voice": ["please", "thank you", "kindly", "appreciate"],
    },
    "value_upsell": {
        "Upsell Product": ["second product", "another model", "upgrade", "bundle"],
        "Warranty Offer": ["warranty", "guarantee", "extended", "protection"],
    },
}

SUBPOINTS = {
    "trust_relationship": {"Greeting": 10, "Introduction": 10, "Company Presentation": 10},
    "clarity_accuracy": {"Product Mention": 10, "Address Confirmation": 10, "Recap": 10, "Tone of Voice": 10},
    "value_upsell": {"Upsell Product": 15, "Warranty Offer": 15},
}

def score_english_text(english_text: str):
    text = " " + english_text.lower() + " "
    details = {s: 0 for s in SECTION_MAX}
    misses = []

    for section, groups in EN_KEYWORDS.items():
        for kpi, keywords in groups.items():
            got = any(kw in text for kw in keywords)
            pts = SUBPOINTS[section][kpi]
            if got:
                details[section] += pts
            else:
                misses.append(kpi)

    total = sum(details.values())
    comment_bits = []
    if "Upsell Product" in misses or "Warranty Offer" in misses:
        comment_bits.append("Upsell opportunities missed.")
    if "Address Confirmation" in misses:
        comment_bits.append("Address not confirmed.")
    if "Greeting" in misses or "Introduction" in misses:
        comment_bits.append("Weak opening.")
    if "Recap" in misses:
        comment_bits.append("Recap missing.")
    comment = " ".join(comment_bits) or "Good structure overall."

    return total, details, misses, comment

# ====== FULL PIPELINE ======
def process_call(file_path: Path, src_hint: str | None = None) -> dict:
    transcript, auto_lang, lang_info = transcribe(file_path)
    lang = (src_hint or auto_lang or "").lower()
    print(f"Detected language: {lang or 'unknown'}  (info={lang_info})")

    # Translate -> English
    if lang in TO_EN_MODELS:
        english = translate(transcript, TO_EN_MODELS[lang])
    else:
        english = transcript

    # English -> Italian
    italian = translate(english, EN_TO_IT)

    # Scoring
    score, section_scores, missing, comment = score_english_text(english)

    result = {
        "call_id": file_path.name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "language_detected": lang or "unknown",
        "stt": {
            "transcript": transcript,
            "detected": lang_info,
        },
        "translation": {
            "english": english,
            "italian": italian,
        },
        "scoring": {
            "total": score,
            "by_section": section_scores,
            "missing_kpis": missing,
            "comment": comment,
        },
    }
    return result

def main():
    ap = argparse.ArgumentParser(description="Self-hosted Call Scoring (Milestone 1)")
    ap.add_argument("--file", required=True, help="Path to .wav/.mp3 in ./recordings")
    ap.add_argument("--lang", default=None, help="Optional source language hint (e.g., bg, ro)")
    ap.add_argument("--save-json", action="store_true", help="Save JSON next to audio file")
    args = ap.parse_args()

    fpath = Path(args.file)
    if not fpath.exists():
        fpath = RECORDINGS_DIR / args.file
    if not fpath.exists():
        raise FileNotFoundError(f"Audio file not found: {args.file}")

    result = process_call(fpath, src_hint=args.lang)

    print("\n========== RESULT JSON ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.save_json:
        out = fpath.with_suffix(".json")
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved report → {out}")

if __name__ == "__main__":
    main()
