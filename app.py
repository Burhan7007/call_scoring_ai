# ==============================
# SIMPLE CACHES (PERFORMANCE)
# ==============================
EMBED_CACHE = {}          # key: text -> embedding tensor
TRANS_CACHE = {}          # key: (lang, text) -> english text
IT_CACHE = {}             # key: english text -> italian text
BACK_EN_CACHE = {}        # key: italian text -> back english

# keep cache from exploding
MAX_CACHE = 5000
def _cache_put(cache: dict, key, val):
    if key in cache:
        return
    if len(cache) >= MAX_CACHE:
        # pop first inserted (simple)
        cache.pop(next(iter(cache)))
    cache[key] = val

# ==============================
# TRANSLATION MODELS EXTENSION
# ==============================
IT_TO_EN = "Helsinki-NLP/opus-mt-it-en"

import os, json, threading
from pathlib import Path
from datetime import datetime
import requests, torch
from flask import (
    Flask, render_template, send_file, request, jsonify, abort,
    redirect, url_for, session, flash
)
from functools import wraps
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.security import generate_password_hash, check_password_hash
from queue import Queue

AUDIO_QUEUE = Queue()

def audio_worker():
    while True:
        try:
            fp, uuid = AUDIO_QUEUE.get()
            print(f"🎧 Worker: starting processing for {fp.name}")
            process_audio(fp, uuid)
        except Exception as e:
            print("Worker error:", e)
        finally:
            AUDIO_QUEUE.task_done()

# Start background worker
threading.Thread(target=audio_worker, daemon=True).start()

# ==============================
# PATHS / ENV  (MUST COME FIRST)
# ==============================
ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
MODELS_DIR = ROOT / "models"
CREDS_FILE = ROOT / "admin_creds.json"

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# EMBEDDING MODEL PATH
# ==============================
EMBED_MODEL_PATH = MODELS_DIR / "hf" / "distiluse-base-multilingual-cased-v2"

# ==============================
# LOAD MULTILINGUAL EMBEDDING MODEL
# ==============================
from sentence_transformers import SentenceTransformer, util
print("🔤 Loading embedding model for AI scoring + product detection...")
embedder = SentenceTransformer(str(EMBED_MODEL_PATH))

# ==============================
# EMBEDDING + SIMILARITY HELPERS (CACHED)
# ==============================
def embed_text_cached(text: str):
    text = (text or "").strip()
    if not text:
        return None
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]
    emb = embedder.encode(text, convert_to_tensor=True)
    _cache_put(EMBED_CACHE, text, emb)
    return emb

def cosine_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        e1 = embed_text_cached(a)
        e2 = embed_text_cached(b)
        return float(util.cos_sim(e1, e2)[0][0])
    except:
        return 0.0


# ==============================
# WHISPER
# ==============================
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
COMPUTE = "float16" if USE_GPU else "int8"

print(f"🎧 Loading Whisper model (small) [{DEVICE}, {COMPUTE}]...")
whisper_model = WhisperModel("small", device=DEVICE, compute_type=COMPUTE)
whisper_model_medium = WhisperModel("medium", device=DEVICE, compute_type=COMPUTE)


# ==============================
# TRANSLATION MODELS (UPDATED)
# ==============================
HF_CACHE = MODELS_DIR / "hf"
TRANSLATORS = {}

TO_EN = {
    "it": "Helsinki-NLP/opus-mt-it-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "bg": "Helsinki-NLP/opus-mt-bg-en",
    "sl": "Helsinki-NLP/opus-mt-sl-en",
    "ro": "Helsinki-NLP/opus-mt-ro-en",
    "pl": "Helsinki-NLP/opus-mt-pl-en",
    "hr": "Helsinki-NLP/opus-mt-hr-en",
    "gr": "Helsinki-NLP/opus-mt-el-en",
}

FALLBACK = "Helsinki-NLP/opus-mt-mul-en"
EN_TO_IT = "Helsinki-NLP/opus-mt-en-it"


def _load_translator(name):
    if name in TRANSLATORS:
        return TRANSLATORS[name]
    print(f"Loading translator: {name}")
    tok = MarianTokenizer.from_pretrained(name, cache_dir=str(HF_CACHE))
    mdl = MarianMTModel.from_pretrained(name, cache_dir=str(HF_CACHE))
    TRANSLATORS[name] = (tok, mdl)
    return tok, mdl



@torch.inference_mode()
def _translate(text, name):
    tok, mdl = _load_translator(name)
    batch = tok([text], return_tensors="pt", padding=True, truncation=True)

    out = mdl.generate(
        **batch,
        max_new_tokens=256,
        num_beams=3,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=1.0,
        early_stopping=True,
    )
    return tok.decode(out[0], skip_special_tokens=True)



def translate_to_english(text, lang):
    model = TO_EN.get(lang, FALLBACK)
    try:
        return _translate(text, model)
    except Exception:
        return _translate(text, FALLBACK)


def translate_en_to_it(text):
    try:
        return _translate(text, EN_TO_IT)
    except Exception:
        return text

def translate_it_to_en(text: str) -> str:
    if not (text or "").strip():
        return ""
    if text in BACK_EN_CACHE:
        return BACK_EN_CACHE[text]
    try:
        out = _translate(text, IT_TO_EN)
    except Exception:
        out = text
    _cache_put(BACK_EN_CACHE, text, out)
    return out

def translation_score_en_it(en_text: str, it_text: str) -> float:
    back_en = translate_it_to_en(it_text)
    return cosine_sim(en_text, back_en)


# ==============================
# LANGUAGE DETECTION
# ==============================
def detect_language_from_country(phone: str):
    if not phone:
        return "en"
    s = str(phone).lstrip("+")
    mapping = {"39": "it", "34": "es", "359": "bg", "386": "sl", "30": "gr", "44": "en", "33": "fr", "49": "de"}
    for pref, lang in sorted(mapping.items(), key=lambda x: -len(x[0])):
        if s.startswith(pref):
            return lang
    return "en"

# ==============================
# SCORING
# ==============================
def score_text(english_text: str):
    t = " " + (english_text or "").lower() + " "
    kpis = {
        "Greeting": [" hello ", " hi ", " good morning ", " good afternoon "],
        "Introduction": [" my name is ", " this is "],
        "Company Presentation": [" company ", " calling from ", " organization "],
        "Product Mention": [" product ", " order ", " item ", " offer "],
        "Address Confirmation": [" address ", " zip ", " postcode ", " confirm your "],
        "Recap": [" confirm ", " recap ", " summary "],
        "Tone of Voice": [" thank you ", " please ", " appreciate "],
        "Upsell Product": [" upgrade ", " second ", " bundle "],
        "Warranty Offer": [" warranty ", " guarantee ", " protection plan "],
    }
    score, missing = 0, []
    for k, keywords in kpis.items():
        if any(kword in t for kword in keywords):
            score += 10
        else:
            missing.append(k)
    comment = "Good call!" if not missing else "Missing: " + ", ".join(missing)
    return score, missing, comment

# ==============================
# IMPROVED DIARIZATION (v3)
# ==============================
def diarize(raw, pause=1.0, max_agent_run=4):
    """
    Improved diarization:
    - Prevents repetition artifacts from Whisper.
    - Splits long repeated segments.
    - Alternates agent/client cleanly.
    """
    dialogue, buf, cur, start, last, run_len = [], [], "Agent", 0.0, 0.0, 0

    def clean_repetition(text: str) -> str:
        """
        Removes heavy repetition artifacts like:
        'no no no no...' OR repeated short phrases.
        Keeps normal speech intact.
        """
        words = text.split()
        if len(words) < 8:
            return text

        # 1) collapse long runs of same word
        cleaned = []
        last = None
        run = 0
        for w in words:
            wl = w.lower()
            if wl == last:
                run += 1
                # allow max 2 repeats
                if run >= 2:
                    continue
            else:
                run = 0
            cleaned.append(w)
            last = wl

        # 2) remove repeated 2-gram loops (like "no no", "yes yes")
        out = []
        i = 0
        while i < len(cleaned):
            if i + 3 < len(cleaned):
                a = (cleaned[i].lower(), cleaned[i+1].lower())
                b = (cleaned[i+2].lower(), cleaned[i+3].lower())
                if a == b and a[0] in {"no", "yes", "ok", "okay"}:
                    # skip one repeated pair
                    out.extend(cleaned[i:i+2])
                    i += 4
                    continue
            out.append(cleaned[i])
            i += 1

        return " ".join(out)

    def flush(b, s, st, en):
        if not b:
            return
        text = " ".join(b).strip()
        text = clean_repetition(text)
        if text:
            dialogue.append({"speaker": s, "text": text, "start": st, "end": en})


    for seg in raw:
        gap = seg.start - last

        # switch speaker if long pause or too many sentences
        if gap >= pause or run_len >= max_agent_run:
            flush(buf, cur, start, last)
            buf = []
            start = seg.start
            cur = "Client" if cur == "Agent" else "Agent"
            run_len = 0

        buf.append(seg.text.strip())
        last = seg.end
        run_len += 1

    flush(buf, cur, start, last)
    return dialogue




# ==============================
# LOGIN SYSTEM
# ==============================
def get_admin_creds():
    if not CREDS_FILE.exists():
        creds = {"username": "admin", "password_hash": generate_password_hash("ChangeMe123!")}
        CREDS_FILE.write_text(json.dumps(creds))
        print("⚠️ Default admin created: username=admin | password=ChangeMe123!")
    else:
        creds = json.loads(CREDS_FILE.read_text())
    return creds

def save_admin_creds(username, new_pw):
    creds = {"username": username, "password_hash": generate_password_hash(new_pw)}
    CREDS_FILE.write_text(json.dumps(creds))
    print("✅ Password updated for", username)

# ==============================
# VOISO CDR
# ==============================
VOISO_KEY = "3dc9442851a083885a85a783329b9552e0406864cba34b62"
VOISO_URL = f"https://cc-rtm01.voiso.com/api/v2/cdr?key={VOISO_KEY}"

def fetch_voiso(uuid):
    try:
        r = requests.get(f"{VOISO_URL}&uuid={uuid}", timeout=25)
        d = r.json()
        if "records" in d and d["records"]:
            rec = d["records"][0]
            return {
                "agent": rec.get("agent"),
                "from": rec.get("from"),
                "to": rec.get("to"),
                "duration": rec.get("duration"),
                "disposition": rec.get("disposition"),
            }
    except Exception as e:
        print("⚠️ fetch_voiso:", e)
    return {}

# def process_audio(file_path: Path, uuid=None):
#     from sentence_transformers import util as st_util

#     # -----------------------------
#     # SEMANTIC SIMILARITY
#     # -----------------------------
#     def sem_sim(a: str, b: str) -> float:
#         if not a or not b:
#             return 0
#         try:
#             e1 = embedder.encode(a, convert_to_tensor=True)
#             e2 = embedder.encode(b, convert_to_tensor=True)
#             return float(st_util.cos_sim(e1, e2)[0][0])
#         except:
#             return 0

#     # -----------------------------
#     # CLEAN / DEDUPE
#     # -----------------------------
#     def clean(t):
#         t = t.replace("..", ".").replace("...", ".")
#         t = " ".join(t.split())
#         return t.strip()

#     def dedupe(lines):
#         out, seen = [], set()
#         for l in lines:
#             if len(l) < 3:
#                 continue
#             low = l.lower()
#             if low not in seen:
#                 seen.add(low)
#                 out.append(l)
#         return out

#     print(f"🎧 Transcribing {file_path.name}")

#     try:
#         # ----------------------------------------------------
#         # 1. TRANSCRIBE
#         # ----------------------------------------------------
#         segments, info = whisper_model.transcribe(str(file_path), vad_filter=True, beam_size=5)
#         raw = [s for s in segments if s.text.strip()]
#         txt = " ".join([s.text.strip() for s in raw])

#         # retry without VAD if too short
#         if len(txt.split()) < 15:
#             print("⚠ Weak transcription → retrying without VAD")
#             segments, info = whisper_model.transcribe(str(file_path), vad_filter=False, beam_size=5)
#             raw = [s for s in segments if s.text.strip()]
#             txt = " ".join([s.text.strip() for s in raw])

#         blank_call = len(txt.split()) < 5

#         # ----------------------------------------------------
#         # 2. DIARIZATION
#         # ----------------------------------------------------
#         dialogue = diarize(raw)

#         # fallback diarization if only one speaker detected
#         if len({d["speaker"] for d in dialogue}) < 2:
#             dialogue = []
#             cur = "Agent"
#             for i, seg in enumerate(raw):
#                 dialogue.append({
#                     "speaker": cur,
#                     "text": seg.text.strip(),
#                     "start": seg.start,
#                     "end": seg.end
#                 })
#                 if i % 2 == 1:
#                     cur = "Client" if cur == "Agent" else "Agent"

#         # dialogue score = talking time
#         total_duration = raw[-1].end if raw else 0
#         spoken = sum(d["end"] - d["start"] for d in dialogue)
#         dialogue_score = int(min(100, (spoken / total_duration) * 100)) if total_duration else 0

#         # ----------------------------------------------------
#         # 3. LANGUAGE DETECTION
#         # ----------------------------------------------------
#         cdr = fetch_voiso(uuid) if uuid else {}

#         lang = (info.language or "").lower().strip()
#         if lang in ("", "unknown"):
#             lang = detect_language_from_country(cdr.get("to") or cdr.get("from"))

#         if lang.startswith("sl"): lang = "sl"
#         if lang.startswith(("hr", "bs", "sr")): lang = "hr"
#         if lang.startswith("el"): lang = "gr"

#         # ----------------------------------------------------
#         # 4. PREP TEXT
#         # ----------------------------------------------------
#         agent_lines = [d["text"] for d in dialogue if d["speaker"] == "Agent"]
#         client_lines = [d["text"] for d in dialogue if d["speaker"] == "Client"]

#         combined_text = " ".join(agent_lines + client_lines)
#         if len(combined_text.split()) < 20:
#             combined_text = txt

#         # ----------------------------------------------------
#         # 5. TRANSLATION — BULLET FORMAT
#         # ----------------------------------------------------
#         en_lines = []
#         it_lines = []

#         for turn in dialogue:
#             sp = turn["speaker"]
#             orig = clean(turn["text"])

#             # English translation
#             if lang == "en":
#                 en_t = orig
#             else:
#                 en_t = clean(translate_to_english(orig, lang))

#             # Italian translation
#             it_t = clean(translate_en_to_it(en_t))

#             en_lines.append(f"- {sp}: {en_t}")
#             it_lines.append(f"- {'Agente' if sp == 'Agent' else 'Cliente'}: {it_t}")

#         en_agent = [l.replace("- Agent: ", "") for l in en_lines if l.startswith("- Agent:")]
#         en_client = [l.replace("- Client: ", "") for l in en_lines if l.startswith("- Client:")]
#         combined_en = " ".join(en_agent + en_client).lower()

#         # ----------------------------------------------------
#         # 6. KPI SCORING
#         # ----------------------------------------------------
#         kpi_desc = {
#             "Greeting": "agent greets politely with hello or good morning",
#             "Introduction": "agent introduces themselves with their name",
#             "Company Presentation": "agent explains the company they represent",
#             "Product Mention": "agent describes the product or order",
#             "Upsell Product": "agent offers extra product or upgrade",
#             "Insurance Upsell": "agent offers warranty or guarantee",
#             "Address Confirmation": "agent confirms the delivery address",
#             "Recap": "agent summarizes the order before finishing the call",
#             "Tone of Voice": "agent speaks politely and thanks the customer",
#         }

#         ai_score, missing = 0, []
#         SIM_THR = 0.18
#         chunks = en_agent

#         for kpi, desc in kpi_desc.items():
#             best = max((sem_sim(c, desc) for c in chunks), default=0)
#             if best >= SIM_THR:
#                 ai_score += 10
#             else:
#                 missing.append(kpi)

#         # keyword backup
#         fallback = {
#             "Greeting": ["hello", "good morning", "hi"],
#             "Introduction": ["my name", "this is"],
#             "Company Presentation": ["calling from", "company"],
#             "Product Mention": ["product", "order", "package"],
#             "Upsell Product": ["extra", "upgrade"],
#             "Insurance Upsell": ["warranty", "guarantee"],
#             "Address Confirmation": ["street", "address", "postcode"],
#             "Recap": ["confirm", "summary"],
#             "Tone of Voice": ["thank you", "thanks"],
#         }

#         for kpi, words in fallback.items():
#             if kpi in missing and any(w in combined_en for w in words):
#                 ai_score += 10
#                 missing.remove(kpi)

#         comment = "Blank or noise-only call" if blank_call else \
#             ("Good call!" if not missing else f"Missing: {', '.join(missing)}")

#         if blank_call:
#             ai_score = 0
#             missing = list(kpi_desc.keys())

#         # ----------------------------------------------------
#         # 7. ORDER STATUS (FIXED)
#         # ----------------------------------------------------
#         order_status = "unknown"
#         disp = (cdr.get("disposition") or "").lower()

#         no_talk = (dialogue_score < 5) or (len(en_agent) + len(en_client) == 0)

#         if no_talk:
#             order_status = "recall"

#         elif any(x in disp for x in ["abandon", "abandoned"]):
#             order_status = "recall"

#         elif disp in ["failed", "no answer", "busy"]:
#             order_status = "recall"

#         else:
#             accept_patterns = [
#                 "i confirm", "i accept", "yes i confirm", "i agree",
#                 "send it", "ok send", "i will receive", "i want it",
#                 "proceed with the order", "yes the order"
#             ]

#             refuse_patterns = [
#                 "i don't want", "i refuse", "cancel the order",
#                 "not interested", "stop the order", "i won't take it",
#                 "i do not want", "no i don't want"
#             ]

#             recall_patterns = [
#                 "call me later", "call later", "later please",
#                 "not now", "try again later", "call back later"
#             ]

#             def match_any(text, patterns, th=0.22):
#                 return any(sem_sim(text, p) > th for p in patterns)

#             if match_any(combined_en, accept_patterns):
#                 order_status = "accepted"
#             elif match_any(combined_en, refuse_patterns):
#                 order_status = "refused"
#             elif match_any(combined_en, recall_patterns):
#                 order_status = "recall"

#         # ----------------------------------------------------
#         # 8. REJECTION REASON
#         # ----------------------------------------------------
#         rejection_reason = None
#         if order_status == "refused":
#             reasons = {
#                 "cancelled": "customer confirms but later cancels",
#                 "fake": "customer says they never ordered anything",
#                 "error": "wrong number or wrong customer",
#                 "objection": "customer refuses because of price or distrust",
#             }

#             best_score, best_label = 0, None
#             for label, desc in reasons.items():
#                 s = sem_sim(combined_en, desc)
#                 if s > best_score:
#                     best_score = s
#                     best_label = label

#             rejection_reason = best_label

#         # ----------------------------------------------------
#         # 9. BUILD JSON RESULT
#         # ----------------------------------------------------
#         res = {
#             "timestamp": datetime.utcnow().isoformat() + "Z",
#             "agent_name": cdr.get("agent", "Unknown"),
#             "customer_phone": cdr.get("to") or cdr.get("from", "Unknown"),
#             "duration": cdr.get("duration", "N/A"),
#             "call_status": cdr.get("disposition", "Unknown"),
#             "language_detected": lang,
#             "translation_score": 0,
#             "dialogue_score": dialogue_score,
#             "order_status": order_status,
#             "rejection_reason": rejection_reason or "-",
#             "blank_call": blank_call,
#             "translation": {
#                 "english": "\n".join(en_lines) if en_lines else "-",
#                 "italian": "\n".join(it_lines) if it_lines else "-",
#             },
#             "dialogue": dialogue,
#             "scoring": {
#                 "total": ai_score,
#                 "missing": missing,
#                 "comment": comment,
#             },
#         }

#         tmp = file_path.with_suffix(".tmp")
#         tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
#         tmp.rename(file_path.with_suffix(".json"))

#         print(f"✅ Saved {file_path.stem}.json")
#         return res

#     except Exception as e:
#         print("❌ process_audio error:", e)
#         return {}


from datetime import datetime
import json
from pathlib import Path

def compute_call_score(total_words: int, dialogue_score: int, both_spoke: bool, blank_call: bool):
    if blank_call:
        return 0

    if total_words >= 120: w = 40
    elif total_words >= 60: w = 30
    elif total_words >= 25: w = 20
    elif total_words >= 12: w = 10
    else: w = 0

    d = min(40, int(dialogue_score * 0.4))  # 0..40
    b = 20 if both_spoke else 0

    return max(0, min(100, w + d + b))


PRODUCT_CATALOG = [
    {"name": "Generic product", "desc": "customer order delivery product package shipment"},
    {"name": "Warranty / insurance", "desc": "warranty guarantee insurance protection plan"},
    {"name": "Upsell", "desc": "upgrade bundle additional extra offer"},
]

def detect_product_ai(combined_en: str, thr: float = 0.30):
    best = 0.0
    best_name = "-"
    for p in PRODUCT_CATALOG:
        s = cosine_sim(combined_en, p["desc"])
        if s > best:
            best = s
            best_name = p["name"]
    return {
        "product_detected": best >= thr,
        "product_name": best_name if best >= thr else "-",
        "product_confidence": round(best, 4),
    }

def text_has_any(t: str, phrases: list[str]) -> bool:
    t = (t or "").lower()
    return any(p in t for p in phrases)

def match_any_sem(text: str, patterns: list[str], th: float = 0.25) -> bool:
    return any(cosine_sim(text, p) >= th for p in patterns)



def process_audio(file_path: Path, uuid=None):
    print(f"🎧 Transcribing {file_path.name}")

    def clean(t):
        t = (t or "").replace("..", ".").replace("...", ".")
        return " ".join(t.split()).strip()

    try:
        # ----------------------------------------------------
        # 1) TRANSCRIBE  (WITH QUALITY RETRY + GREEK BOOST)
        # ----------------------------------------------------
        segments, info = whisper_model.transcribe(str(file_path), vad_filter=True, beam_size=5)
        raw = [s for s in segments if (s.text or "").strip()]
        txt = " ".join([s.text.strip() for s in raw])

        # ---- helpers ----
        def looks_hallucinated(t: str) -> bool:
            ws = (t or "").lower().split()
            if len(ws) < 8:
                return True
            uniq_ratio = len(set(ws)) / max(1, len(ws))
            return uniq_ratio < 0.25

        lang_guess = (info.language or "").lower().strip()

        # Retry conditions:
        need_retry = (
            len(txt.split()) < 12
            or looks_hallucinated(txt)
            or lang_guess in ("el", "gr")
        )

        if need_retry:
            print("⚠ Quality retry transcription...")
            # If greek, try medium (recommended)
            # NOTE: you must define whisper_model_medium globally (see below)
            if lang_guess in ("el", "gr"):
                segments, info = whisper_model_medium.transcribe(str(file_path), vad_filter=True, beam_size=5)
            else:
                segments, info = whisper_model.transcribe(str(file_path), vad_filter=False, beam_size=5)

            raw = [s for s in segments if (s.text or "").strip()]
            txt = " ".join([s.text.strip() for s in raw])

        blank_call = len(txt.split()) < 5

        # ----------------------------------------------------
        # 2) DIARIZATION
        # ----------------------------------------------------
        dialogue = diarize(raw)

        if len({d["speaker"] for d in dialogue}) < 2:
            # fallback alternating
            dialogue = []
            cur = "Agent"
            for i, seg in enumerate(raw):
                dialogue.append({"speaker": cur, "text": seg.text.strip(), "start": seg.start, "end": seg.end})
                if i % 2 == 1:
                    cur = "Client" if cur == "Agent" else "Agent"

        total_duration = raw[-1].end if raw else 0
        spoken = sum(d["end"] - d["start"] for d in dialogue)
        dialogue_score = int(min(100, (spoken / total_duration) * 100)) if total_duration else 0

        # ----------------------------------------------------
        # 3) CDR + LANGUAGE
        # ----------------------------------------------------
        cdr = fetch_voiso(uuid) if uuid else {}

        lang = (info.language or "").lower().strip()
        if lang in ("", "unknown"):
            lang = detect_language_from_country(cdr.get("to") or cdr.get("from"))

        if lang.startswith("sl"): lang = "sl"
        if lang.startswith(("hr", "bs", "sr")): lang = "hr"
        if lang.startswith("el"): lang = "gr"

        # ----------------------------------------------------
        # 4) TRANSLATION (EN + IT) + translate_score + retry
        # ----------------------------------------------------
        en_lines, it_lines = [], []
        agent_en_texts, client_en_texts = [], []

        for turn in dialogue:
            sp = turn["speaker"]
            orig = clean(turn["text"])

            # cache translate -> EN
            if lang == "en":
                en_t = orig
            else:
                k = (lang, orig)
                if k in TRANS_CACHE:
                    en_t = TRANS_CACHE[k]
                else:
                    en_t = clean(translate_to_english(orig, lang))
                    _cache_put(TRANS_CACHE, k, en_t)

            # cache EN -> IT
            if en_t in IT_CACHE:
                it_t = IT_CACHE[en_t]
            else:
                it_t = clean(translate_en_to_it(en_t))
                _cache_put(IT_CACHE, en_t, it_t)

            en_lines.append(f"- {sp}: {en_t}")
            it_lines.append(f"- {'Agente' if sp == 'Agent' else 'Cliente'}: {it_t}")

            if sp == "Agent": agent_en_texts.append(en_t)
            else: client_en_texts.append(en_t)

        combined_en = clean(" ".join(agent_en_texts + client_en_texts)).lower()
        total_words = len(combined_en.split())

        agent_spoke = len(agent_en_texts) > 0
        client_spoke = len(client_en_texts) > 0
        both_spoke = agent_spoke and client_spoke

        # translation_score (compare EN vs IT->EN back translation)
        # use full transcript text (EN)
        full_en_for_score = clean(" ".join(agent_en_texts + client_en_texts))
        full_it_for_score = clean("\n".join([l.split(": ", 1)[1] for l in it_lines if ": " in l]))

        translate_score = 0.0
        if full_en_for_score and full_it_for_score:
            translate_score = translation_score_en_it(full_en_for_score, full_it_for_score)

        # if translation is suspiciously low, redo IT once (same pipeline)
        # (kept simple to avoid loops)
        if translate_score and translate_score < 0.45:
            it_lines = []
            for line in en_lines:
                # rebuild IT from EN line again
                if ": " in line:
                    en_t = clean(line.split(": ", 1)[1])
                    it_t = clean(translate_en_to_it(en_t))
                    it_lines.append(f"- {'Agente' if 'Agent:' in line else 'Cliente'}: {it_t}")
            # recompute score
            full_it_for_score = clean("\n".join([l.split(": ", 1)[1] for l in it_lines if ": " in l]))
            translate_score = translation_score_en_it(full_en_for_score, full_it_for_score)

        # ----------------------------------------------------
        # 5) KPI SCORING (keep your AI KPI scoring)
        # ----------------------------------------------------
        score_val, missing, comment = score_text(combined_en)

        # reduce score=0 except blank/noise calls
        # if not blank_call and score is 0, give minimum 10 (prevents “0 on real calls”)
        if (not blank_call) and score_val == 0 and total_words >= 12:
            score_val = 10

        # ----------------------------------------------------
        # 6) call_score (Milestone 3)
        # ----------------------------------------------------
        call_score = compute_call_score(total_words, dialogue_score, both_spoke, blank_call)

        # ----------------------------------------------------
        # 7) ORDER STATUS (CLIENT-INTENT FIRST)
        # ----------------------------------------------------
        disp = (cdr.get("disposition") or "").lower()
        bad_disp = ["abandon", "abandoned", "failed", "busy", "no answer", "dialer_abandoned", "system_abandoned"]

        client_en_only = clean(" ".join(client_en_texts)).lower()
        agent_en_only  = clean(" ".join(agent_en_texts)).lower()

        # hard rules
        if blank_call or not client_spoke or dialogue_score < 10:
            order_status = "recall"
        elif any(x in disp for x in bad_disp):
            order_status = "recall"
        else:
            accept_patterns = [
                "i confirm", "i accept", "yes i confirm", "yes", "ok", "okay",
                "send it", "proceed", "i will receive", "i will take it",
                "i confirm the order", "i confirm address", "i agree"
            ]
            refuse_patterns = [
                "i don't want", "not interested", "cancel", "stop", "refuse",
                "i will not take it", "no i don't want"
            ]
            recall_patterns = [
                "call later", "later", "not now", "call back", "tomorrow"
            ]

            # priority: CUSTOMER intent
            if match_any_sem(client_en_only, accept_patterns, th=0.23) or text_has_any(client_en_only, ["i confirm", "i accept", "i agree", "send it"]):
                order_status = "accepted"
            elif match_any_sem(client_en_only, refuse_patterns, th=0.23) or text_has_any(client_en_only, ["not interested", "cancel", "refuse"]):
                order_status = "refused"
            elif match_any_sem(client_en_only, recall_patterns, th=0.23) or text_has_any(client_en_only, ["call later", "not now", "tomorrow"]):
                order_status = "recall"
            else:
                # if client didn't express intent, use call quality
                # long+both spoke → assume recall rather than wrong accept/refuse
                order_status = "recall"


        # ----------------------------------------------------
        # 8) REJECTION REASONS (Milestone 2 feature, keep simple)
        # ----------------------------------------------------
        rejection_reason = "-"
        if order_status == "refused":
            reasons = {
                "cancelled": "customer confirmed but later cancelled the order",
                "fake": "customer says they never placed any order",
                "error": "wrong person or wrong number",
                "objection": "customer refuses due to price or distrust",
            }
            best_score, best_label = 0.0, "-"
            for label, desc in reasons.items():
                s = cosine_sim(combined_en, desc)
                if s > best_score:
                    best_score, best_label = s, label
            rejection_reason = best_label

        # ----------------------------------------------------
        # 9) PRODUCT DETECTION (Milestone 3)
        # ----------------------------------------------------
        product_info = detect_product_ai(combined_en)

        # ----------------------------------------------------
        # 10) SAVE JSON
        # ----------------------------------------------------
        res = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent_name": cdr.get("agent", "Unknown"),
            "customer_phone": cdr.get("to") or cdr.get("from", "Unknown"),
            "duration": cdr.get("duration", "N/A"),
            "call_status": cdr.get("disposition", "Unknown"),
            "language_detected": lang,

            "blank_call": blank_call,
            "dialogue_score": dialogue_score,
            "call_score": call_score,

            "translation_score": round(float(translate_score or 0.0), 4),

            "order_status": order_status,
            "rejection_reason": rejection_reason,

            "product": product_info,

            "translation": {
                "english": "\n".join(en_lines) if en_lines else "-",
                "italian": "\n".join(it_lines) if it_lines else "-",
            },
            "dialogue": dialogue,
            "scoring": {
                "total": score_val,
                "missing": missing,
                "comment": comment,
            },
        }

        tmp = file_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.rename(file_path.with_suffix(".json"))

        print(f"✅ Saved {file_path.stem}.json")
        return res

    except Exception as e:
        print("❌ process_audio error:", e)
        return {}





# ==============================
# FLASK APP
# ==============================
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "supersecretkey"

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("admin"):
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

# ==============================
# AUTH ROUTES
# ==============================
@app.route("/login", methods=["GET", "POST"])
def login():
    creds = get_admin_creds()
    if request.method == "POST":
        user = request.form.get("username", "").strip()
        pw = request.form.get("password", "").strip()
        if user == creds["username"] and check_password_hash(creds["password_hash"], pw):
            session["admin"] = user
            flash("✅ Logged in successfully!", "success")
            return redirect(url_for("home"))
        flash("❌ Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        save_admin_creds("admin", "ChangeMe123!")
        flash("🔁 Password reset to default (admin / ChangeMe123!)", "info")
        return redirect(url_for("login"))
    return render_template("forgot_password.html")

@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    creds = get_admin_creds()
    if request.method == "POST":
        old_pw = request.form.get("old_password", "").strip()
        new_pw = request.form.get("new_password", "").strip()
        confirm_pw = request.form.get("confirm_password", "").strip()
        if not check_password_hash(creds["password_hash"], old_pw):
            flash("❌ Old password incorrect", "danger")
        elif new_pw != confirm_pw:
            flash("⚠️ Passwords do not match", "warning")
        elif len(new_pw) < 6:
            flash("⚠️ Password too short (min 6 chars)", "warning")
        else:
            save_admin_creds(creds["username"], new_pw)
            flash("✅ Password updated successfully!", "success")
            return redirect(url_for("home"))
    return render_template("change_password.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return redirect(url_for("home"))

# ==============================
# MAIN ROUTES
# ==============================
@app.route("/")
@login_required
def home():
    q = (request.args.get("q") or "").strip().lower()
    phone_q = (request.args.get("phone") or "").strip()
    agent_q = (request.args.get("agent") or "").strip().lower()
    lang_q = (request.args.get("lang") or "").strip().lower()
    status_q = (request.args.get("status") or "").strip().lower()


    # ORDER STATUS FILTER — now no default ("show all")
    order_status_q = (request.args.get("order_status") or "").strip().lower()

    # REJECTION FILTER — optional
    rejection_q = (request.args.get("rejection") or "").strip().lower()

    all_rows, items = [], []

    for f in RECORDINGS_DIR.glob("call_*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d["_id"] = f.stem.replace("call_", "")

            # existing safe defaults
            d["agent_name"] = d.get("agent_name") or "Unknown"
            d["language_detected"] = d.get("language_detected") or "Unknown"
            d["call_status"] = d.get("call_status") or "Unknown"
            d["duration_display"] = str(d.get("duration", "N/A"))

            # NEW: order status + rejection (for display + normalized for filtering)
            order_status_val = (d.get("order_status") or "").strip()
            rejection_val = (d.get("rejection_reason") or "").strip()

            d["order_status"] = order_status_val or "-"          # for template display
            d["rejection_reason"] = rejection_val or "-"         # for template display
            d["_order_status_norm"] = order_status_val.lower()
            d["_rejection_norm"] = rejection_val.lower()

            all_rows.append(d)
        except Exception as e:
            print("⚠️ Error reading file:", f.name, e)

    agents = sorted({r["agent_name"] for r in all_rows})
    langs = sorted({r["language_detected"] for r in all_rows})
    statuses = sorted({r["call_status"] for r in all_rows})

    for r in all_rows:
        # Call status filter
        if status_q and r.get("call_status", "").lower() != status_q:
            continue

        # NEW: order status filter
        if order_status_q and r.get("_order_status_norm", "") != order_status_q:
            continue

        # NEW: rejection reason filter
        if rejection_q and r.get("_rejection_norm", "") != rejection_q:
            continue

        # existing filters
        if agent_q and agent_q not in r.get("agent_name", "").lower():
            continue
        if lang_q and lang_q != r.get("language_detected", "").lower():
            continue
        if phone_q and phone_q not in (r.get("customer_phone") or ""):
            continue
        if q and q not in (r.get("translation", {}).get("english", "").lower()):
            continue

        items.append(r)

    items = sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)
    return render_template(
        "index.html",
        items=items,
        agents=agents,
        langs=langs,
        statuses=statuses,
    )




@app.route("/call/<cid>")
@login_required
def detail(cid):
    jf = RECORDINGS_DIR / f"call_{cid}.json"
    if not jf.exists():
        abort(404)
    d = json.loads(jf.read_text(encoding="utf-8"))
    return render_template("detail.html", d=d, cid=cid)

@app.route("/export/csv")
@login_required
def export_csv():
    valid = []
    for f in RECORDINGS_DIR.glob("call_*.json"):
        try:
            valid.append(json.loads(f.read_text(encoding="utf-8")))
        except:
            pass

    if not valid:
        abort(404)

    # ensure new fields exist
    for row in valid:
        row["order_status"] = row.get("order_status", "")
        row["rejection_reason"] = row.get("rejection_reason", "")

    df = pd.json_normalize(valid)
    out = RECORDINGS_DIR / "export.csv"
    df.to_csv(out, index=False)
    return send_file(out, as_attachment=True)


@app.route("/report/<cid>.pdf")
@login_required
def report_pdf(cid):
    jf = RECORDINGS_DIR / f"call_{cid}.json"
    if not jf.exists():
        abort(404)

    d = json.loads(jf.read_text(encoding="utf-8"))
    out = RECORDINGS_DIR / f"report_{cid}.pdf"

    c = canvas.Canvas(str(out), pagesize=A4)
    w, h = A4
    y = h - 40

    def line(text, dy=16):
        nonlocal y
        c.drawString(40, y, (text or "")[:120])
        y -= dy

    c.setFont("Helvetica-Bold", 14)
    line(f"Call Report — {cid}", 22)

    c.setFont("Helvetica", 11)
    line(f"Agent: {d.get('agent_name')} | Customer: {d.get('customer_phone')}")
    line(f"Language: {d.get('language_detected')} | Duration: {d.get('duration')} | Status: {d.get('call_status')}")

    # NEW FIELDS
    line(f"Order Status: {d.get('order_status', '-')}")
    line(f"Rejection: {d.get('rejection_reason', '-')}")

    s = d.get("scoring", {})
    line(f"Score: {s.get('total', 0)} / 100")
    miss = s.get("missing") or []
    if miss:
        line("Missing KPIs: " + ", ".join(miss))

    line("")
    line("Transcript (EN):", 18)

    for chunk in (d.get("translation", {}).get("english", "")).split("\n"):
        line(chunk)
        if y < 80:
            c.showPage()
            y = h - 40
            c.setFont("Helvetica", 11)

    c.showPage()
    c.save()
    return send_file(out, as_attachment=True)


@app.route("/voiso-webhook", methods=["POST"])
def voiso_webhook():
    try:
        payload = request.get_json(silent=True) or {}
        print("📥 Incoming webhook payload:", payload)

        call = (
            payload.get("data")
            or payload.get("payload")
            or payload
        )

        if not isinstance(call, dict):
            return jsonify({"status": "error", "msg": "invalid call data"}), 200

        call_id = (
            call.get("uuid")
            or call.get("id")
            or datetime.utcnow().strftime("%Y%m%d%H%M%S")
        )

        # 1. DIRECT RECORDING URL (most reliable)
        url = (
            call.get("recording")
            or call.get("recording_url")
            or call.get("audio")
            or call.get("file")
        )

        # 2. IF MISSING → TRY CDR QUIETLY
        if not url:
            cdr_url = call.get("cdr_url")
            if cdr_url:
                try:
                    print("🔄 Fetching CDR...")
                    cdr_raw = requests.get(cdr_url, timeout=10)
                    if cdr_raw.status_code == 200:
                        cdr_json = cdr_raw.json()
                        url = (
                            cdr_json.get("recording")
                            or cdr_json.get("recording_url")
                            or cdr_json.get("audio")
                            or cdr_json.get("file")
                        )
                except Exception as e:
                    print("❌ CDR fetch failed (ignored):", e)

        # 3. IF STILL NO URL → ACCEPT WEBHOOK BUT DO NOTHING
        if not url:
            print("❌ No recording URL available (ignored).")
            return jsonify({"status": "ok", "id": call_id}), 200

        # 4. DOWNLOAD RECORDING
        dest_audio = RECORDINGS_DIR / f"call_{call_id}.mp3"
        print(f"⬇️ Downloading audio → {dest_audio.name}")

        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest_audio.write_bytes(r.content)
        except Exception as e:
            print("❌ Download failed:", url, e)
            return jsonify({"status": "ok", "id": call_id}), 200

        # 5. QUEUE FOR AI PROCESSING
        AUDIO_QUEUE.put((dest_audio, call_id))
        print(f"📌 Queued for processing: {dest_audio.name}")

        return jsonify({"status": "ok", "id": call_id}), 200

    except Exception as e:
        print("❌ Webhook error:", e)
        return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

