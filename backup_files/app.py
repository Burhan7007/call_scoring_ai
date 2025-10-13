import json, threading
from pathlib import Path
from datetime import datetime
import requests, torch
from flask import Flask, render_template, send_file, request, jsonify, abort, redirect, url_for
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ==============================
# PATHS / ENV
# ==============================
ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
MODELS_DIR = ROOT / "models"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
COMPUTE = "float16" if USE_GPU else "int8"

print(f"🎧 Loading Whisper model (small) [{DEVICE}, {COMPUTE}]...")
whisper_model = WhisperModel("tiny", device=DEVICE, compute_type=COMPUTE)

# ==============================
# TRANSLATION MODELS
# ==============================
HF_CACHE = MODELS_DIR / "hf"
TRANSLATORS = {}
TO_EN = {
    "it": "Helsinki-NLP/opus-mt-it-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "bg": "Helsinki-NLP/opus-mt-bg-en",
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
    out = mdl.generate(**batch, max_new_tokens=2048)
    return tok.decode(out[0], skip_special_tokens=True)

def translate_to_english(text, lang):
    try:
        model = TO_EN.get(lang, FALLBACK)
        return _translate(text, model)
    except Exception:
        return _translate(text, FALLBACK)

def translate_en_to_it(text):
    try:
        return _translate(text, EN_TO_IT)
    except Exception:
        return text

# ==============================
# LANGUAGE BY PHONE COUNTRY PREFIX (fallback)
# ==============================
def detect_language_from_country(phone: str):
    if not phone:
        return "en"
    s = str(phone).lstrip("+")
    mapping = {
        "39": "it",   # Italy
        "34": "es",   # Spain
        "359": "bg",  # Bulgaria
        "386": "sl",  # Slovenia
        "30": "gr",   # Greece
        "44": "en",   # UK
        "33": "fr",   # France
        "49": "de",   # Germany
    }
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
# SIMPLE DIARIZATION
# ==============================
def diarize(raw, pause=1.2):
    dialogue, buf, cur, start, last = [], [], "Agent", 0.0, 0.0
    def flush(b, s, st, en):
        if b:
            dialogue.append({"speaker": s, "text": " ".join(b), "start": st, "end": en})
    for seg in raw:
        gap = seg.start - last
        if gap >= pause:
            flush(buf, cur, start, last)
            buf, start = [], seg.start
            cur = "Client" if cur == "Agent" else "Agent"
        buf.append(seg.text.strip())
        last = seg.end
    flush(buf, cur, start, last)
    return dialogue

# ==============================
# VOISO (CDR)
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

# ==============================
# PROCESS ONE AUDIO
# ==============================
def process_audio(file_path: Path, uuid=None):
    print(f"🎧 Transcribing {file_path.name}")
    try:
        segments, info = whisper_model.transcribe(str(file_path), vad_filter=True, beam_size=5)
        raw = list(segments)
        txt = " ".join([s.text.strip() for s in raw])
        lang = (info.language or "").lower() or "en"

        cdr = fetch_voiso(uuid) if uuid else {}
        if lang == "unknown" or not lang:
            lang = detect_language_from_country(cdr.get("to") or cdr.get("from"))

        dialogue = diarize(raw)
        en = txt if lang == "en" else translate_to_english(txt, lang)
        it = translate_en_to_it(en)
        total, missing, comment = score_text(en)

        res = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent_name": cdr.get("agent", "Unknown"),
            "customer_phone": cdr.get("to") or cdr.get("from", "Unknown"),
            "duration": cdr.get("duration", "N/A"),
            "call_status": cdr.get("disposition", "Unknown"),
            "language_detected": lang,
            "translation": {"english": en, "italian": it},
            "dialogue": dialogue,
            "scoring": {"total": total, "missing": missing, "comment": comment},
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
# HELPERS
# ==============================
def _safe_load_json(p: Path):
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return None
        return json.loads(txt)
    except Exception as e:
        print(f"⚠️ Skipping {p.name}: {e}")
        return None

def _normalize(d: dict) -> dict:
    d.setdefault("agent_name", "Unknown")
    d.setdefault("customer_phone", "Unknown")
    d.setdefault("duration", "N/A")
    d.setdefault("call_status", "Unknown")
    d.setdefault("language_detected", "Unknown")
    d.setdefault("translation", {})
    d["translation"].setdefault("english", "")
    d["translation"].setdefault("italian", "")
    d.setdefault("dialogue", [])
    d.setdefault("scoring", {"total": 0, "missing": [], "comment": ""})
    return d

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/health")
def health():
    return jsonify({"ok": True}), 200

# --- LOGIN FLOW FIX ---
@app.route('/')
def home():
    return redirect(url_for('login_page'))

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    q = (request.args.get("q") or "").strip().lower()
    phone_q = (request.args.get("phone") or "").strip()
    agent_q = (request.args.get("agent") or "").strip().lower()
    lang_q = (request.args.get("lang") or "").strip().lower()
    status_q = (request.args.get("status") or "answered").strip().lower()

    all_rows, items = [], []
    for f in RECORDINGS_DIR.glob("call_*.json"):
        d = _safe_load_json(f)
        if not d:
            continue
        d = _normalize(d)
        d["_id"] = f.stem.replace("call_", "")
        all_rows.append(d)

    agents = sorted({r.get("agent_name") for r in all_rows if r.get("agent_name")})
    langs = sorted({r.get("language_detected") for r in all_rows if r.get("language_detected")})
    statuses = sorted({r.get("call_status") for r in all_rows if r.get("call_status")})

    for r in all_rows:
        if status_q and r.get("call_status", "").lower() != status_q:
            continue
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
        items=items, agents=agents, langs=langs, statuses=statuses,
        q=q, phone=phone_q, agent=agent_q, lang=lang_q, status=status_q
    )

# === Other routes (no change) ===
@app.route("/call/<cid>")
def detail(cid):
    jf = RECORDINGS_DIR / f"call_{cid}.json"
    if not jf.exists():
        abort(404)
    d = _safe_load_json(jf) or {}
    d = _normalize(d)
    return render_template("detail.html", d=d, cid=cid)

@app.route("/export/csv")
def export_csv():
    valid = []
    for f in RECORDINGS_DIR.glob("call_*.json"):
        d = _safe_load_json(f)
        if not d:
            continue
        valid.append(_normalize(d))
    if not valid:
        abort(404)
    df = pd.json_normalize(valid)
    out = RECORDINGS_DIR / "export.csv"
    df.to_csv(out, index=False)
    return send_file(out, as_attachment=True)

@app.route("/report/<cid>.pdf")
def report_pdf(cid):
    jf = RECORDINGS_DIR / f"call_{cid}.json"
    if not jf.exists():
        abort(404)
    d = _normalize(_safe_load_json(jf) or {})
    out = RECORDINGS_DIR / f"report_{cid}.pdf"

    c = canvas.Canvas(str(out), pagesize=A4)
    w, h = A4
    y = h - 40
    def line(text, dy=16):
        nonlocal y
        c.drawString(40, y, (text or "")[:120])
        y -= dy

    c.setFont("Helvetica-Bold", 14); line(f"Call Report — {cid}", 22)
    c.setFont("Helvetica", 11)
    line(f"Timestamp: {d.get('timestamp','')}")
    line(f"Agent: {d.get('agent_name')}   |   Customer: {d.get('customer_phone')}")
    line(f"Language: {d.get('language_detected')}   |   Duration: {d.get('duration')}   |   Status: {d.get('call_status')}")
    s = d.get("scoring", {})
    line(f"Score: {s.get('total',0)} / 100")
    miss = s.get("missing") or []
    if miss:
        line("Missing KPIs: " + ", ".join(miss))
    line("")
    line("Transcript (EN):", 18)
    for chunk in (d.get("translation", {}).get("english", "")).split("\n"):
        line(chunk)
        if y < 80: c.showPage(); y = h - 40; c.setFont("Helvetica", 11)
    c.showPage(); c.save()
    return send_file(out, as_attachment=True)

# === Webhook ===
@app.route("/voiso-webhook", methods=["POST"])
def voiso_webhook():
    try:
        payload = request.get_json(force=True)
        call = payload.get("data", payload)
        call_id = call.get("uuid") or call.get("id") or datetime.utcnow().strftime("%Y%m%d%H%M%S")
        url = call.get("recording") or call.get("recording_url")
        if not url:
            return jsonify({"status": "error", "msg": "no recording"}), 400

        dest_audio = RECORDINGS_DIR / f"call_{call_id}.mp3"
        print(f"⬇️ Downloading {url} → {dest_audio.name}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest_audio.write_bytes(r.content)

        threading.Thread(target=process_audio, args=(dest_audio, call_id), daemon=True).start()
        return jsonify({"status": "ok", "id": call_id}), 200
    except Exception as e:
        print("❌ Webhook error:", e)
        return jsonify({"status": "error", "msg": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Server → http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
