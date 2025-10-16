import os, json, threading, logging
from pathlib import Path
from datetime import datetime
import requests, torch
from flask import (
    Flask, render_template, send_file, request, jsonify,
    abort, redirect, url_for, session, flash
)
from functools import wraps
from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.security import generate_password_hash, check_password_hash

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ==============================
# PATHS / ENV
# ==============================
ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
MODELS_DIR = ROOT / "models"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# WHISPER (accuracy-first)
# ==============================
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
COMPUTE = "float16" if USE_GPU else "int8"

# Use the more accurate 'base' model (vs tiny)
logging.info(f"🎧 Loading Whisper model (base) [{DEVICE}, {COMPUTE}]...")
whisper_model = WhisperModel("base", device=DEVICE, compute_type=COMPUTE)

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
    logging.info(f"🌐 Loading translator: {name}")
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
    """Best-effort: agar translation fail ho jaye to original text hi return karo."""
    try:
        if not text.strip():
            return ""
        # Agar already English hai, direct return
        if (lang or "").lower() == "en":
            return text
        model = TO_EN.get(lang, FALLBACK)
        return _translate(text, model)
    except Exception as e:
        logging.warning(f"EN translation skipped (fallback to source). Reason: {e}")
        return text  # scoring ke liye English hi treat karenge

def translate_en_to_it(text):
    """Best-effort: IT fail ho to empty string (UI clean rahega)."""
    try:
        if not text.strip():
            return ""
        return _translate(text, EN_TO_IT)
    except Exception as e:
        logging.warning(f"IT translation skipped. Reason: {e}")
        return ""

# ==============================
# LANGUAGE DETECTION
# ==============================
def detect_language_from_country(phone: str):
    if not phone:
        return "en"
    s = str(phone).lstrip("+")
    mapping = {
        "39": "it", "34": "es", "359": "bg", "386": "sl",
        "30": "gr", "44": "en", "33": "fr", "49": "de"
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
    for k, words in kpis.items():
        if any(w in t for w in words):
            score += 10
        else:
            missing.append(k)
    comment = "Good call!" if not missing else "Missing: " + ", ".join(missing)
    return score, missing, comment

# ==============================
# DIARIZATION (simple gap-based)
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
# LOGIN SYSTEM
# ==============================
CREDS_FILE = ROOT / "admin_creds.json"

def get_admin_creds():
    if not CREDS_FILE.exists():
        creds = {"username": "admin", "password_hash": generate_password_hash("ChangeMe123!")}
        CREDS_FILE.write_text(json.dumps(creds))
        logging.warning("⚠️ Default admin created: username=admin | password=ChangeMe123!")
    else:
        creds = json.loads(CREDS_FILE.read_text())
    return creds

def save_admin_creds(username, new_pw):
    creds = {"username": username, "password_hash": generate_password_hash(new_pw)}
    CREDS_FILE.write_text(json.dumps(creds))
    logging.info(f"✅ Password updated for {username}")

# ==============================
# NORMALIZER
# ==============================
def _normalize(d: dict) -> dict:
    d.setdefault("agent_name", "Unknown")
    d.setdefault("customer_phone", "Unknown")
    d.setdefault("duration", "N/A")
    d.setdefault("call_status", "Unknown")
    d.setdefault("language_detected", "Unknown")
    d.setdefault("translation", {"english": "", "italian": ""})
    d.setdefault("scoring", {"total": 0, "missing": [], "comment": ""})
    return d

# ==============================
# FLASK APP
# ==============================
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace_this_secret_key")

def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("admin"):
            flash("Please login first.", "warning")
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return wrapped

@app.route("/")
def home():
    if not session.get("admin"):
        return redirect(url_for("login_page"))
    return redirect(url_for("dashboard"))

# ==============================
# AUTH ROUTES
# ==============================
@app.route("/login", methods=["GET", "POST"])
def login_page():
    creds = get_admin_creds()
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        if u == creds["username"] and check_password_hash(creds["password_hash"], p):
            session["admin"] = u
            flash("✅ Logged in successfully", "success")
            return redirect(url_for("dashboard"))
        flash("❌ Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login_page"))

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        save_admin_creds("admin", "ChangeMe123!")
        flash("🔁 Password reset to default (admin / ChangeMe123!)", "info")
        return redirect(url_for("login_page"))
    return render_template("forgot_password.html")

@app.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    creds = get_admin_creds()
    if request.method == "POST":
        old_pw = request.form.get("old_password", "")
        new_pw = request.form.get("new_password", "")
        confirm_pw = request.form.get("confirm_password", "")
        if not check_password_hash(creds["password_hash"], old_pw):
            flash("Old password incorrect", "danger")
        elif new_pw != confirm_pw:
            flash("New passwords do not match", "warning")
        elif len(new_pw) < 6:
            flash("Password too short (min 6 chars)", "warning")
        else:
            save_admin_creds(creds["username"], new_pw)
            flash("✅ Password changed successfully!", "success")
            return redirect(url_for("dashboard"))
    return render_template("change_password.html")

# ==============================
# EXPORT CSV
# ==============================
@app.route("/export/csv")
@login_required
def export_csv():
    data = []
    for f in RECORDINGS_DIR.glob("call_*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d = _normalize(d)
            data.append(d)
        except Exception as e:
            logging.warning(f"Skip {f}: {e}")
    if not data:
        abort(404)
    df = pd.json_normalize(data)
    out = RECORDINGS_DIR / "export.csv"
    df.to_csv(out, index=False)
    return send_file(out, as_attachment=True)

# ==============================
# DASHBOARD PAGES (thin wrappers)
# ==============================
@app.route("/dashboard")
@login_required
def dashboard():
    # Dashboard itself is served by dashboard.py (same process)
    # We just render template from here too for simplicity.
    # The template expects items etc.; dashboard.py fills them.
    # When dashboard.py is used as separate WSGI, this route
    # won’t be hit. Keeping it for compatibility.
    from dashboard import dashboard as dash_fn  # lazy import
    return dash_fn()

@app.route("/call/<cid>")
@login_required
def detail(cid):
    jf = RECORDINGS_DIR / f"call_{cid}.json"
    if not jf.exists():
        abort(404)
    d = json.loads(jf.read_text(encoding="utf-8"))
    d = _normalize(d)
    return render_template("detail.html", d=d, cid=cid)

@app.route("/report/<cid>.pdf")
@login_required
def report_pdf(cid):
    jf = RECORDINGS_DIR / f"call_{cid}.json"
    if not jf.exists():
        abort(404)
    d = json.loads(jf.read_text(encoding="utf-8"))
    d = _normalize(d)
    out = RECORDINGS_DIR / f"report_{cid}.pdf"

    c = canvas.Canvas(str(out), pagesize=A4)
    w, h = A4
    y = h - 40
    def line(t, dy=16):
        nonlocal y
        c.drawString(40, y, (t or "")[:120])
        y -= dy

    c.setFont("Helvetica-Bold", 14); line(f"Call Report — {cid}", 22)
    c.setFont("Helvetica", 11)
    line(f"Agent: {d.get('agent_name')} | Customer: {d.get('customer_phone')}")
    line(f"Language: {d.get('language_detected')} | Duration: {d.get('duration')} | Status: {d.get('call_status')}")
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
            c.showPage(); y = h - 40; c.setFont("Helvetica", 11)
    c.showPage(); c.save()
    return send_file(out, as_attachment=True)

# ==============================
# VOISO WEBHOOK
# ==============================
@app.route("/voiso-webhook", methods=["POST"])
def voiso_webhook():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        call = data.get("data", {}) or {}
        call_id = call.get("id") or datetime.utcnow().strftime("%Y%m%d%H%M%S")
        recording_url = call.get("recording")
        called_number = call.get("called_number", "Unknown")

        logging.info(f"📨 Webhook for Call ID: {call_id}")

        # Detect language as early as possible
        lang_guess = detect_language_from_country(called_number)

        # Normalize duration (may be dict or seconds)
        duration = call.get("duration", "N/A")
        if isinstance(duration, dict):
            # keep original structure; dashboard will pretty print talk_time
            pass

        call_json = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "call_id": call_id,
            "agent_name": (call.get("agent") or {}).get("name", "Unknown"),
            "customer_phone": called_number,
            "duration": duration,                       # keep as provided
            "call_status": call.get("disposition", "Unknown"),
            "cdr_url": call.get("cdr_url"),
            "recording_url": recording_url,
            "language_detected": lang_guess,
            "translation": {"english": "", "italian": ""},
            "scoring": {"total": 0, "missing": [], "comment": ""},
            "dialogue": []
        }

        jf = RECORDINGS_DIR / f"call_{call_id}.json"
        jf.write_text(json.dumps(call_json, indent=2))
        logging.info(f"💾 Metadata saved → {jf}")

        # Background processing (audio → transcript → translate → score)
        threading.Thread(target=process_voiso_call, args=(call_json, jf)).start()

        return jsonify({"status": "received"}), 200

    except Exception as e:
        logging.exception("Webhook error")
        return jsonify({"error": str(e)}), 500


def process_voiso_call(call_json: dict, jf_path: Path):
    """Download, transcribe, translate, and score the call — accuracy-first, safe on memory."""
    import subprocess, traceback, os

    try:
        recording_url = call_json.get("recording_url")
        if not recording_url:
            logging.warning("❌ No recording URL in payload.")
            return

        # === 1️⃣ Download recording ===
        audio_path = RECORDINGS_DIR / f"{jf_path.stem}.mp3"
        logging.info(f"⬇️ Downloading audio: {recording_url}")
        with requests.get(recording_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(audio_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"💾 Saved: {audio_path.name}")

        # === 2️⃣ Convert MP3 → WAV (more stable for Whisper) ===
        wav_path = audio_path.with_suffix(".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", str(wav_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            logging.info(f"🎧 Converted to WAV: {wav_path.name}")
            audio_for_asr = str(wav_path)
        except Exception as e:
            logging.warning(f"⚠️ FFmpeg conversion failed ({e}); using MP3 directly.")
            audio_for_asr = str(audio_path)

        # === 3️⃣ Transcribe ===
        logging.info(f"🗣️  Transcribing {os.path.basename(audio_for_asr)} …")
        segments, _ = whisper_model.transcribe(audio_for_asr, beam_size=3, vad_filter=True)

        text_segments = [seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()]
        if not text_segments:
            logging.warning("⚠️ No text decoded from audio.")
            record = json.loads(jf_path.read_text(encoding="utf-8"))
            record["translation"] = {"english": "", "italian": ""}
            record["scoring"] = {"total": 0, "missing": list(score_text('')[1]), "comment": "No speech detected"}
            record["dialogue"] = []
            jf_path.write_text(json.dumps(record, indent=2))
            return

        # === 4️⃣ Simple diarization (gap-based) ===
        dialogue = diarize(segments)
        full_text = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in dialogue])

        # === 5️⃣ Language detection & translation ===
        lang = (call_json.get("language_detected") or "en").lower()
        english_text = translate_to_english(full_text, lang)
        italian_text = translate_en_to_it(english_text)

        # === 6️⃣ Scoring ===
        score, missing, comment = score_text(english_text)

        # === 7️⃣ Update JSON record ===
        record = json.loads(jf_path.read_text(encoding="utf-8"))
        record["translation"] = {"english": english_text, "italian": italian_text}
        record["scoring"] = {"total": score, "missing": missing, "comment": comment}
        record["dialogue"] = dialogue
        jf_path.write_text(json.dumps(record, indent=2))

        logging.info(f"✅ Processed & saved → {jf_path.name} | Score={score}")

    except Exception as e:
        logging.error(f"❌ Error processing {jf_path.name}: {e}\n{traceback.format_exc()}")

    finally:
        # === 8️⃣ Cleanup temp files ===
        for f in [audio_path, wav_path]:
            try:
                if f.exists():
                    f.unlink()
                    logging.debug(f"🧹 Deleted temp: {f}")
            except Exception:
                pass


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    logging.info("🚀 Running → http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
