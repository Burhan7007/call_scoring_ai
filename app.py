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

# ==============================
# PATHS / ENV
# ==============================
ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
MODELS_DIR = ROOT / "models"
CREDS_FILE = ROOT / "admin_creds.json"

RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# WHISPER
# ==============================
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
COMPUTE = "float16" if USE_GPU else "int8"

print(f"🎧 Loading Whisper model (small) [{DEVICE}, {COMPUTE}]...")
whisper_model = WhisperModel("small", device=DEVICE, compute_type=COMPUTE)

# ==============================
# TRANSLATION MODELS
# ==============================
HF_CACHE = MODELS_DIR / "hf"
TRANSLATORS = {}
TO_EN = {"it": "Helsinki-NLP/opus-mt-it-en", "es": "Helsinki-NLP/opus-mt-es-en", "bg": "Helsinki-NLP/opus-mt-bg-en"}
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
    - Alternates Agent/Client based on silence or repetition.
    - Forces speaker switch after a few turns if one side talks too long.
    """
    dialogue, buf, cur, start, last, run_len = [], [], "Agent", 0.0, 0.0, 0

    def flush(b, s, st, en):
        if b:
            text = " ".join(b).strip()
            if text:
                dialogue.append({"speaker": s, "text": text, "start": st, "end": en})

    for seg in raw:
        gap = seg.start - last
        duration = seg.end - seg.start

        # Switch speaker if silence or repetition indicates turn-taking
        if gap >= pause or run_len >= max_agent_run or len(buf) > 4 and seg.text.strip().endswith("?"):
            flush(buf, cur, start, last)
            buf, start = [], seg.start
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

def process_audio(file_path: Path, uuid=None):
    print(f"🎧 Transcribing {file_path.name}")
    try:
        # Pass 1: Transcribe with VAD filter for cleaner segmentation
        segments, info = whisper_model.transcribe(str(file_path), vad_filter=True, beam_size=5)
        raw = [s for s in segments if s.text.strip()]
        txt = " ".join([s.text.strip() for s in raw])

        # Pass 2 fallback: if too short or repetitive, retry without VAD
        if len(txt.split()) < 15 or len(set(txt.split())) < 5:
            print("⚠️ Fallback pass — reprocessing without VAD (to capture missed speech)")
            segments, info = whisper_model.transcribe(str(file_path), vad_filter=False, beam_size=5)
            raw = [s for s in segments if s.text.strip()]
            txt = " ".join([s.text.strip() for s in raw])

        # Apply diarization (Agent/Client split)
        dialogue = diarize(raw)

        # Fallback: alternate if only one speaker detected
        if len({d["speaker"] for d in dialogue}) < 2:
            print("⚠️ Diarization fallback — alternating Agent/Client")
            dialogue = []
            cur = "Agent"
            for idx, seg in enumerate(raw):
                dialogue.append({
                    "speaker": cur,
                    "text": seg.text.strip(),
                    "start": seg.start,
                    "end": seg.end
                })
                if idx % 2 == 1:
                    cur = "Client" if cur == "Agent" else "Agent"

        # Detect language from Whisper or phone prefix
        lang = (info.language or "").lower() or "en"
        cdr = fetch_voiso(uuid) if uuid else {}
        if lang == "unknown" or not lang:
            lang = detect_language_from_country(cdr.get("to") or cdr.get("from"))

        # Build full transcript from dialogue text
        combined_text = " ".join([d["text"] for d in dialogue if d.get("text")]).strip()

        # Fallback if Whisper missed speech
        if len(combined_text.split()) < 20:
            print("⚠️ Transcript too short — regenerating from raw segments")
            combined_text = " ".join([s.text.strip() for s in raw if s.text.strip()])

        # ✅ Translate for both Agent and Client parts separately
        agent_text = " ".join([x["text"] for x in dialogue if x["speaker"] == "Agent"]).strip()
        client_text = " ".join([x["text"] for x in dialogue if x["speaker"] == "Client"]).strip()

        if lang == "en":
            en_agent, en_client = agent_text, client_text
            en_full = combined_text
        else:
            en_agent = translate_to_english(agent_text, lang) if agent_text else ""
            en_client = translate_to_english(client_text, lang) if client_text else ""
            en_full = translate_to_english(combined_text, lang)

        # Italian translations (Agent/Client divided too)
        it_agent = translate_en_to_it(en_agent)
        it_client = translate_en_to_it(en_client)
        it_full = translate_en_to_it(en_full)

        # Improved scoring logic — avoid false zeros
        meaningful = any(k in en_full.lower() for k in [
            "hello", "thank", "confirm", "product", "address", "please", "order",
            "good morning", "afternoon", "call", "customer", "service"
        ])
        word_count = len(en_full.split())

        if word_count < 20:
            total, missing, comment = 0, [], "Low-content or very short call (under 20 words)"
        elif not meaningful and word_count < 40:
            total, missing, comment = 0, [], "Likely non-meaningful / background noise"
        else:
            total, missing, comment = score_text(en_full)

        # Build final structured result
        res = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent_name": cdr.get("agent", "Unknown"),
            "customer_phone": cdr.get("to") or cdr.get("from", "Unknown"),
            "duration": cdr.get("duration", "N/A"),
            "call_status": cdr.get("disposition", "Unknown"),
            "language_detected": lang,
            "translation": {
                "english": {
                    "agent": en_agent,
                    "client": en_client,
                    "full": en_full
                },
                "italian": {
                    "agent": it_agent,
                    "client": it_client,
                    "full": it_full
                }
            },
            "dialogue": dialogue,
            "scoring": {"total": total, "missing": missing, "comment": comment},
        }

        # Save output JSON
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
    status_q = (request.args.get("status") or "answered").strip().lower()

    all_rows, items = [], []

    for f in RECORDINGS_DIR.glob("call_*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d["_id"] = f.stem.replace("call_", "")
            d["agent_name"] = d.get("agent_name") or "Unknown"
            d["language_detected"] = d.get("language_detected") or "Unknown"
            d["call_status"] = d.get("call_status") or "Unknown"
            d["duration_display"] = str(d.get("duration", "N/A"))
            all_rows.append(d)
        except Exception as e:
            print("⚠️ Error reading file:", f.name, e)

    agents = sorted({r["agent_name"] for r in all_rows})
    langs = sorted({r["language_detected"] for r in all_rows})
    statuses = sorted({r["call_status"] for r in all_rows})

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
    return render_template("index.html", items=items, agents=agents, langs=langs, statuses=statuses)

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
    s = d.get("scoring", {})
    line(f"Score: {s.get('total',0)} / 100")
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

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    print("🚀 Server → http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)