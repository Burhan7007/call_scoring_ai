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
# WHISPER
# ==============================
USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
COMPUTE = "float16" if USE_GPU else "int8"

print(f"🎧 Loading Whisper model (small) [{DEVICE}, {COMPUTE}]...")
whisper_model = WhisperModel("small", device=DEVICE, compute_type=COMPUTE)

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
    out = mdl.generate(**batch, max_new_tokens=512)
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

    def clean_repetition(t):
        # Remove repeated tokens like "no no no no no..."
        words = t.split()
        cleaned = []
        last_word = ""
        for w in words:
            if w.lower() != last_word:
                cleaned.append(w)
            last_word = w.lower()
        return " ".join(cleaned)

    def flush(b, s, st, en):
        if not b:
            return
        text = clean_repetition(" ".join(b).strip())
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

def process_audio(file_path: Path, uuid=None):
    from sentence_transformers import util as st_util

    # -----------------------------
    # helper: semantic similarity
    # -----------------------------
    def sem_sim(a: str, b: str) -> float:
        if not a or not b:
            return 0
        try:
            e1 = embedder.encode(a, convert_to_tensor=True)
            e2 = embedder.encode(b, convert_to_tensor=True)
            return float(st_util.cos_sim(e1, e2)[0][0])
        except:
            return 0

    def clean(t):
        t = t.replace("..", ".").replace("...", ".")
        return " ".join(t.split()).strip()

    print(f"🎧 Transcribing {file_path.name}")

    try:
        # ----------------------------------------------------
        # 1. TRANSCRIBE
        # ----------------------------------------------------
        segments, info = whisper_model.transcribe(str(file_path), vad_filter=True, beam_size=5)
        raw = [s for s in segments if s.text.strip()]
        txt = " ".join([s.text.strip() for s in raw])

        # retry without VAD if too short
        if len(txt.split()) < 12:
            print("⚠ Weak transcription → retrying without VAD")
            segments, info = whisper_model.transcribe(str(file_path), vad_filter=False, beam_size=5)
            raw = [s for s in segments if s.text.strip()]
            txt = " ".join([s.text.strip() for s in raw])

        blank_call = len(txt.split()) < 5

        # ----------------------------------------------------
        # 2. DIARIZATION
        # ----------------------------------------------------
        dialogue = diarize(raw)

        # if diarization fails → fallback
        if len({d["speaker"] for d in dialogue}) < 2:
            dialogue = []
            cur = "Agent"
            for i, seg in enumerate(raw):
                dialogue.append({
                    "speaker": cur,
                    "text": seg.text.strip(),
                    "start": seg.start,
                    "end": seg.end
                })
                if i % 2 == 1:
                    cur = "Client" if cur == "Agent" else "Agent"

        total_duration = raw[-1].end if raw else 0
        spoken = sum(d["end"] - d["start"] for d in dialogue)
        dialogue_score = int(min(100, (spoken / total_duration) * 100)) if total_duration else 0

        # ----------------------------------------------------
        # 3. CDR + LANGUAGE
        # ----------------------------------------------------
        cdr = fetch_voiso(uuid) if uuid else {}

        lang = (info.language or "").lower().strip()
        if lang in ("", "unknown"):
            lang = detect_language_from_country(cdr.get("to") or cdr.get("from"))

        if lang.startswith("sl"): lang = "sl"
        if lang.startswith(("hr", "bs", "sr")): lang = "hr"
        if lang.startswith("el"): lang = "gr"

        # ----------------------------------------------------
        # 4. TEXT PREP
        # ----------------------------------------------------
        agent_lines = [d["text"] for d in dialogue if d["speaker"] == "Agent"]
        client_lines = [d["text"] for d in dialogue if d["speaker"] == "Client"]

        combined_text = " ".join(agent_lines + client_lines)
        if len(combined_text.split()) < 20:
            combined_text = txt

        # ----------------------------------------------------
        # 5. TRANSLATION
        # ----------------------------------------------------
        en_lines = []
        it_lines = []

        for turn in dialogue:
            sp = turn["speaker"]
            orig = clean(turn["text"])

            # English translation
            if lang == "en":
                en_t = orig
            else:
                en_t = clean(translate_to_english(orig, lang))

            # Italian translation
            it_t = clean(translate_en_to_it(en_t))

            en_lines.append(f"- {sp}: {en_t}")
            it_lines.append(f"- {'Agente' if sp == 'Agent' else 'Cliente'}: {it_t}")

        en_agent = [l.replace("- Agent: ", "") for l in en_lines if l.startswith("- Agent:")]
        en_client = [l.replace("- Client: ", "") for l in en_lines if l.startswith("- Client:")]
        combined_en = " ".join(en_agent + en_client).lower()

        # speech metrics
        total_words = len(combined_en.split())
        agent_spoke = len(en_agent) > 0
        client_spoke = len(en_client) > 0

        # ----------------------------------------------------
        # 6. KPI SCORE — KEEP EXACTLY AS YOUR ORIGINAL LOGIC
        # ----------------------------------------------------
        score_val, missing, comment = score_text(combined_en)

        # ----------------------------------------------------
        # 7. ORDER STATUS — FIXED VERSION (CLIENT COMPLAINT SOLVED)
        # ----------------------------------------------------
        disp = (cdr.get("disposition") or "").lower()
        system_dispo = ["abandon", "abandoned", "dialer_abandoned", "failed", "busy", "no answer"]

        # RULE 1 — If client NEVER spoke, cannot be accepted or refused
        if not client_spoke:
            order_status = "recall"

        # RULE 2 — System dispositions ALWAYS recall
        elif any(x in disp for x in system_dispo):
            order_status = "recall"

        # RULE 3 — Very small conversation → recall
        elif total_words < 15 or dialogue_score < 20 or blank_call:
            order_status = "recall"

        else:
            # USE INTENT DETECTION
            accept_patterns = [
                "i confirm", "i accept", "yes i confirm", "i agree",
                "send it", "ok send", "i will receive", "i want it",
                "proceed with the order", "yes the order"
            ]

            refuse_patterns = [
                "i don't want", "i refuse", "cancel the order",
                "not interested", "stop the order", "i do not want",
                "no i don't want"
            ]

            recall_patterns = [
                "call me later", "call later", "later please",
                "not now", "try again later"
            ]

            def match_any(text, patterns, th=0.25):
                return any(sem_sim(text, p) >= th for p in patterns)

            # detect acceptance
            if match_any(combined_en, accept_patterns):
                order_status = "accepted"

            elif match_any(combined_en, refuse_patterns):
                order_status = "refused"

            elif match_any(combined_en, recall_patterns):
                order_status = "recall"

            else:
                order_status = "recall"

        # ----------------------------------------------------
        # 8. REJECTION REASON — Milestone 2
        # ----------------------------------------------------
        rejection_reason = "-"
        if order_status == "refused":
            reasons = {
                "cancelled": "customer confirmed but later cancelled",
                "fake": "customer says they never ordered anything",
                "error": "wrong number or wrong customer",
                "objection": "customer refuses due to price or distrust",
            }

            best_score, best_label = 0, "-"
            for label, desc in reasons.items():
                s = sem_sim(combined_en, desc)
                if s > best_score:
                    best_score = s
                    best_label = label

            rejection_reason = best_label

        # ----------------------------------------------------
        # 9. SAVE OUTPUT
        # ----------------------------------------------------
        res = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent_name": cdr.get("agent", "Unknown"),
            "customer_phone": cdr.get("to") or cdr.get("from", "Unknown"),
            "duration": cdr.get("duration", "N/A"),
            "call_status": cdr.get("disposition", "Unknown"),
            "language_detected": lang,
            "order_status": order_status,
            "rejection_reason": rejection_reason,
            "blank_call": blank_call,
            "dialogue_score": dialogue_score,
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


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    print("🚀 Server → http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)