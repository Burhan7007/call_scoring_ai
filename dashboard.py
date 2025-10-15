import os, json
from pathlib import Path
from datetime import datetime
from functools import wraps
import pandas as pd
from flask import (
    Flask, render_template, send_file, redirect, url_for,
    request, abort, session, flash, jsonify
)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.security import generate_password_hash, check_password_hash

ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace_this_secret_key")

def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

# ---- helpers ----
def _safe_json_load(p: Path):
    try:
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            return None
        return json.loads(txt)
    except Exception:
        return None

def _format_mmss(seconds: int) -> str:
    if seconds is None:
        return "0s"
    try:
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"
    except Exception:
        return "0s"

def _pretty_duration(dur):
    """
    Voiso sends duration sometimes as:
      {"total": 118, "dialing_time": 11, "talk_time": 108}
    We want to show talk_time as mm:ss.
    If a plain int is sent, show that.
    """
    if isinstance(dur, dict):
        talk = dur.get("talk_time")
        return _format_mmss(talk)
    if isinstance(dur, (int, float, str)) and str(dur).isdigit():
        return _format_mmss(int(dur))
    return str(dur)

def load_all_calls():
    rows = []
    for f in RECORDINGS_DIR.glob("call_*.json"):
        d = _safe_json_load(f)
        if not d:
            continue
        d["_id"] = f.stem.replace("call_", "")
        # defaults
        d.setdefault("agent_name", "Unknown")
        d.setdefault("customer_phone", "Unknown")
        d.setdefault("duration", "N/A")
        d.setdefault("call_status", "Unknown")
        d.setdefault("language_detected", "Unknown")
        d.setdefault("translation", {"english": "", "italian": ""})
        d.setdefault("scoring", {"total": 0, "missing": [], "comment": ""})

        # 👇 override duration for display
        d["duration_display"] = _pretty_duration(d.get("duration"))
        rows.append(d)
    return sorted(rows, key=lambda x: x.get("timestamp", ""), reverse=True)

# ---- AUTH ROUTES (kept minimal for compatibility with app.py) ----
@app.route("/login", methods=["GET", "POST"])
def login():
    # app.py handles full auth; this exists for direct dashboard run
    creds_file = ROOT / "admin_creds.json"
    if request.method == "POST":
        if creds_file.exists():
            creds = json.loads(creds_file.read_text())
            u = request.form.get("username","").strip()
            p = request.form.get("password","").strip()
            if u == creds.get("username") and check_password_hash(creds.get("password_hash",""), p):
                session["admin"] = u
                return redirect(url_for("dashboard"))
        flash("Invalid login", "danger")
    return render_template("login.html")

@app.route("/")
def home():
    return redirect(url_for("dashboard"))

# ---- DASHBOARD ----
@app.route("/dashboard")
@login_required
def dashboard():
    q = (request.args.get("q") or "").strip().lower()
    agent_q = (request.args.get("agent") or "").strip().lower()
    lang_q = (request.args.get("lang") or "").strip().lower()
    phone_q = (request.args.get("phone") or "").strip()
    # ✅ default to "answered" if not specified
    status_q = (request.args.get("status") or "answered").strip().lower()

    all_calls = load_all_calls()
    filtered = []

    for r in all_calls:
        if q and q not in r["translation"]["english"].lower():
            continue
        if agent_q and agent_q not in (r.get("agent_name") or "").lower():
            continue
        if lang_q and lang_q != (r.get("language_detected") or "").lower():
            continue
        if phone_q and phone_q not in (r.get("customer_phone") or ""):
            continue
        # ✅ default status applied here
        if status_q and status_q != (r.get("call_status") or "").lower():
            continue
        filtered.append(r)

    agents = sorted(set([r.get("agent_name") for r in all_calls if r.get("agent_name")]))
    langs = sorted(set([r.get("language_detected") for r in all_calls if r.get("language_detected")]))
    statuses = sorted(set([r.get("call_status") for r in all_calls if r.get("call_status")]))

    # NOTE: your template likely prints {{ item.duration }}.
    # To align column data to header, print {{ item.duration_display }}
    # in templates/index.html where Duration column is rendered.
    return render_template(
        "index.html",
        items=filtered, agents=agents, langs=langs, statuses=statuses,
        q=q, agent=agent_q, lang=lang_q, phone=phone_q, status=status_q
    )

# ---- CSV / PDF (kept for feature parity) ----
@app.route("/export/csv")
@login_required
def export_csv():
    calls = load_all_calls()
    if not calls:
        abort(404)
    df = pd.json_normalize(calls)
    out = RECORDINGS_DIR / "export.csv"
    df.to_csv(out, index=False)
    return send_file(out, as_attachment=True)

@app.route("/report/<cid>.pdf")
@login_required
def report_pdf(cid):
    jf = RECORDINGS_DIR / f"call_{cid}.json"
    if not jf.exists():
        abort(404)
    d = _safe_json_load(jf)
    if not d:
        abort(404)

    out = RECORDINGS_DIR / f"report_{cid}.pdf"
    c = canvas.Canvas(str(out), pagesize=A4)
    w, h = A4
    y = h - 40
    def line(txt, dy=18):
        nonlocal y
        c.drawString(40, y, txt)
        y -= dy

    c.setFont("Helvetica-Bold", 14)
    line(f"Call Report — {cid}", 22)
    c.setFont("Helvetica", 11)
    line(f"Timestamp: {d.get('timestamp', '')}")
    line(f"Agent: {d.get('agent_name', 'Unknown')} | Customer: {d.get('customer_phone', 'Unknown')}")
    line(f"Language: {d.get('language_detected', 'Unknown')} | Duration: {_pretty_duration(d.get('duration'))} | Status: {d.get('call_status', 'Unknown')}")
    s = d.get("scoring", {})
    line(f"Score: {s.get('total', 0)} / 100")
    if s.get("missing"):
        line("Missing KPIs: " + ", ".join(s["missing"]))
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

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200
