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

# ==============================
# PATHS & DIRECTORIES
# ==============================
ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# ADMIN LOGIN SYSTEM
# ==============================
CREDS_FILE = ROOT / "admin_creds.json"

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
    print(f"✅ Password updated for {username}")

# ==============================
# FLASK SETUP
# ==============================
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace_this_secret_key")

def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

# ==============================
# HELPERS
# ==============================
def _safe_json_load(p: Path):
    try:
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            return None
        return json.loads(txt)
    except Exception:
        return None

def load_all_calls():
    rows = []
    for f in RECORDINGS_DIR.glob("call_*.json"):
        d = _safe_json_load(f)
        if not d:
            continue
        d["_id"] = f.stem.replace("call_", "")
        d.setdefault("agent_name", "Unknown")
        d.setdefault("customer_phone", "Unknown")
        d.setdefault("duration", "N/A")
        d.setdefault("call_status", "Unknown")
        d.setdefault("language_detected", "Unknown")
        d.setdefault("translation", {"english": "", "italian": ""})
        d.setdefault("scoring", {"total": 0, "missing": [], "comment": ""})
        rows.append(d)
    return sorted(rows, key=lambda x: x.get("timestamp", ""), reverse=True)

# ==============================
# AUTH ROUTES
# ==============================
@app.route("/login", methods=["GET", "POST"])
def login():
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
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))

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

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        save_admin_creds("admin", "ChangeMe123!")
        flash("🔁 Password reset to default (admin / ChangeMe123!)", "info")
        return redirect(url_for("login"))
    return render_template("forgot_password.html")

# ==============================
# DASHBOARD & DATA ROUTES
# ==============================
@app.route("/")
def home():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
@login_required
def dashboard():
    q = (request.args.get("q") or "").strip().lower()
    agent_q = (request.args.get("agent") or "").strip().lower()
    lang_q = (request.args.get("lang") or "").strip().lower()
    phone_q = (request.args.get("phone") or "").strip()
    status_q = (request.args.get("status") or "").strip().lower()

    all_calls = load_all_calls()
    filtered = []

    for r in all_calls:
        if q and q not in r["translation"]["english"].lower():
            continue
        if agent_q and agent_q not in r["agent_name"].lower():
            continue
        if lang_q and lang_q != r["language_detected"].lower():
            continue
        if phone_q and phone_q not in r["customer_phone"]:
            continue
        if status_q and status_q != r["call_status"].lower():
            continue
        filtered.append(r)

    agents = sorted(set([r["agent_name"] for r in all_calls]))
    langs = sorted(set([r["language_detected"] for r in all_calls]))
    statuses = sorted(set([r["call_status"] for r in all_calls]))

    return render_template(
        "index.html",
        items=filtered, agents=agents, langs=langs, statuses=statuses,
        q=q, agent=agent_q, lang=lang_q, phone=phone_q, status=status_q
    )

@app.route("/call/<cid>")
@login_required
def call_detail(cid):
    jf = RECORDINGS_DIR / f"call_{cid}.json"
    if not jf.exists():
        abort(404)
    d = _safe_json_load(jf)
    if not d:
        abort(404)
    return render_template("detail.html", d=d, cid=cid)

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
    line(f"Language: {d.get('language_detected', 'Unknown')} | Duration: {d.get('duration', 'N/A')} | Status: {d.get('call_status', 'Unknown')}")
    s = d.get("scoring", {})
    line(f"Score: {s.get('total', 0)} / 100")
    if s.get("missing"):
        line("Missing KPIs: " + ", ".join(s["missing"]))
    line("")
    line("Transcript (EN):", 18)
    for chunk in (d["translation"].get("english", "")).split("\n"):
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

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("🖥️ Dashboard running at http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)
