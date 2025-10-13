from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import cm

def _wrap_text(text: str, max_chars: int = 95) -> List[str]:
    lines = []
    for para in (text or "").split("\n"):
        p = para.strip()
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars) or max_chars
            lines.append(p[:cut])
            p = p[cut:].lstrip()
        if p:
            lines.append(p)
    return lines

def _draw_kpi_bar(c: canvas.Canvas, x, y, w, h, val, vmax):
    ratio = max(0.0, min(1.0, (val or 0) / float(vmax or 1)))
    c.setFillColor(colors.HexColor("#3b82f6"))
    c.rect(x, y, w * ratio, h, fill=True, stroke=0)
    c.setStrokeColor(colors.black)
    c.rect(x, y, w, h, fill=False, stroke=1)

def generate_pdf(result: Dict[str, Any], out_path: Path):
    out_path = Path(out_path)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    W, H = A4
    margin = 2 * cm
    x = margin
    y = H - margin

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "AI Call Scoring Report")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Call ID: {result.get('call_id','')}")
    y -= 14
    c.drawString(x, y, f"Timestamp (UTC): {result.get('timestamp','')}")
    y -= 14
    c.drawString(x, y, f"Detected Language: {result.get('language_detected','')}")
    y -= 18

    # Summary Scores
    scoring = result.get("scoring", {})
    total = scoring.get("total", 0)
    by_section = scoring.get("by_section", {})
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, f"Total Score: {total}")
    y -= 18

    bar_w, bar_h = (W - 2*margin - 120), 10
    c.setFont("Helvetica", 10)
    for section, val in by_section.items():
        if y < margin + 120:
            c.showPage(); y = H - margin
        c.drawString(x, y, section.replace("_", " ").title())
        _draw_kpi_bar(c, x + 120, y - 2, bar_w, bar_h, val, 40 if "clarity" in section else 30)
        c.drawString(x + 120 + bar_w + 8, y, str(val))
        y -= 16

    # Comment
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Comment:")
    y -= 14
    c.setFont("Helvetica", 10)
    for line in _wrap_text(scoring.get("comment","")):
        if y < margin + 80:
            c.showPage(); y = H - margin
            c.setFont("Helvetica", 10)
        c.drawString(x, y, line)
        y -= 12

    # Translations
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "English Translation:")
    y -= 14
    c.setFont("Helvetica", 9)
    for line in _wrap_text(result.get("translation", {}).get("english","")):
        if y < margin + 80:
            c.showPage(); y = H - margin; c.setFont("Helvetica", 9)
        c.drawString(x, y, line); y -= 11

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Italian Translation:")
    y -= 14
    c.setFont("Helvetica", 9)
    for line in _wrap_text(result.get("translation", {}).get("italian","")):
        if y < margin + 80:
            c.showPage(); y = H - margin; c.setFont("Helvetica", 9)
        c.drawString(x, y, line); y -= 11

    # Transcript (speaker-labeled)
    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Transcript:")
    y -= 14
    c.setFont("Helvetica", 9)
    for seg in result.get("stt", {}).get("segments", []):
        line = f"{seg.get('speaker','?')}: {seg.get('text','').strip()}"
        for wl in _wrap_text(line, max_chars=110):
            if y < margin + 60:
                c.showPage(); y = H - margin; c.setFont("Helvetica", 9)
            c.drawString(x, y, wl); y -= 11

    c.showPage()
    c.save()
