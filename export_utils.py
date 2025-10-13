"""
export_utils.py
Export call scoring results to CSV / Excel for Milestone 2
----------------------------------------------------------
Uses pandas and openpyxl to generate summary or detailed reports.
"""

import json
import pandas as pd
from pathlib import Path


def export_to_csv(results: list, out_path: str | Path) -> Path:
    """
    Exports multiple call results (dicts) to a CSV file.
    Each row: call_id, language, total score, comment.
    """
    rows = []
    for r in results:
        s = r.get("scoring", {})
        rows.append({
            "Call ID": r.get("call_id"),
            "Language": r.get("language_detected"),
            "Score Total": s.get("total", 0),
            "Comment": s.get("comment", ""),
        })
    df = pd.DataFrame(rows)
    out_path = Path(out_path).with_suffix(".csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Exported CSV → {out_path}")
    return out_path


def export_to_excel(results: list, out_path: str | Path, detailed: bool = True) -> Path:
    """
    Exports results to Excel workbook.
    Sheet 1: summary table
    Sheet 2+: individual call breakdowns if detailed=True
    """
    out_path = Path(out_path).with_suffix(".xlsx")
    writer = pd.ExcelWriter(out_path, engine="openpyxl")

    # --- Summary sheet
    summary = []
    for r in results:
        s = r.get("scoring", {})
        summary.append({
            "Call ID": r.get("call_id"),
            "Language": r.get("language_detected"),
            "Score Total": s.get("total", 0),
            "Comment": s.get("comment", ""),
        })
    pd.DataFrame(summary).to_excel(writer, index=False, sheet_name="Summary")

    # --- Detailed sheets
    if detailed:
        for r in results:
            sid = r.get("call_id", "call")
            s = r.get("scoring", {})
            df = pd.DataFrame(list(s.get("by_section", {}).items()), columns=["Section", "Score"])
            df["Missing KPIs"] = ", ".join(s.get("missing_kpis", []))
            df.to_excel(writer, index=False, sheet_name=sid[:28])  # Excel sheet name limit

    writer.close()
    print(f"✅ Exported Excel → {out_path}")
    return out_path


# Quick manual test
if __name__ == "__main__":
    sample_json = {
        "call_id": "demo_1.wav",
        "language_detected": "en",
        "scoring": {
            "total": 80,
            "by_section": {"trust_relationship": 25, "clarity_accuracy": 35, "value_upsell": 20},
            "missing_kpis": ["Recap"],
            "comment": "Recap missing."
        }
    }
    export_to_csv([sample_json], "test_report")
    export_to_excel([sample_json], "test_report")
