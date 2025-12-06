from typing import Dict

from semantic_ai import _clean_text  # reuse helper


def detect_order_status(agent_text: str, client_text: str) -> Dict:
    """
    Returns:
      {
        "status": "accepted" | "refused" | "to_recall" | "unknown",
        "confidence": float,
        "flags": {...},
        "rejection": {
            "type": "cancelled_in_call" | "fake_lead" | "data_error" | "commercial_objection" | None,
            "reason": str
        }
      }
    """
    a = _clean_text(agent_text or "")
    c = _clean_text(client_text or "")
    full = f"{a} {c}"

    status = "unknown"
    conf = 0.3
    flags = {
        "callback_requested": False,
        "too_short": False,
    }
    rejection = {"type": None, "reason": ""}

    words = full.split()
    if len(words) < 8:
        flags["too_short"] = True

    # ----- TO RECALL -------------------------------------------------
    recall_markers = [
        "call me later",
        "another time",
        "i can't talk now",
        "i am busy",
        "call back",
        "tomorrow please",
        "later please",
    ]
    if any(m in full for m in recall_markers):
        status = "to_recall"
        conf = 0.8
        flags["callback_requested"] = True
        return {
            "status": status,
            "confidence": conf,
            "flags": flags,
            "rejection": rejection,
        }

    # ----- ACCEPTED ---------------------------------------------------
    accept_markers = [
        "yes i confirm",
        "yes i accept",
        "it's ok",
        "order is confirmed",
        "i will receive the order",
        "okay send it",
        "ok send it",
        "yes send it",
        "yes i want it",
        "i want the product",
    ]
    if any(m in full for m in accept_markers):
        status = "accepted"
        conf = 0.8

    # ----- REFUSED ----------------------------------------------------
    refuse_markers = [
        "i don't want",
        "i am not interested",
        "not interested",
        "cancel the order",
        "i did not order anything",
        "i never ordered",
        "wrong number",
        "i don't know this",
        "i won't buy",
        "too expensive",
    ]
    if any(m in full for m in refuse_markers):
        status = "refused"
        conf = max(conf, 0.85)

        # Now classify the REFUSED reason for Milestone 2
        txt = full

        # Fake / non-real lead
        if "never ordered" in txt or "did not order" in txt or "i didn't order" in txt:
            rejection["type"] = "fake_lead"
            rejection["reason"] = "Customer says they never ordered."

        # Data error
        elif "wrong number" in txt or "you called the wrong person" in txt:
            rejection["type"] = "data_error"
            rejection["reason"] = "Wrong number / data error."

        # Cancelled in call
        elif "cancel the order" in txt or "i want to cancel" in txt:
            rejection["type"] = "cancelled_in_call"
            rejection["reason"] = "Customer explicitly cancelled the order."

        # Commercial objection (price, timing, product)
        elif "too expensive" in txt or "can't afford" in txt or "no money" in txt:
            rejection["type"] = "commercial_objection"
            rejection["reason"] = "Price / commercial objection."
        else:
            rejection["type"] = "commercial_objection"
            rejection["reason"] = "Generic refusal / objection."

    return {
        "status": status,
        "confidence": conf,
        "flags": flags,
        "rejection": rejection,
    }
