from fastapi import FastAPI, Request
import requests
import os
from datetime import datetime

app = FastAPI()

# Folder jahan mp3 save hongi
SAVE_DIR = "recordings"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/voiso-webhook")
async def voiso_webhook(request: Request):
    try:
        payload = await request.json()
        print("📩 Webhook Received:", payload)

        # Extract call data
        call_data = payload.get("data", {})
        call_id = call_data.get("id") or datetime.utcnow().strftime("%Y%m%d%H%M%S")

        # Recording URL ka sahi field
        recording_url = call_data.get("recording")

        if recording_url:
            filename = f"{SAVE_DIR}/call_{call_id}.mp3"
            print(f"🎧 Downloading {recording_url} -> {filename}")

            resp = requests.get(recording_url, timeout=30)
            if resp.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"✅ Recording saved: {filename} (size {len(resp.content)} bytes)")
                return {"status": "ok", "file": filename}
            else:
                print("❌ Error downloading recording:", resp.status_code, resp.text)
                return {"status": "error", "msg": "download failed"}
        else:
            print("⚠️ No recording found in webhook payload")
            return {"status": "error", "msg": "no recording in payload"}

    except Exception as e:
        print("❌ Webhook processing failed:", e)
        return {"status": "error", "msg": str(e)}
