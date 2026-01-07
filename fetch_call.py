import requests
import json
from datetime import datetime, timedelta

# API key and cluster
CDR_API_KEY = "3dc9442851a083885a85a783329b9552e0406864cba34b62"
CLUSTER_ID = "cc-rtm01"

# Correct endpoint with v2
list_url = f"https://{CLUSTER_ID}.voiso.com/api/v2/cdr"

# Date range (last 7 days)
date_to = datetime.utcnow()
date_from = date_to - timedelta(days=7)

params = {
    "key": CDR_API_KEY,
    "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "date_to": date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "limit": 3
}

print("Fetching CDR list...")
resp = requests.get(list_url, params=params, timeout=30)

print("Status Code:", resp.status_code)

if resp.status_code != 200:
    print("❌ Error fetching CDRs:", resp.text)
    exit()

try:
    data = resp.json()
    print("\n✅ JSON Response:\n", json.dumps(data, indent=2))
except Exception as e:
    print("❌ Failed to parse JSON:", e)
    print("Raw Response:", resp.text)
    exit()

cdrs = data.get("cdrs", [])
if not cdrs:
    print("⚠️ No calls found in this date range.")
    exit()

# Pick first call
call = cdrs[0]
call_id = call.get("id")
operator = call.get("operator_name")
duration = call.get("duration_sec")
recording_url = call.get("recording_url")

print("\n=== Latest Call Details ===")
print("Call ID:", call_id)
print("Operator:", operator)
print("Duration (sec):", duration)
print("Recording URL:", recording_url)

# Download recording if available
if recording_url:
    print("\nDownloading recording...")
    rec_resp = requests.get(recording_url, timeout=30)
    if rec_resp.status_code == 200:
        filename = f"call_{call_id}.mp3" if call_id else "test_call.mp3"
        with open(filename, "wb") as f:
            f.write(rec_resp.content)
        print(f"✅ Recording saved as {filename}")
        print(f"File size: {len(rec_resp.content)} bytes")
    else:
        print("❌ Error downloading recording:", rec_resp.status_code, rec_resp.text)
else:
    print("⚠️ No recording available for this call.")
