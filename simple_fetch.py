# simple_fetch.py - Minimal version to test one recording download
import requests

CALL_CENTER_API_KEY = "d34e2a3e83d34a32cb55ad178def0d00f9e748a2f8209280"
CLUSTER_ID = "cc-rtm01"

list_url = f"https://{CLUSTER_ID}.voiso.com/cdr/list"
headers = {"X-API-KEY": CALL_CENTER_API_KEY}

print("Getting latest call...")
resp = requests.get(list_url, headers=headers, params={"limit": 1}, timeout=30)

if resp.status_code == 200:
    # Try to parse as JSON
    try:
        data = resp.json()
        calls = data.get("cdrs", [])
        if calls:
            call = calls[0]
            recording_url = call.get("recording_url")
            call_id = call.get("id")
            
            if recording_url:
                print(f"Downloading recording for call {call_id}...")
                rec_resp = requests.get(recording_url, headers=headers, timeout=30)
                
                if rec_resp.status_code == 200:
                    filename = "test_recording.mp3"
                    with open(filename, "wb") as f:
                        f.write(rec_resp.content)
                    print(f"✅ Success! Saved as {filename}")
                    print(f"File size: {len(rec_resp.content)} bytes")
                else:
                    print(f"Download failed: {rec_resp.status_code}")
            else:
                print("No recording URL found")
        else:
            print("No calls found")
    except:
        print("Response is not JSON, saving as HTML for analysis...")
        with open("response.html", "w") as f:
            f.write(resp.text)
        print("Saved response as response.html")
else:
    print(f"Error: {resp.status_code}")