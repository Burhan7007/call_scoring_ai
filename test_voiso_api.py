import requests
import json

url = "https://cc-rtm01.voiso.com/api/v2/cdr?key=3dc9442851a083885a85a783329b9552e0406864cba34b62&limit=3"

try:
    resp = requests.get(url, timeout=30)
    print("Status Code:", resp.status_code)
    print(resp.text[:1000])  # preview response
    data = resp.json()
    print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)
