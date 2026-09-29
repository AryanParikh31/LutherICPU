import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from luther_web.server import app

client = TestClient(app)

endpoints = [
    "/simulation",
    "/api/scene/info",
    "/api/cameras",
    "/api/scene/splat",
    "/api/scene/binary",
    "/api/scene/texture",
    "/api/telemetry",
    "/api/scenes"
]

print("=== RUNNING FASTAPI TESTCLIENT VERIFICATION ===")
all_passed = True
for ep in endpoints:
    try:
        r = client.get(ep)
        status = r.status_code
        c_type = r.headers.get("content-type", "")
        length = len(r.content)
        print(f"[{status}] {ep} -> type: {c_type}, size: {length:,} bytes")
        if status != 200:
            all_passed = False
    except Exception as e:
        print(f"[FAIL] {ep} -> {e}")
        all_passed = False

if all_passed:
    print("\n>>> ALL ENDPOINTS RETURNED 200 OK SUCCESS! <<<")
else:
    print("\n>>> SOME ENDPOINTS FAILED <<<")
