import requests

endpoints = [
    "http://127.0.0.1:8080/",
    "http://127.0.0.1:8080/simulation",
    "http://127.0.0.1:8080/api/scene/obj",
    "http://127.0.0.1:8080/api/scene/mtl",
    "http://127.0.0.1:8080/api/scene/texture",
    "http://127.0.0.1:8080/api/scene/default/buffer",
    "http://127.0.0.1:8080/api/proof_render/truck",
    "http://127.0.0.1:8080/api/telemetry"
]

print("=== VERIFYING SERVER ENDPOINTS ===")
for url in endpoints:
    try:
        r = requests.get(url, timeout=5)
        print(f"[{r.status_code}] {url} -> Content-Type: {r.headers.get('Content-Type')}, Length: {len(r.content):,} bytes")
    except Exception as e:
        print(f"[ERR] {url} -> {e}")
