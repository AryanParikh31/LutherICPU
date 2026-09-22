"""
lutherICPU Standalone Desktop 3D Simulation Viewer & SIBR Player.

Usage:
  python luther_display.py
  python luther_display.py --3d
  python luther_display.py --turntable
  python luther_display.py --scene truck
"""

import os
import sys
import argparse
import time
import threading

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from luther_renderer.display_window import display_simulation_window


def launch_interactive_3d_viewport(port: int = 8080):
    """Launches the 60 FPS hardware-accelerated interactive 3D SIBR simulation viewport in a native desktop window."""
    import socket
    import uvicorn
    import webbrowser
    from luther_web.server import app

    # Robust port binding check
    def is_port_in_use(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', p)) == 0

    active_port = port
    if is_port_in_use(active_port):
        # Check if it's already responding to our telemetry endpoint
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://127.0.0.1:{active_port}/api/telemetry", timeout=0.8)
            if req.status == 200:
                print(f"[*] Connected to existing lutherICPU simulation server on port {active_port}.")
            else:
                for p_cand in range(8081, 8095):
                    if not is_port_in_use(p_cand):
                        active_port = p_cand
                        break
        except Exception:
            for p_cand in range(8081, 8095):
                if not is_port_in_use(p_cand):
                    active_port = p_cand
                    break

    # Start server if not already responding
    try:
        import urllib.request
        urllib.request.urlopen(f"http://127.0.0.1:{active_port}/api/telemetry", timeout=0.5)
    except Exception:
        def run_srv():
            config = uvicorn.Config(app, host="127.0.0.1", port=active_port, log_level="warning")
            server = uvicorn.Server(config)
            server.run()

        thread = threading.Thread(target=run_srv, daemon=True)
        thread.start()
        time.sleep(1.2)

    url = f"http://127.0.0.1:{active_port}/simulation"
    print("\n" + "=" * 80)
    print(" [*] LAUNCHING LUTHERICPU 3D GAUSSIAN SPLATTING & TEXTURED MESH SIMULATION PLATFORM (60 FPS)")
    print(f" Viewport Active at: {url}")
    print(" Controls: Left-Click + Drag: 360 Orbit | Mouse Wheel: Zoom In/Out | Right-Click: Pan | WASD: Fly")
    print("=" * 80 + "\n")

    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--enable-webgl --ignore-gpu-blocklist --enable-gpu-rasterization --no-sandbox"

    try:
        from PySide6.QtWidgets import QApplication, QMainWindow
        from PySide6.QtCore import QUrl
        from PySide6.QtWebEngineWidgets import QWebEngineView

        app_qt = QApplication.instance() or QApplication(sys.argv)
        win = QMainWindow()
        win.setWindowTitle("lutherICPU — 3D Gaussian Splatting & Textured Mesh Simulation Platform (60 FPS)")
        win.resize(1400, 900)
        view = QWebEngineView()
        view.setUrl(QUrl(url))
        win.setCentralWidget(view)
        win.show()
        app_qt.exec()
    except Exception as e:
        print(f"[*] Opening browser interactive 3D simulation viewport: {url} (PySide notice: {e})")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[*] Exiting viewer.")


def main():
    parser = argparse.ArgumentParser(description="lutherICPU Standalone Desktop SIBR Simulation Player")
    parser.add_argument("--scene", "-s", type=str, default=None, help="Scene name to display (auto-detected if omitted)")
    parser.add_argument("--output", "-o", type=str, default=os.path.join(PROJECT_ROOT, "output"), help="Output root directory")
    parser.add_argument("--proof-dir", "-p", type=str, default=None, help="Explicit path to proof_renders directory")
    parser.add_argument("--software", "--cv", action="store_true", help="Launch pure CPU software projection window fallback")
    parser.add_argument("--port", type=int, default=8080, help="Port for 3D simulation server")
    args = parser.parse_args()

    scene_name = args.scene
    proof_dir = args.proof_dir

    # Auto-detect latest scene from latest_simulation.json
    latest_json = os.path.join(args.output, "latest_simulation.json")
    if os.path.exists(latest_json):
        try:
            import json
            with open(latest_json, "r") as f:
                lat_data = json.load(f)
                if not scene_name:
                    scene_name = lat_data.get("scene_name")
                if not proof_dir and lat_data.get("output_dir"):
                    if os.path.exists(lat_data["output_dir"]):
                        proof_dir = lat_data["output_dir"]
        except Exception:
            pass

    if not scene_name:
        scene_name = "simulation"

    if args.software:
        # Software fallback mode
        if not proof_dir:
            candidates = [
                os.path.join(args.output, f"{scene_name}_3dgs"),
                os.path.join(args.output, f"{scene_name}_sim"),
                os.path.join(args.output, f"{scene_name}_simulation", "proof_renders"),
                os.path.join(args.output, "proof_renders"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    proof_dir = c
                    break
        print("\n" + "=" * 80)
        print(" [*] LAUNCHING LUTHERICPU PURE CPU SOFTWARE 3D PROJECTION WINDOW")
        print("=" * 80 + "\n")
        display_simulation_window(proof_dir or os.path.join(args.output, "proof_renders"), scene_name=scene_name)
    else:
        # Master Default: Launch 60 FPS 3D Gaussian Splatting & Mesh Simulation Viewport
        launch_interactive_3d_viewport(port=args.port)


if __name__ == "__main__":
    main()
