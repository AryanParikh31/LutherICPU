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
    """Launches the 60 FPS hardware-accelerated interactive 3D SIBR simulation viewport."""
    from luther_web.server import app
    import uvicorn
    import webbrowser

    def run_srv():
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        server.run()

    thread = threading.Thread(target=run_srv, daemon=True)
    thread.start()
    time.sleep(1.0)

    url = f"http://127.0.0.1:{port}/simulation"
    print("\n" + "=" * 80)
    print(" [*] LAUNCHING LUTHERICPU INTERACTIVE 3D SIBR SIMULATION PLATFORM (60 FPS)")
    print(f" Viewport Active at: {url}")
    print(" Mouse Controls: Left-Click + Drag to Orbit 360 deg | Right-Click to Pan | Scroll to Zoom")
    print("=" * 80 + "\n")

    try:
        from PySide6.QtWidgets import QApplication, QMainWindow
        from PySide6.QtCore import QUrl
        from PySide6.QtWebEngineWidgets import QWebEngineView

        app_qt = QApplication.instance() or QApplication(sys.argv)
        win = QMainWindow()
        win.setWindowTitle("lutherICPU 3D SIBR Simulation Viewer (60 FPS Native Desktop)")
        win.resize(1400, 900)
        view = QWebEngineView()
        view.setUrl(QUrl(url))
        win.setCentralWidget(view)
        win.show()
        app_qt.exec()
    except Exception:
        # Fallback to system browser
        webbrowser.open(url)
        print("[*] Opened interactive 3D viewport in default browser.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[*] Exiting viewer.")


from luther_renderer.sibr_desktop_app import launch_sibr_desktop
from luther_renderer.display_window import display_simulation_window


def main():
    parser = argparse.ArgumentParser(description="lutherICPU Standalone Desktop SIBR Simulation Player")
    parser.add_argument("--scene", "-s", type=str, default="truck", help="Scene name to display (default: truck)")
    parser.add_argument("--output", "-o", type=str, default=os.path.join(PROJECT_ROOT, "output"), help="Output root directory")
    parser.add_argument("--proof-dir", "-p", type=str, default=None, help="Explicit path to proof_renders directory")
    parser.add_argument("--opencv", action="store_true", help="Launch lightweight OpenCV turntable window instead of SIBR Desktop GUI")
    args = parser.parse_args()

    proof_dir = args.proof_dir
    if not proof_dir:
        candidates = [
            os.path.join(args.output, "truck_photos_final", "proof_renders"),
            os.path.join(args.output, "truck_run", "proof_renders"),
            os.path.join(args.output, f"{args.scene}_40k_sim", "proof_renders"),
            os.path.join(args.output, f"{args.scene}_roof_verified", "proof_renders"),
            os.path.join(args.output, f"{args.scene}_simulation", "proof_renders"),
            os.path.join(args.output, "proof_renders"),
        ]
        for c in candidates:
            if os.path.exists(c):
                proof_dir = c
                break

    if args.opencv:
        display_simulation_window(proof_dir or os.path.join(args.output, "proof_renders"), scene_name=args.scene)
        return

    # Master Default: Launch Pure Native Desktop SIBR 3D Simulation Platform (Zero Browser)
    print("\n" + "=" * 80)
    print(" [*] LAUNCHING LUTHERICPU STANDALONE DESKTOP SIBR 3D SIMULATION PLATFORM")
    print(" [*] Zero Browser Dependencies | Full Interactive Mouse Controls")
    print("=" * 80 + "\n")
    launch_sibr_desktop(scene_name=args.scene, proof_dir=proof_dir)


if __name__ == "__main__":
    main()
