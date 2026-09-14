"""
Standalone Desktop Forensic SIBR Viewer (Agent Group 5)
Interactive 60 FPS Desktop Viewer for lutherICPU 3D scenes with metric caliper,
camera frustums, ground-truth comparison slider, and courtroom dossier viewer.
"""

import os
import sys
import threading
import time
import uvicorn
from luther_web.server import app

def start_server_in_background():
    """Starts the FastAPI WebGL2 server on a background thread."""
    config = uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server

def launch_pyside_viewer(url: str):
    """Launches PySide6 Desktop Window with Edge/Chromium Web Engine."""
    from PySide6.QtWidgets import QApplication, QMainWindow
    from PySide6.QtCore import QUrl
    
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
        has_webengine = True
    except ImportError:
        has_webengine = False
        
    app_qt = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("lutherICPU Forensic 3D Simulation Platform (60 FPS Native Desktop)")
    window.resize(1400, 900)
    
    if has_webengine:
        view = QWebEngineView()
        view.setUrl(QUrl(url))
        window.setCentralWidget(view)
    else:
        # Fallback to system browser launcher if QtWebEngine is not bundled
        import webbrowser
        webbrowser.open(url)
        print(f"[*] Opening browser to: {url}")
        return
        
    window.show()
    sys.exit(app_qt.exec())

def main():
    print("=" * 80)
    print(" 🖥️ LAUNCHING LUTHERICPU DESKTOP FORENSIC VIEWER (AGENT GROUP 5)")
    print(" Native WebGL2 Hardware Triangle Rasterizer (60 FPS)")
    print("=" * 80)
    
    # 1. Start Server
    start_server_in_background()
    time.sleep(1.0)
    url = "http://127.0.0.1:8080/simulation"
    
    print(f"[*] 3D SIBR Simulation Engine active at: {url}")

    
    # 2. Try PySide6 or browser
    try:
        launch_pyside_viewer(url)
    except Exception as e:
        import webbrowser
        print(f"[!] PySide Desktop Window notice ({e}), opening default browser: {url}")
        webbrowser.open(url)

if __name__ == "__main__":
    main()
