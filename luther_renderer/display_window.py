"""
lutherICPU Native Desktop Simulation Display Player.

Pops up a high-performance desktop window displaying the generated 3D simulation:
- 360-Degree Continuous Turntable Simulation playback with smooth rotation
- Toggle between Hero 1080p, Side 1080p, 4K UHD, and 360° Turntable views
- Interactive keyboard controls:
  * [Space]: Pause / Resume 360° rotation
  * [Left / Right Arrow]: Manually rotate 360° view angle
  * [T]: Switch to 360° Turntable Simulation
  * [H]: Switch to 1080p Hero Angle
  * [S]: Switch to 1080p Side Profile
  * [Q] or [Esc]: Exit display window
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
from PIL import Image, ImageSequence

logger = logging.getLogger("lutherICPU.Display")


def load_gif_frames(gif_path: str) -> List[np.ndarray]:
    """Extracts all RGB frames from a turntable GIF as BGR OpenCV images."""
    if not os.path.exists(gif_path):
        return []
    gif = Image.open(gif_path)
    frames = []
    for frame in ImageSequence.Iterator(gif):
        rgb_frame = frame.convert("RGB")
        bgr = cv2.cvtColor(np.array(rgb_frame), cv2.COLOR_RGB2BGR)
        frames.append(bgr)
    return frames


def display_simulation_window(
    proof_dir_or_dict,
    scene_name: str = "Simulation",
    window_title: Optional[str] = None
):
    """
    Launches an interactive desktop window displaying the reconstructed 3D simulation.
    """
    if window_title is None:
        window_title = f"lutherICPU 3D Simulation Player - [{scene_name.upper()}]"

    hero_img = None
    side_img = None
    turntable_frames = []

    import glob
    if isinstance(proof_dir_or_dict, dict):
        hero_p = proof_dir_or_dict.get("hero_1080p")
        side_p = proof_dir_or_dict.get("side_1080p")
        gif_p = proof_dir_or_dict.get("turntable_gif")
    elif isinstance(proof_dir_or_dict, str):
        if os.path.isdir(proof_dir_or_dict):
            # Try exact scene name match first
            hero_p = os.path.join(proof_dir_or_dict, f"{scene_name}_simulation_1080p_hero.png")
            side_p = os.path.join(proof_dir_or_dict, f"{scene_name}_simulation_1080p_side.png")
            gif_p = os.path.join(proof_dir_or_dict, f"{scene_name}_360_simulation.gif")

            # Fallback to glob matches
            if not os.path.exists(hero_p):
                cands = glob.glob(os.path.join(proof_dir_or_dict, "*hero*.png")) or glob.glob(os.path.join(proof_dir_or_dict, "*.png"))
                hero_p = cands[0] if cands else None
            if not os.path.exists(side_p):
                cands = glob.glob(os.path.join(proof_dir_or_dict, "*side*.png"))
                side_p = cands[0] if cands else None
            if not os.path.exists(gif_p):
                cands = glob.glob(os.path.join(proof_dir_or_dict, "*360*.gif")) or glob.glob(os.path.join(proof_dir_or_dict, "*.gif"))
                gif_p = cands[0] if cands else None
        else:
            gif_p = proof_dir_or_dict
            hero_p = None
            side_p = None
    else:
        logger.warning("Invalid display path provided.")
        return

    if hero_p and os.path.exists(hero_p):
        hero_img = cv2.imread(hero_p)
    if side_p and os.path.exists(side_p):
        side_img = cv2.imread(side_p)
    if gif_p and os.path.exists(gif_p):
        turntable_frames = load_gif_frames(gif_p)

    if not turntable_frames and hero_img is None and side_img is None:
        logger.warning("No render assets found to display.")
        return

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1280, 720)

    current_mode = "TURNTABLE" if turntable_frames else ("HERO" if hero_img is not None else "SIDE")
    state = {
        "frame_idx": 0,
        "is_paused": True,  # PAUSED BY DEFAULT: Waits for user control
        "current_mode": current_mode,
        "is_dragging": False,
        "last_x": 0,
        "drag_accum": 0.0
    }
    total_frames = max(1, len(turntable_frames))
    delay_ms = 40  # ~25 fps

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["is_dragging"] = True
            state["last_x"] = x
            state["is_paused"] = True  # Pause on mouse interaction
        elif event == cv2.EVENT_MOUSEMOVE and state["is_dragging"]:
            dx = x - state["last_x"]
            state["last_x"] = x
            if total_frames > 0:
                # 1 frame per 16 pixels of horizontal mouse drag
                state["drag_accum"] += dx / 16.0
                step = int(state["drag_accum"])
                if step != 0:
                    state["frame_idx"] = (state["frame_idx"] + step) % total_frames
                    state["drag_accum"] -= step
                    state["current_mode"] = "TURNTABLE"
        elif event == cv2.EVENT_LBUTTONUP:
            state["is_dragging"] = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Scroll wheel to rotate angle
            if flags > 0:
                state["frame_idx"] = (state["frame_idx"] - 1) % total_frames
            else:
                state["frame_idx"] = (state["frame_idx"] + 1) % total_frames
            state["current_mode"] = "TURNTABLE"
            state["is_paused"] = True

    cv2.setMouseCallback(window_title, on_mouse)

    logger.info(f"Displaying 3D simulation for '{scene_name}'. Use mouse drag / arrows to orbit. Press [Q] to close.")

    while True:
        mode = state["current_mode"]
        f_idx = state["frame_idx"]
        paused = state["is_paused"]

        if mode == "TURNTABLE" and turntable_frames:
            canvas = turntable_frames[f_idx].copy()
            angle_deg = int((f_idx / total_frames) * 360)
            status_text = f"360 Orbit | Angle: {angle_deg} deg | {'[PAUSED - Drag Mouse to Rotate]' if paused else '[PLAYING]'}"
            if not paused:
                state["frame_idx"] = (f_idx + 1) % total_frames
        elif mode == "HERO" and hero_img is not None:
            canvas = hero_img.copy()
            status_text = "DSLR Hero Angle (Photo 001) Simulation"
        elif mode == "SIDE" and side_img is not None:
            canvas = side_img.copy()
            status_text = "1080p Side Profile Angle Simulation"
        else:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            status_text = "No view available"

        # Overlay HUD banner cleanly on a fresh copy
        h, w = canvas.shape[:2]
        hud_h = 44
        cv2.rectangle(canvas, (0, h - hud_h), (w, h), (10, 14, 20), -1)
        cv2.line(canvas, (0, h - hud_h), (w, h - hud_h), (0, 240, 255), 1)

        cv2.putText(canvas, f"lutherICPU: {scene_name.upper()} | {status_text}", (16, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[Mouse Drag / Scroll]: Orbit 360  [Space]: Play/Pause  [T]: 360  [H]: Hero  [S]: Side  [Q]: Exit",
                    (w - 750, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.imshow(window_title, canvas)
        key = cv2.waitKeyEx(delay_ms)

        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32:  # Spacebar
            state["is_paused"] = not state["is_paused"]
        elif key in (81, 2424832, 0x250000):  # Left Arrow
            state["frame_idx"] = (state["frame_idx"] - 1) % total_frames
            state["current_mode"] = "TURNTABLE"
            state["is_paused"] = True
        elif key in (83, 2555904, 0x270000):  # Right Arrow
            state["frame_idx"] = (state["frame_idx"] + 1) % total_frames
            state["current_mode"] = "TURNTABLE"
            state["is_paused"] = True
        elif key in (ord('t'), ord('T')):
            if turntable_frames:
                state["current_mode"] = "TURNTABLE"
        elif key in (ord('h'), ord('H')):
            if hero_img is not None:
                state["current_mode"] = "HERO"
        elif key in (ord('s'), ord('S')):
            if side_img is not None:
                state["current_mode"] = "SIDE"

    cv2.destroyWindow(window_title)


if __name__ == "__main__":
    test_proof_dir = os.path.join(os.path.dirname(__file__), "..", "output", "truck_test", "proof_renders")
    display_simulation_window(test_proof_dir, scene_name="truck")
