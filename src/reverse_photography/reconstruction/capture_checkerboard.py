"""Capture synchronized checkerboard image pairs for stereo calibration.

Run this ON THE MACHINE WITH THE TWO REALSENSE CAMERAS ATTACHED. It opens a
live preview of both cameras side by side, detects the checkerboard in each
view in real time, and saves a matched pair every time you press SPACE.

The projector is NOT involved -- point a PHYSICAL printed checkerboard at the
two cameras. (Projecting a checkerboard would ruin corner detection.) Light the
board well; the projector, if on, should show plain white only.

Output (consumed directly by calibration.py / stereo_calibrate):
    <checkerboard_dir>/<serial1>/pair_000.png, pair_001.png, ...
    <checkerboard_dir>/<serial2>/pair_000.png, pair_001.png, ...
The i-th file in each folder is the same instant, which is what stereo
calibration requires.

Controls
    SPACE : save the current pair (only allowed when the board is found in BOTH)
    u     : undo / delete the last saved pair
    q/ESC : finish and quit

Tips for a good calibration (aim for 15-25 pairs):
    - The board must be FULLY visible in BOTH views at once.
    - Vary it: near/far, left/right/up/down, and especially TILT it (don't keep
      it flat-on -- tilted poses are what pin down the geometry).
    - Cover the whole image area across your shots, including the corners.
    - Hold still when you press SPACE; motion blur ruins corner accuracy.
    - Make sure config.checkerboard_cols/rows/square_mm match YOUR board.
      Counts are INNER corners: a board with 10x7 squares has 9x6 inner corners.
"""
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

from config import CONFIG


def start_pipeline(serial, width, height):
    """Open one RealSense color stream. Auto-exposure ON so the board is clear
    (unlike the capture run, calibration cares about visibility, not frame-to-
    frame consistency)."""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    for sensor in profile.get_device().query_sensors():
        if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
            if sensor.supports(rs.option.frames_queue_size):
                sensor.set_option(rs.option.frames_queue_size, 1)
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, 1)
    return pipe


def find_board(gray, pattern):
    """Fast checkerboard test. Returns (found, corners)."""
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    return cv2.findChessboardCorners(gray, pattern, flags)


def main(cfg=CONFIG):
    s1, s2 = cfg.camera_serials
    pattern = (cfg.checkerboard_cols, cfg.checkerboard_rows)
    W, H = cfg.image_width, cfg.image_height

    out1 = os.path.join(cfg.checkerboard_dir, s1)
    out2 = os.path.join(cfg.checkerboard_dir, s2)
    os.makedirs(out1, exist_ok=True)
    os.makedirs(out2, exist_ok=True)

    print(f"Opening cameras {s1} and {s2} at {W}x{H} ...")
    p1 = start_pipeline(s1, W, H)
    p2 = start_pipeline(s2, W, H)

    # Warm up so auto-exposure settles before the first preview.
    for _ in range(20):
        p1.wait_for_frames(timeout_ms=2000)
        p2.wait_for_frames(timeout_ms=2000)

    print(f"Looking for a {pattern[0]}x{pattern[1]} inner-corner board "
          f"({cfg.checkerboard_square_mm} mm squares).")
    print("SPACE=save pair   u=undo last   q/ESC=quit")

    saved = 0
    try:
        while True:
            f1 = p1.wait_for_frames(timeout_ms=2000).get_color_frame()
            f2 = p2.wait_for_frames(timeout_ms=2000).get_color_frame()
            if not f1 or not f2:
                continue
            img1 = np.asanyarray(f1.get_data())
            img2 = np.asanyarray(f2.get_data())
            g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            ok1, c1 = find_board(g1, pattern)
            ok2, c2 = find_board(g2, pattern)

            # Build the preview (draw corners, status, count).
            v1, v2 = img1.copy(), img2.copy()
            cv2.drawChessboardCorners(v1, pattern, c1, ok1)
            cv2.drawChessboardCorners(v2, pattern, c2, ok2)
            both = ok1 and ok2
            for v, ok, name in ((v1, ok1, s1), (v2, ok2, s2)):
                color = (0, 200, 0) if ok else (0, 0, 255)
                cv2.putText(v, f"{name[-4:]}: {'FOUND' if ok else 'no board'}",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            preview = cv2.hconcat([v1, v2])
            preview = cv2.resize(preview, (preview.shape[1] // 2,
                                           preview.shape[0] // 2))
            banner = (0, 200, 0) if both else (0, 0, 255)
            cv2.putText(preview,
                        f"pairs saved: {saved}   "
                        f"{'SPACE to save' if both else 'need board in BOTH'}",
                        (15, preview.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner, 2)
            cv2.imshow("checkerboard capture (SPACE=save  u=undo  q=quit)",
                       preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):                 # q or ESC
                break
            if key == ord('u') and saved > 0:         # undo last
                saved -= 1
                for d in (out1, out2):
                    fp = os.path.join(d, f"pair_{saved:03d}.png")
                    if os.path.exists(fp):
                        os.remove(fp)
                print(f"  undid pair_{saved:03d}; {saved} remain")
            if key == ord(' '):
                if not both:
                    print("  not saved: board must be found in BOTH views")
                    continue
                cv2.imwrite(os.path.join(out1, f"pair_{saved:03d}.png"), img1)
                cv2.imwrite(os.path.join(out2, f"pair_{saved:03d}.png"), img2)
                saved += 1
                print(f"  saved pair_{saved-1:03d}  (total {saved})")
    finally:
        p1.stop()
        p2.stop()
        cv2.destroyAllWindows()

    print(f"\nDone. {saved} pairs in {cfg.checkerboard_dir}/<serial>/")
    if saved < 10:
        print("WARNING: fewer than 10 pairs -- calibration may be unreliable. "
              "15-25 well-varied poses is the sweet spot.")
    print("Next:  python calibration.py    (computes R, T -> stereo_calib.npz)")


if __name__ == "__main__":
    sys.exit(main())
