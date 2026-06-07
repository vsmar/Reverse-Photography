import os
import json
import cv2
import pygame
import numpy as np
from tqdm import tqdm
from camera_controls import CameraController
from projection_control import Projector
import reconstruction_control as decoder


CAMERA_SERIALS = ["105322251697", "046322251346"]
OUTPUT_DIR = "captures"

# breaks up waiting into smaller chunks to allow intermediate processing
def _wait_ms(ms):
    end = pygame.time.get_ticks() + ms
    while pygame.time.get_ticks() < end:
        pygame.event.pump()
        pygame.time.wait(30)


def run_session(run_name="test_run", pattern="structured", inverse=True,
                delay_ms=500, settle_ms=100, flush_frames=2,
                display_number=1, pattern_res_pxl=4, decode=False):

    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Set up projector
    projector = Projector(display_number=display_number,
                          pattern_res_pxl=pattern_res_pxl,
                          inverse=inverse)

    if pattern == "raster":
        projector.generate_rasters()
    elif pattern == "hadamard":
        projector.generate_hadamard()
    else:
        projector.generate_structured_light()

    base_matrix = projector.pattern_matrix[0::2] if inverse else projector.pattern_matrix
    np.save(os.path.join(run_dir, "pattern_matrix.npy"), base_matrix)

    # Set up cameras
    cam = CameraController(CAMERA_SERIALS)
    cam.setup_cameras()

    raw_dirs = {}
    for serial in CAMERA_SERIALS:
        cam_dir = os.path.join(run_dir, serial)
        raw_dir = os.path.join(cam_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_dirs[serial] = raw_dir

    # Store metadata
    meta = {
        "pattern": pattern,
        "inverse": inverse,
        "grid_dimensions": projector.grid_dimensions,
        "n_cells": projector.n_cells,
        "pattern_res_pxl": projector.pattern_res_pxl,
        "settle_ms": settle_ms,
        "delay_ms": delay_ms,
        "flush_frames": flush_frames,
        "camera_serials": CAMERA_SERIALS,
    }

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Capture loop
    n = len(projector.patterns)
    try:
        with tqdm(total=n, desc=f"Capturing ({pattern})", unit="frame") as pbar:
            for i in projector.patterns:
                pygame.event.pump()

                projector.proj_pattern(i)
                _wait_ms(settle_ms)

                for serial in CAMERA_SERIALS:
                    frame = cam.capture_frame(serial, flush_frames=flush_frames)
                    if frame is not None:
                        raw_path = os.path.join(raw_dirs[serial], f"raw_{i:04d}.png")
                        cv2.imwrite(raw_path, frame)

                _wait_ms(delay_ms)
                pbar.update(1)

    finally:
        cam.stop_cameras()
        projector.quit()

    for serial in CAMERA_SERIALS:
        cam_dir = os.path.join(run_dir, serial)
        raw_dir = raw_dirs[serial]

        if inverse:
            for i in tqdm(range(projector.num_base_patterns),
                          desc=f"Processing {serial}", unit="frame"):

                orig_path = os.path.join(raw_dir, f"raw_{2*i:04d}.png")
                inv_path = os.path.join(raw_dir, f"raw_{2*i + 1:04d}.png")

                orig = cv2.imread(orig_path, cv2.IMREAD_COLOR)
                inv = cv2.imread(inv_path, cv2.IMREAD_COLOR)

                if orig is None or inv is None:
                    print(f"[WARN] Missing inverse pair {i} for {serial}")
                    continue

                if i == 0:
                    cv2.imwrite(os.path.join(cam_dir, "frame_0.png"), orig)
                else:
                    mask = (orig.astype(np.float32) > inv.astype(np.float32)).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(cam_dir, f"frame_{i}.png"), mask)

        else:
            for i in tqdm(range(n), desc=f"Processing {serial}", unit="frame"):
                raw_path = os.path.join(raw_dir, f"raw_{i:04d}.png")
                frame = cv2.imread(raw_path, cv2.IMREAD_COLOR)

                if frame is None:
                    print(f"[WARN] Missing raw frame {i} for {serial}")
                    continue

                cv2.imwrite(os.path.join(cam_dir, f"frame_{i}.png"), frame)

    if decode:
        print("Decoding...")
        decoder.run_decode(run_name, pattern, serials=CAMERA_SERIALS)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="test_run")
    parser.add_argument("--pattern", default="structured")
    parser.add_argument("--inverse", action="store_true", help="use inverse patterns")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--delay", type=int, default=200, help="Hold time after capture (ms)")
    parser.add_argument("--settle", type=int, default=200, help="Settle time before capture (ms)")
    parser.add_argument("--flush-frames", type=int, default=2, help="Frames to discard before capture")
    parser.add_argument("--res", type=int, default=4, help="Pattern resolution (px)")
    parser.add_argument("--display", type=int, default=1, help="Monitor index for projector")
    args = parser.parse_args()

    run_session(
        run_name=args.run_name,
        pattern=args.pattern,
        inverse=args.inverse,
        delay_ms=args.delay,
        settle_ms=args.settle,
        flush_frames=args.flush_frames,
        pattern_res_pxl=args.res,
        display_number=args.display,
        decode=args.decode,
    )