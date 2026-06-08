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
BIT_MARGIN = 3
DELTA_PNG_ZERO = 32768

###############################################
# HELPER FUNCTIONS                            #
###############################################
def _wait_ms(ms):
    end = pygame.time.get_ticks() + ms
    while pygame.time.get_ticks() < end:
        pygame.event.pump()
        pygame.time.wait(30)


def _read_raw(raw_dir, idx):
    return cv2.imread(os.path.join(raw_dir, f"raw_{idx:04d}.png"), cv2.IMREAD_COLOR)


def _save_complement_products(raw_dir, out_dir, idx):
    orig = _read_raw(raw_dir, 2 * idx)
    comp = _read_raw(raw_dir, 2 * idx + 1)
    if orig is None or comp is None:
        return False

    if idx == 0:
        cv2.imwrite(os.path.join(out_dir, "frame_0.png"), orig)
        return True

    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.int16)
    comp_gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY).astype(np.int16)
    delta = orig_gray - comp_gray

    np.save(os.path.join(out_dir, f"delta_{idx:04d}.npy"), delta)
    cv2.imwrite(os.path.join(out_dir, f"delta_{idx:04d}.png"),
                (delta.astype(np.int32) + DELTA_PNG_ZERO).astype(np.uint16))

    mask = (delta > BIT_MARGIN).astype(np.uint8) * 255
    conf = np.clip(np.abs(delta).astype(np.float32) * 255.0 / 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, f"frame_{idx}.png"), mask)
    cv2.imwrite(os.path.join(out_dir, f"confidence_{idx:04d}.png"), conf)
    return True


def post_process_run(run_dir, serials, complement, n_raw, n_base):
    for serial in serials:
        cam_dir = os.path.join(run_dir, serial)
        raw_dir = os.path.join(cam_dir, "raw")
        proc_dir = os.path.join(cam_dir, "processed")
        os.makedirs(proc_dir, exist_ok=True)

        if complement:
            for i in tqdm(range(n_base), desc=f"Processing {serial}", unit="frame"):
                ok = _save_complement_products(raw_dir, proc_dir, i)
                if not ok:
                    print(f"[WARN] Missing complement pair {i} for {serial}")
        else:
            for i in tqdm(range(n_raw), desc=f"Processing {serial}", unit="frame"):
                frame = _read_raw(raw_dir, i)
                if frame is None:
                    print(f"[WARN] Missing raw frame {i} for {serial}")
                    continue
                cv2.imwrite(os.path.join(proc_dir, f"frame_{i}.png"), frame)


###############################################
# Run Code                                    #
###############################################
def run_session(run_name="test_run", pattern="structured", complement=True,
                delay_ms=500, settle_ms=100, flush_frames=2,
                display_number=1, pattern_res_pxl=4, decode=False, color=True):

    run_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Set up projector
    projector = Projector(display_number=display_number,
                          pattern_res_pxl=pattern_res_pxl,
                          complement=complement)

    if pattern == "raster":
        projector.generate_rasters()
    elif pattern == "hadamard":
        projector.generate_hadamard()
    else:
        projector.generate_structured_light()

    base_matrix = projector.pattern_matrix[0::2] if complement else projector.pattern_matrix
    np.save(os.path.join(run_dir, "pattern_matrix.npy"), base_matrix)

    # Set up cameras
    cam = CameraController(CAMERA_SERIALS)
    cam.setup_cameras()

    raw_dirs = {}
    for serial in CAMERA_SERIALS:
        raw_dir = os.path.join(run_dir, serial, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_dirs[serial] = raw_dir

    # Save metadata for later decoding reference
    meta = {
        "pattern": pattern,
        "complement": complement,
        "pattern_res_pxl": projector.pattern_res_pxl,
        "camera_serials": CAMERA_SERIALS,
        "capture": {
            "settle_ms": settle_ms,
            "delay_ms": delay_ms,
            "flush_frames": flush_frames,
        },
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
                        cv2.imwrite(os.path.join(raw_dirs[serial], f"raw_{i:04d}.png"), frame)

                _wait_ms(delay_ms)
                pbar.update(1)
    finally:
        cam.stop_cameras()
        projector.quit()

    post_process_run(run_dir, CAMERA_SERIALS, complement, n,
                     projector.num_base_patterns)
    
    if decode:
        print("Decoding...")
        decoder.run_decode(run_name=run_name, captures_root=OUTPUT_DIR, color=color)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="test_run")
    parser.add_argument("--pattern", default="structured")
    parser.add_argument("--complement", action="store_true", help="use complement patterns")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--delay", type=int, default=200, help="Hold time after capture (ms)")
    parser.add_argument("--settle", type=int, default=200, help="Settle time before capture (ms)")
    parser.add_argument("--flush-frames", type=int, default=2, help="Frames to discard before capture")
    parser.add_argument("--res", type=int, default=4, help="Pattern resolution (px)")
    parser.add_argument("--display", type=int, default=1, help="Monitor index for projector")
    parser.add_argument("--grayscale", action="store_false", help="Decode in grayscale instead of color")
    args = parser.parse_args()

    run_session(
        run_name=args.run_name,
        pattern=args.pattern,
        complement=args.complement,
        delay_ms=args.delay,
        settle_ms=args.settle,
        flush_frames=args.flush_frames,
        pattern_res_pxl=args.res,
        display_number=args.display,
        decode=args.decode,
        color=args.grayscale,
    )
