from camera_controls import CameraController as Cam, CAMERA_SERIALS, OUTPUT_DIR
from projection_control import Projector as Proj
import argparse
import time
import pygame
import numpy as np
from tqdm import tqdm
import reconstruction_control as decoder


def run_photography_session(delay_ms=500, num_frames=None, run_name="test_run",
                            pattern="structured", random_count=300, fill_prob=0.5,
                            display_number=1, pattern_res_pxl=3,
                            decode=False, contrast_frac=0.2, color_dual=False,
                            decode_serial=None):
    projector = Proj(display_number=display_number, delay_ms=delay_ms, pattern_res_pxl=pattern_res_pxl)
    controller = Cam()

    if pattern == "raster":
        patterns = projector.generate_rasters()
    elif pattern == "hadamard":
        patterns = projector.generate_hadamard()
    elif pattern == "structured":
        patterns = projector.generate_structured_light()
    else:
        print(f"Unknown pattern type: {pattern}")
        return

    projector.save_pattern_matrix(run_name=run_name, pattern=pattern)

    num_frames = min(num_frames, len(patterns)) if num_frames is not None else len(patterns)
    pipelines = controller.dual_camera_setup(run_name)

    print(f"Starting capture of {num_frames} frames with interval {delay_ms}ms...")
    time.sleep(2)

    captured = 0
    try:
        for i in tqdm(range(num_frames), desc="Capturing", unit="frame"):
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            projector.proj_pattern(patterns[i])
            time.sleep(delay_ms / 1000.0)
            controller.capture_single_frame(pipelines, i)
            captured = i + 1

        print(f"Capture complete. {captured}/{num_frames} frames saved.")

    except Exception as e:
        print(f"Error during photography session: {e}")
        raise

    finally:
        controller.end_capture(pipelines)
        projector.quit()

    # ---- Decode stage (runs after capture + projector teardown) ----
    if decode:
        if captured < num_frames:
            print("Capture was interrupted; skipping decode (incomplete frame set).")
            return
        print("\n=== Decoding captures into dual photos ===")
        decoder.run_decode(
            run_name=run_name,
            pattern=pattern,
            serials=CAMERA_SERIALS,
            captures_root=OUTPUT_DIR,
            contrast_frac=contrast_frac,
            color=color_dual,
            only_serial=decode_serial,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the projector + dual-camera capture, then optionally decode."
    )
    parser.add_argument("--test",            action="store_true",  help="Run a short 3-frame smoke test.")
    parser.add_argument("--delay-ms",        type=float, default=200,        help="Delay between frames (ms).")
    parser.add_argument("--frames",          type=int,   default=None,       help="Number of frames. Defaults to all patterns.")
    parser.add_argument("--run-name",        type=str,   default="test_run", help="Output subfolder name.")
    parser.add_argument("--pattern",         choices=["raster", "hadamard", "structured"], default="structured")
    parser.add_argument("--display",         type=int,   default=1,          help="Monitor index for projector (0=primary).")
    parser.add_argument("--pattern_res_pxl", type=int,   default=4,          help="Projector pattern pixel resolution.")
    # decode options
    parser.add_argument("--decode",          action="store_true", help="Decode into dual photos after capture.")
    parser.add_argument("--contrast-frac",   type=float, default=0.2,        help="Min contrast (fraction of reference) to trust a pixel.")
    parser.add_argument("--color-dual",      action="store_true", help="Build color dual photo(s) from the reference frame.")
    parser.add_argument("--decode-serial",   default=None,        help="Decode only this one camera serial. Omit to decode all.")
    args = parser.parse_args()

    run_photography_session(
        delay_ms=args.delay_ms,
        num_frames=args.frames,
        run_name=args.run_name,
        pattern=args.pattern,
        display_number=args.display,
        pattern_res_pxl=args.pattern_res_pxl,
        decode=args.decode,
        contrast_frac=args.contrast_frac,
        color_dual=args.color_dual,
        decode_serial=args.decode_serial,
    )