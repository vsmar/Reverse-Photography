from camera_controls import CameraController as Cam
from projection_control import Projector as Proj
import argparse
import time
import pygame
import numpy as np
from tqdm import tqdm


def run_photography_session(delay_ms=500, num_frames=None, run_name="test_run",
                            pattern="structured", random_count=300, fill_prob=0.5,
                            display_number=1, pattern_res_pxl=3):
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
    pipelines  = controller.dual_camera_setup(run_name)

    print(f"Starting capture of {num_frames} frames with interval {delay_ms}ms...")
    time.sleep(2)

    try:
        for i in tqdm(range(num_frames), desc="Capturing", unit="frame"):
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            projector.proj_pattern(patterns[i])
            time.sleep(delay_ms / 1000.0)
            controller.capture_single_frame(pipelines, i)

        print(f"Capture complete. {i+1}/{num_frames} frames saved.")

    except Exception as e:
        print(f"Error during photography session: {e}")
        raise

    finally:
        controller.end_capture(pipelines)
        projector.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the projector and dual-camera photography session."
    )
    parser.add_argument("--test",         action="store_true",  help="Run a short 3-frame smoke test.")
    parser.add_argument("--delay-ms",     type=float, default=200,        help="Delay between frames (ms).")
    parser.add_argument("--frames",       type=int,   default=None,       help="Number of frames. Defaults to all patterns.")
    parser.add_argument("--run-name",     type=str,   default="test_run", help="Output subfolder name.")
    parser.add_argument("--pattern",      choices=["raster", "hadamard", "structured"], default="structured")
    parser.add_argument("--display",      type=int,   default=1,          help="Monitor index for projector (0=primary).")
    parser.add_argument("--pattern_res_pxl",  type=int,   default=4,          help="Projector pattern pixel resolution.")
    args = parser.parse_args()

    run_photography_session(
        delay_ms=args.delay_ms,
        num_frames=args.frames,
        run_name=args.run_name,
        pattern=args.pattern,
        display_number=args.display,
        pattern_res_pxl=args.pattern_res_pxl,
    )