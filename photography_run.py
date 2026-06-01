from camera_controls import CameraController as Cam
from projection_control import Projector as Proj
import argparse
import time
import pygame


def run_photography_session(delay_ms=500, num_frames=None, run_name="test_run", 
                            pattern="raster", random_count=300, fill_prob=0.5):
    projector = Proj(delay_ms=delay_ms)
    controller = Cam()

    # Set up Projector Patterns
    #patterns = projector.generate_rasters()

    if pattern == "raster":
        patterns = projector.generate_rasters()
    elif pattern == "hadamard":
        patterns = projector.generate_rasters()
    elif pattern == "random":
        patterns = projector.generate_rasters()
    else:
        return 

    # save measurement matrix 
    projector.save_pattern_matrix(run_name=run_name, pattern=pattern)

    if num_frames is None:
        num_frames = len(patterns)
    else:
        num_frames = min(num_frames, len(patterns))

    pipelines = controller.dual_camera_setup(run_name)

    i = 0
    running = True

    print(f"Starting capture of {num_frames} frames with interval {delay_ms}ms...")
    time.sleep(2)

    try:
        while running and i < num_frames:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

                # Draw current pattern
            projector.proj_pattern(patterns[i])
            time.sleep(delay_ms / 1000.0)

            controller.capture_single_frame(pipelines, i)

            i += 1
    except Exception as e:
        print(f"Error during photography session: {e}")

    finally:
        controller.end_capture(pipelines)
        pygame.quit()


def test_photography_session(delay_ms=500):
    """Run a short smoke test of the projector and camera pipeline."""
    run_photography_session(delay_ms=delay_ms, num_frames=3, run_name="test_run")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the projector and dual-camera photography session.")
    parser.add_argument("--test", action="store_true", help="Run a short 3-frame smoke test.")
    parser.add_argument("--delay-ms", type=float, default=200, help="Delay between projector updates and captures, in milliseconds.")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to capture. Defaults to all projector patterns.")
    parser.add_argument("--run-name", type=str, default="test_run", help="Output subfolder name for this run.")
    parser.add_argument("--pattern", choices=["raster", "hadamard", "random"], default="raster")
    parser.add_argument("--random-count", type=int, default=300, help="Number of random patterns.")
    parser.add_argument("--fill-prob", type=float, default=0.5)
    args = parser.parse_args()

    if args.test:
        test_photography_session(delay_ms=args.delay_ms)
    else:
        run_photography_session(delay_ms=args.delay_ms, num_frames=args.frames, run_name=args.run_name,
                                pattern=args.pattern, random_count=args.random_count, fill_prob=args.fill_prob,
        )

