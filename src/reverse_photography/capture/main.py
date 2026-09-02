"""Run the full capture chain: photograph -> decode -> fuse.

Equivalent to running photography_run.py with --decode, then fuse.py on the
same run -- the two terminal calls we used to run by hand, one after another.

    python src/reverse_photography/capture/main.py --run-name my_scan
"""
import argparse

from fuse import fuse
from photography_run import OUTPUT_DIR, run_session


def main(run_name="test_run", pattern="structured", complement=True,
         delay_ms=500, settle_ms=100, flush_frames=2, display_number=1,
         pattern_res_pxl=4, color=True, exposure=200,
         captures_root=OUTPUT_DIR):
    run_session(
        run_name=run_name,
        pattern=pattern,
        complement=complement,
        delay_ms=delay_ms,
        settle_ms=settle_ms,
        flush_frames=flush_frames,
        display_number=display_number,
        pattern_res_pxl=pattern_res_pxl,
        decode=True,
        color=color,
        exposure=exposure,
    )
    return fuse(run_name=run_name, captures_root=captures_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="test_run")
    parser.add_argument("--pattern", default="structured")
    parser.add_argument("--complement", action="store_true", help="use complement patterns")
    parser.add_argument("--delay", type=int, default=500, help="Hold time after capture (ms)")
    parser.add_argument("--settle", type=int, default=100, help="Settle time before capture (ms)")
    parser.add_argument("--flush-frames", type=int, default=2, help="Frames to discard before capture")
    parser.add_argument("--res", type=int, default=4, help="Pattern resolution (px)")
    parser.add_argument("--display", type=int, default=1, help="Monitor index for projector")
    parser.add_argument("--grayscale", action="store_false", help="Decode in grayscale instead of color")
    parser.add_argument("--exposure", type=int, default=200, help="Camera exposure time (ms)")
    args = parser.parse_args()

    main(
        run_name=args.run_name,
        pattern=args.pattern,
        complement=args.complement,
        delay_ms=args.delay,
        settle_ms=args.settle,
        flush_frames=args.flush_frames,
        display_number=args.display,
        pattern_res_pxl=args.res,
        color=args.grayscale,
        exposure=args.exposure,
    )
