import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import argparse
from projection_control import Projector

OUTPUT_DIR = "captures"
CAMERA_1_SERIAL = "105322251697"
CAMERA_2_SERIAL = "046322251346"
CAPTURE_INTERVAL_S = 0.2


def start_pipeline(serial: str):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 5)

    pipeline.start(config)
    return pipeline


def capture_rgb(num_frames: int, projector_update=None, interval_s: float = CAPTURE_INTERVAL_S):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipelines = {
        CAMERA_1_SERIAL: start_pipeline(CAMERA_1_SERIAL),
        CAMERA_2_SERIAL: start_pipeline(CAMERA_2_SERIAL),
    }

    time.sleep(2)  # give the cameras time to warm up before requesting frames

    try:
        for i in range(num_frames):
            # if projector_update is not None:
            #     projector_update(i)

            time.sleep(interval_s)

            for serial, pipeline in pipelines.items():
                frames = pipeline.wait_for_frames(timeout_ms=1500)
                color_frame = frames.get_color_frame()

                if not color_frame:
                    print(f"Frame {i} ({serial}): no color frame received, skipping.")
                    continue

                image = np.asanyarray(color_frame.get_data())
                camera_dir = os.path.join(OUTPUT_DIR, serial)
                os.makedirs(camera_dir, exist_ok=True)
                filename = os.path.join(camera_dir, f"{i}.png")
                cv2.imwrite(filename, image)
                print(f"Saved: {filename}")

    finally:
        for pipeline in pipelines.values():
            pipeline.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture RGB stills from two Intel RealSense D455 cameras.")
    parser.add_argument("num_frames", type=int, help="Number of frames to capture.")
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=CAPTURE_INTERVAL_S * 1000,
        help="Delay between projector update and frame capture, in milliseconds.",
    )
    args = parser.parse_args()

    capture_rgb(args.num_frames, interval_s=args.interval_ms / 1000.0)