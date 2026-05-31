import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import argparse

OUTPUT_DIR = "captures"
CAMERA_SERIALS = ["105322251697", "046322251346"]
CAPTURE_INTERVAL_S = 0.2

class CameraController:
    @staticmethod
    def start_pipeline(serial: str):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 480, 270, rs.format.bgr8, 5)

        pipeline.start(config)
        return pipeline
    
    def dual_camera_setup(self, run_name="default_run"):
        pipelines = {}

        for serial in CAMERA_SERIALS:
            cam_dir = os.path.join(OUTPUT_DIR, run_name, serial)
            os.makedirs(cam_dir, exist_ok=True)
            pipelines[cam_dir] = self.start_pipeline(serial)

        print(f"Cameras initialized: {list(pipelines.keys())}")

        time.sleep(1) # give the cameras time to warm up before requesting frames

        return pipelines
    
    def capture_single_frame(self, pipelines, frame_id):
        for cam_dir, pipeline in pipelines.items():
            frames = pipeline.wait_for_frames(timeout_ms=1500)
            color_frame = frames.get_color_frame()

            if not color_frame:
                print(f"No frame received from {cam_dir}.")
                continue

            image = np.asanyarray(color_frame.get_data())
            filename = os.path.join(cam_dir, f"frame_{frame_id}.png")
            cv2.imwrite(filename, image)
            print(f"Saved: {filename}")
    
    def end_capture(self, pipelines):
        for pipeline in pipelines.values():
            pipeline.stop()
     
    def test_capture(self, num_frames=10, interval_s: float = CAPTURE_INTERVAL_S):
        pipelines = self.dual_camera_setup("test_run")

        print(f"Starting capture of {num_frames} frames with interval {interval_s}s...")

        try:
            for i in range(num_frames):
                # Delay time for projection
                time.sleep(interval_s)

                self.capture_single_frame(pipelines, i)
                
        finally:
            self.end_capture(pipelines)
            

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

    controller = CameraController()
    controller.test_capture(args.num_frames, interval_s=args.interval_ms / 1000.0)