import pyrealsense2 as rs
import numpy as np
import cv2
import threading


class CameraController:
    def __init__(self, serials):
        self.serials = serials
        self.pipelines = {}

    def setup_cameras(self, gain=0, exposure=200):
        self.gain = None
        self.exposure = None

        for serial in self.serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
            profile = pipeline.start(config)

            device = profile.get_device()
            for sensor in device.query_sensors():
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    if sensor.supports(rs.option.frames_queue_size):
                        sensor.set_option(rs.option.frames_queue_size, 1)
                    if sensor.supports(rs.option.enable_auto_exposure):     # disable auto-exposure
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                        if sensor.supports(rs.option.exposure):
                            sensor.set_option(rs.option.exposure, exposure)
                            self.exposure = exposure
                    if sensor.supports(rs.option.gain):
                        sensor.set_option(rs.option.gain, gain)
                        self.gain = gain

            self.pipelines[serial] = pipeline

        print("Warming up cameras...")
        for _ in range(30):
            for pipeline in self.pipelines.values():
                pipeline.wait_for_frames()

    def capture_frame(self, serial, flush_frames=2, timeout_ms=5000):
        pipeline = self.pipelines[serial]

        # Drop frames that may have started exposure before/during the projector transition.
        for _ in range(flush_frames):
            pipeline.wait_for_frames(timeout_ms=timeout_ms)

        frames = pipeline.wait_for_frames(timeout_ms=timeout_ms)
        color = frames.get_color_frame()
        if not color:
            return None

        return np.asanyarray(color.get_data()).copy()

    def stop_cameras(self):
        for pipeline in self.pipelines.values():
            pipeline.stop()


if __name__ == '__main__':
    import argparse
    import time
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to capture.")
    parser.add_argument("--delay",  type=int, default=500, help="Interval between captures (ms).")
    parser.add_argument("--serials", nargs="+", default=["105322251697", "046322251346"])
    args = parser.parse_args()

    cam = CameraController(args.serials)
    cam.setup_cameras()

    for serial in cam.serials:
        os.makedirs(rf"camera_test\{serial}", exist_ok=True)

    for i in range(args.frames):
        for serial in cam.serials:
            frame = cam.capture_frame(serial)
            if frame is not None:
                path = rf"camera_test\{serial}\test_{i}.png"
                cv2.imwrite(path, frame)
                print(f"Saved frame {i+1}/{args.frames} from {serial}")
        time.sleep(args.delay / 1000)

    cam.stop_cameras()