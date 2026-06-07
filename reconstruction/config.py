"""
Central configuration for the reverse-photography 3D reconstruction pipeline.

Everything that depends on HOW the data was captured lives here, so that when
the capture details are finalized you only edit this file -- no pipeline code
needs to change. Search for "TBD" to find the values waiting on the team.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # --- Cameras -----------------------------------------------------------
    # RealSense serial numbers (from camera_controls.py).
    camera_serials: tuple = ("105322251697", "046322251346")
    # MUST match the resolution the cameras actually capture at (RealSense D455
    # color stream is configured for 1280x800 in camera_controls.py). The
    # checkerboard images and the factory-intrinsics read in calibration.py both
    # use this size, so a mismatch silently corrupts the calibration.
    image_width: int = 1280
    image_height: int = 800

    # --- Capture layout ----------------------------------------------------
    # Root folder produced by photography_run.py.
    captures_root: str = "captures"
    # Per-run subfolder; set this to the run you want to reconstruct.
    run_name: str = "test_run"
    # File-name template inside each camera folder. {i} is the frame index.
    frame_template: str = "frame_{i}.png"

    # --- Projector / structured light -------------------------------------
    # TBD: one of "raster", "hadamard", "random" (waiting on the team).
    pattern: str = "raster"
    # Projector grid geometry (square_size in projection_control.py).
    projector_square_size: int = 40
    # TBD: projector resolution in cells; used only to sanity-check the
    # measurement matrix. Leave None to skip the check.
    projector_cols: int = None
    projector_rows: int = None
    # Path to the saved measurement matrix (which cells were ON per frame).
    # Not needed for "raster"; required for "hadamard" / "random".
    measurement_matrix_path: str = None
    # Only used if pattern == "random".
    random_seed: int = 0

    # --- Decoding ----------------------------------------------------------
    # A pixel is trusted only if its (max - min) intensity over the sequence
    # exceeds this. Tune against real data.
    min_contrast: int = 30

    # --- Calibration -------------------------------------------------------
    # Checkerboard for stereo extrinsics. NOTE: counts are INNER corners,
    # not squares (inner corners = squares - 1 in each direction).
    # Physical board: 8 squares wide x 6 squares tall, 15 mm squares
    # -> inner corners 7 wide (cols) x 5 tall (rows). OpenCV wants (cols, rows).
    checkerboard_cols: int = 7
    checkerboard_rows: int = 5
    checkerboard_square_mm: float = 15.0
    checkerboard_dir: str = "calibration/checkerboard"
    # Where the computed stereo calibration is cached.
    calibration_path: str = "calibration/stereo_calib.npz"

    # --- Output ------------------------------------------------------------
    output_ply: str = "reconstruction.ply"

    def cam_dir(self, serial: str) -> str:
        """Folder holding one camera's frames for the current run."""
        return os.path.join(self.captures_root, self.run_name, serial)


CONFIG = Config()
