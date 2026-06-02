"""Run the full reverse-photography 3D reconstruction.

Offline only -- needs the cached stereo calibration and the captured frames,
but NO cameras attached. Adjust paths and the pattern in config.py, then:

    python run_reconstruction.py
"""
from config import CONFIG
from calibration import load_calibration
from decoding import build_correspondences
from triangulation import triangulate
from io_utils import load_image_stack, load_measurement_matrix, save_ply


def main(cfg=CONFIG):
    s1, s2 = cfg.camera_serials

    # 1. Calibration: computed once by calibration.py, just loaded here.
    K1, d1, K2, d2, R, T = load_calibration(cfg.calibration_path)

    # 2. Load the two image stacks for this run.
    stack1 = load_image_stack(cfg.cam_dir(s1), cfg.frame_template)
    stack2 = load_image_stack(cfg.cam_dir(s2), cfg.frame_template)
    print(f"Loaded stacks: {stack1.shape} and {stack2.shape}")

    # 3. The measurement matrix is only needed for hadamard / random.
    M = None
    if cfg.pattern in ("hadamard", "random"):
        if cfg.measurement_matrix_path is None:
            raise ValueError(
                f"pattern '{cfg.pattern}' needs config.measurement_matrix_path"
            )
        M = load_measurement_matrix(cfg.measurement_matrix_path)

    # 4. Decode -> match -> triangulate -> save.
    pts1, pts2 = build_correspondences(stack1, stack2, M, cfg.min_contrast)
    print(f"Matched {len(pts1)} pixel pairs across the two cameras.")

    points = triangulate(pts1, pts2, K1, d1, K2, d2, R, T)
    print(f"Triangulated {len(points)} 3D points.")

    save_ply(cfg.output_ply, points)
    print(f"Saved point cloud to {cfg.output_ply}")


if __name__ == "__main__":
    main()
