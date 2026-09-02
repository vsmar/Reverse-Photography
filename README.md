# Reverse Photography

3D scanning by inverting the usual roles of camera and light: instead of a
fixed light source and a camera that captures shape, a projector "sees" the
scene by encoding a unique binary pattern into every one of its cells, and two
RealSense cameras decode which cell lit each pixel. That per-pixel identity is
what makes structured-light triangulation possible.

The project has two halves that share one capture:

1. **Capture + live decode** (`src/reverse_photography/capture/`) -- drives the
   projector and both cameras, decodes each camera's frames into a
   projector's-eye "dual photo", and fuses the two cameras into one relit
   image. Needs the hardware attached.
2. **Offline 3D reconstruction** (`src/reverse_photography/reconstruction/`) --
   an early effort to also turn a capture into a calibrated point cloud.
   Attempts metric reconstruction via real stereo calibration, and a further
   extension that calibrates the projector itself as a third camera. See the
   disclaimer in its own README -- it did not reach working results within the project span.

## Repo layout

```
src/reverse_photography/
  capture/           camera + projector drivers, capture run, live decode, fuse
    camera_controls.py       RealSense D455 pipeline wrapper
    projection_control.py    projector pattern generation + display
    photography_run.py       drives a full capture session
    decode.py                per-camera structured-light decode -> dual photo
    fuse.py                  combine both cameras' dual photos into one image
    main.py                  runs the full chain: photograph -> decode -> fuse
  reconstruction/    offline 3D reconstruction attempt -- see its README.md

archive/             superseded decoder experiments, kept for reference only
captures/            capture run output (large; see note below)
camera_test/         ad hoc camera test shots
visual_aids/         projector_gradient.py and the test pattern it generates
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and physical RealSense D455 cameras
+ a projector for the capture half (the reconstruction half only needs
previously captured data).

```
uv sync
```

This creates a `.venv` and installs everything from `pyproject.toml`
(numpy, opencv-python, pygame, pyrealsense2, scipy, screeninfo, tqdm,
matplotlib).

## Running a capture

From the repo root, with both cameras and the projector attached, run the
whole chain (photograph -> decode -> fuse) at once:

```
uv run python src/reverse_photography/capture/main.py --run-name my_scan
```

This projects the structured-light pattern set, captures synchronized frames
from both cameras into `captures/my_scan/`, decodes each camera's frames into
a `dual_photo.png`, and fuses both cameras into one relit image. See
`--help` for pattern choice, timing, exposure, etc. -- the defaults match what
we normally ran with. The three stages are also runnable individually via
`photography_run.py`, `decode.py`, and `fuse.py` in the same directory.

## Running the offline 3D reconstruction

See [src/reverse_photography/reconstruction/README.md](src/reverse_photography/reconstruction/README.md).

## Data folders

`captures/` and `camera_test/` hold real capture data. New capture runs are not auto-tracked going forward (see `.gitignore`) -- add them explicitly if you want a run committed.

## Course project

Final project for University of Washington CSE 576 (Computer Vision).

- **Victor Marcenac** -- capture + decode/fuse pipeline
- **Tawsif Ahmed** -- capture + decode/fuse pipeline
- **Bahaa Alattar** -- CNN denoising (not part of this repo)
- **Gyungmin Ko** -- 3D reconstruction
