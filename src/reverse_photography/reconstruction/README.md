# 3D Reconstruction (Reverse Photography, Multi-View)

> **Disclaimer:** this README has been AI generated and not vetted by the codes original author.
> This pipeline was contributed by a different team member (Gyungmin Ko) than the rest of this repository, and did not reach working results within the project span -- likely in part due to insufficient capture and calibration quality stemming from hardware limitations. 
> The primary maintainer of this repo cannot attest to the correctness of the workflow or the accuracy of the documentation below; treat it as an unverified, as-left snapshot rather than an official description of what works.

Attempts to reconstruct a 3D point cloud from a structured-light capture taken
with two RealSense D455 cameras and a projector. This is the reconstruction
half of the project; it consumes the frames produced by `photography_run.py`.

## How it works

Triangulation: a point seen by two cameras lies where the two back-projected
rays cross. The projector solves the hard part -- matching a pixel in camera 1
to the right pixel in camera 2 -- by lighting the scene one cell (or one coded
pattern) at a time, so each surface point carries an identifying "code" in both
views.

The pipeline has four stages, one module each:

1. `calibration.py` -- factory intrinsics from each RealSense + relative pose
   (R, T) between the two cameras from checkerboard images. Run once; cached.
2. `decoding.py` -- turn the two image stacks into matched pixel pairs. This is
   the only pattern-dependent stage (raster vs hadamard vs random).
3. `triangulation.py` -- matched pixels + calibration -> 3D points.
4. `io_utils.py` -- load image stacks / measurement matrix, write the `.ply`.

`run_reconstruction.py` ties them together. `config.py` holds every value that
depends on how the data was captured.

This lives at `src/reverse_photography/reconstruction/` in the repo; paths below
are relative to this directory unless noted otherwise.

## Repo layout

```
config.py              all capture-dependent settings (edit this, not the code)
calibration.py         stereo calibration (touches the cameras; run once)
decoding.py            structured-light decoding -> correspondences
triangulation.py       correspondences -> 3D points
io_utils.py            image stack / matrix loading, PLY writing
run_reconstruction.py  main entry point (offline)
```

## Status

Done and ready (no data needed to be correct):
- Full pipeline wiring, CLI, and config layout
- RealSense intrinsics extraction and checkerboard stereo calibration
- Triangulation and PLY export
- Decoding for raster, plus a general linear decoder (works for hadamard /
  random once the measurement matrix is provided)

Waiting on the team (tracked as `TBD` in `config.py`):
- `pattern`: raster / hadamard / random
- `measurement_matrix_path` (+ `random_seed` if random)
- `projector_cols` / `projector_rows`
- Confirmation of resolution, folder layout, and frame naming
- Checkerboard captures from both cameras + board specs

Until those land, the structure is final -- only the values change.

## Running it

Install deps (from the repo root):

```
uv sync
```

Calibrate once, with both cameras attached and checkerboard images placed under
`calibration/checkerboard/<serial>/`. Run from *this* directory so the
CWD-relative paths in `config.py` resolve correctly:

```
cd src/reverse_photography/reconstruction
uv run python calibration.py
```

Then reconstruct a run (offline, no cameras needed):

```
uv run python run_reconstruction.py
```

The other pipeline scripts (`reconstruct_metric.py`, `projector_calibration.py`,
`reconstruct_projcam.py`, `synthesize_projector_view.py`, `error_heatmap.py`,
`make_extension_figure.py`) resolve their own paths from `__file__`, so they can
be run from the repo root instead, e.g.:

```
uv run python src/reverse_photography/reconstruction/reconstruct_metric.py <run_name>
```

This writes `reconstruction.ply`, viewable in MeshLab or CloudCompare.

## When the data arrives

1. Set `pattern` in `config.py`.
2. If hadamard / random, set `measurement_matrix_path` (and `random_seed`).
3. Point `run_name` / `captures_root` at the real capture folder.
4. Confirm `image_width/height`, `frame_template`, and the projector grid.
5. Run calibration, then `run_reconstruction.py`.
