# Archive

Earlier, superseded decoder implementations kept for reference. Neither file is
part of the installable `reverse_photography` package and neither is maintained
going forward — the current decode pipeline lives in
`src/reverse_photography/capture/decode.py` (live capture) and
`src/reverse_photography/reconstruction/` (offline 3D reconstruction).

- `reconstruction_1to1.py` -- codebook-lookup decoder for binary structured-light
  captures, one projector cell per code.
- `reconstruction_test.py` -- an expanded version of the same idea with cell
  clustering and outlier disambiguation.

Both import from `camera_controls`, which has since moved to
`src/reverse_photography/capture/camera_controls.py`, so they will not run
as-is without adjusting that import.
