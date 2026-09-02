import cv2
import numpy as np


def make_projector_code_gradient(grid_dim=256, scale=4, out_path="projector_code_gradient.png"):
    y, x = np.indices((grid_dim, grid_dim))

    b = (255 * x / max(grid_dim - 1, 1)).astype(np.uint8)
    g = (255 * y / max(grid_dim - 1, 1)).astype(np.uint8)
    r = np.full((grid_dim, grid_dim), 255, dtype=np.uint8)

    img = np.dstack([b, g, r])

    if scale != 1:
        img = cv2.resize(
            img,
            (grid_dim * scale, grid_dim * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    cv2.imwrite(out_path, img)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    make_projector_code_gradient(
        grid_dim=256,
        scale=4,
        out_path="projector_code_gradient.png",
    )