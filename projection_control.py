import pygame
import time
from screeninfo import get_monitors
import numpy as np
import os
from scipy.linalg import hadamard

OUTPUT_DIR = "patterns"


class Projector:
    def __init__(self, display_number=1, square_size=2, delay_ms=200):
        self.delay = delay_ms / 1000
        self.square_size = square_size or 1

        pygame.init()
        monitors = get_monitors()
        self.proj = monitors[display_number]

        # Largest power-of-2 grid that fits the projector
        self.max_power_2 = int(np.floor(np.log2(
            min(self.proj.width, self.proj.height) / self.square_size
        )))
        self.grid_dimensions = int(2 ** self.max_power_2)
        self.n_cells = self.grid_dimensions ** 2

        # Pixel extent of the square grid
        self.grid_px = self.grid_dimensions * self.square_size

        # Center the grid on the projector surface
        self.grid_x = (self.proj.width  - self.grid_px) // 2
        self.grid_y = (self.proj.height - self.grid_px)

        # Full-screen window on the chosen projector
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{self.proj.x},{self.proj.y}"
        self.screen = pygame.display.set_mode(
            (self.proj.width, self.proj.height), pygame.NOFRAME
        )
        self.screen.fill((0, 0, 0))
        pygame.display.flip()

        self.pattern_matrix = None
        self.patterns = []

        print(
            f"Configuration:\n"
            f"  Projector:  {self.proj.width}x{self.proj.height}\n"
            f"  Grid:       {self.grid_dimensions}x{self.grid_dimensions} cells\n"
            f"  Square:     {self.square_size} px\n"
            f"  Grid area:  {self.grid_px}x{self.grid_px} px "
            f"centered at ({self.grid_x}, {self.grid_y})"
        )

    # ------------------------------------------------------------------ #
    #  Pattern storage — generation is just storing the matrix           #
    # ------------------------------------------------------------------ #

    def _store_matrix(self, matrix):
        """Store pattern matrix and return lightweight index handles."""
        self.pattern_matrix = matrix.astype(np.uint8)
        self.patterns = range(len(matrix))
        return self.patterns

    def generate_rasters(self):
        """One-hot (identity) patterns — one lit cell per frame."""
        return self._store_matrix(np.eye(self.n_cells, dtype=np.uint8))

    def _hadamard_sequency_order(self, H):
        """Reorder Hadamard rows by sequency (Walsh ordering) — 
        gives spatially diverse patterns from the start."""
        n = H.shape[0]
        # Number of sign changes per row = sequency
        sign_changes = np.diff(np.sign(H), axis=1)
        sequency = (sign_changes != 0).sum(axis=1)
        return H[np.argsort(sequency)]

    def generate_hadamard(self):
        order  = 1 << (self.n_cells - 1).bit_length()
        H      = hadamard(order)
        H      = self._hadamard_sequency_order(H)   # Walsh-ordered
        matrix = (H[:self.n_cells, :] > 0).astype(np.uint8)
        return self._store_matrix(matrix)
    
    # ------------------------------------------------------------------ #
    #  Pixel array sensing approach: using minimal patterned matrices    #
    # ------------------------------------------------------------------ #

    A = np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]], dtype=np.uint8)
    B = np.array([[0,0,1,0],[1,1,1,0],[0,1,1,1],[0,1,0,0]], dtype=np.uint8)

    def generate_multiscale_masks(self):
        """
        Core multiscale pattern generator.
        tile=True:  fill the grid by tiling the scaled mask
        tile=False: show only the top-left corner, pad remainder with zeros
        """
        n_scales = int(np.log2(self.grid_dimensions // 4)) + 1
        g = self.grid_dimensions
        patterns = []

        def fit_to_grid(arr):
            out = np.zeros((g, g), dtype=np.uint8)
            h = min(arr.shape[0], g)
            w = min(arr.shape[1], g)
            out[:h, :w] = arr[:h, :w]
            return out

        for i in range(n_scales + 1):
            block = 2 ** i          # pixels per base-mask cell: 1, 2, 4, ..., grid_dim/4 + 1 (corner of matrix)
            scaled_size = 4 * block # full scaled mask size in pixels

            A_scaled = np.kron(self.A, np.ones((block, block), dtype=np.uint8))
            B_scaled = np.kron(self.B, np.ones((block, block), dtype=np.uint8))

            if (scaled_size < g):
                reps = g // scaled_size          # how many times to tile
                A_scaled = np.tile(A_scaled, (reps, reps))
                B_scaled = np.tile(B_scaled, (reps, reps))
            
            A_scaled = fit_to_grid(A_scaled)
            B_scaled = fit_to_grid(B_scaled)

            patterns.append(A_scaled.flatten())
            patterns.append(B_scaled.flatten())

        patterns.append(np.ones(g * g, dtype=np.uint8))  # white reference
        return self._store_matrix(np.array(patterns)[::-1])

    # Alias so callers using either name work
    def generate_structured_light(self):
        return self.generate_multiscale_masks()

    def generate_structured_light_complementary(self):
        """Complementary version of the structured-light set.

        Layout (this is what the decoder relies on):
          row 0            : all-white reference
          rows 1,2         : code_0,  inverse(code_0)
          rows 3,4         : code_1,  inverse(code_1)
          ...
        Each code bit is decided by comparing a pattern frame against its
        inverse frame (which is brighter), which is far more robust to
        glossy/specular surfaces than a single brightness threshold.
        """
        # Build the same code rows as the normal generator, minus the white
        # reference, in the same order.
        base = self.generate_multiscale_masks()           # stores reversed matrix
        full = self.pattern_matrix.copy()                 # (white_ref + codes)
        white = full[0:1]                                 # (1, n_cells)
        codes = full[1:]                                  # (K, n_cells)

        # Interleave each code with its inverse
        interleaved = np.empty((codes.shape[0] * 2, codes.shape[1]), dtype=np.uint8)
        interleaved[0::2] = codes
        interleaved[1::2] = 1 - codes

        matrix = np.vstack([white, interleaved])          # (1 + 2K, n_cells)
        return self._store_matrix(matrix)

    # ------------------------------------------------------------------ #
    #  Projection                                                        #
    # ------------------------------------------------------------------ #

    def proj_pattern(self, pattern_index):
        """
        Render one pattern frame:
          1. Reshape the matrix row to a tiny (grid_dim x grid_dim) surface.
          2. Scale it up to full grid_px size in C via transform.scale.
          3. Blit onto a black full-screen canvas at the centred offset.
        """
        row = self.pattern_matrix[pattern_index].reshape(
            self.grid_dimensions, self.grid_dimensions
        ).astype(np.uint8)

        # 8-bit palette surface: 0 = black, 1 = white
        tiny = pygame.Surface((self.grid_dimensions, self.grid_dimensions), depth=8)
        tiny.set_palette([(0, 0, 0), (255, 255, 255)])
        pygame.surfarray.blit_array(tiny, row.T)  # .T: surfarray is (x, y)

        # Scale up using nearest-neighbour in C — fast and exact for B&W
        scaled = pygame.transform.scale(tiny, (self.grid_px, self.grid_px))

        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled, (self.grid_x, self.grid_y))
        pygame.display.flip()

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save_pattern_matrix(self, run_name="default_run", pattern="rasters",
                            complementary=True):
        pattern_dir = os.path.join(OUTPUT_DIR, run_name, pattern)
        os.makedirs(pattern_dir, exist_ok=True)
        np.save(os.path.join(pattern_dir, "pattern_matrix.npy"), self.pattern_matrix)
        meta = {
            "grid_dimensions": self.grid_dimensions,
            "n_cells":         self.n_cells,
            "square_size":     self.square_size,
            "grid_px":         self.grid_px,
            "grid_x":          self.grid_x,
            "grid_y":          self.grid_y,
            "proj_width":      self.proj.width,
            "proj_height":     self.proj.height,
            "complementary":   complementary,
        }
        np.save(os.path.join(pattern_dir, "grid_meta.npy"), meta)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def quit(self):
        """Cleanly tear down pygame. Call instead of bare pygame.quit()."""
        pygame.quit()

    def test_projection(self):
        """Cycle through all patterns. Press ESC to abort."""
        if not len(self.patterns):
            self.generate_rasters()

        running = True
        for i in self.patterns:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break
            if not running:
                break

            self.proj_pattern(i)
            time.sleep(self.delay)

        self.quit()