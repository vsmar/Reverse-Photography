import pygame
import ctypes
import time
from screeninfo import get_monitors
import pandas as pd
import numpy as np 
import os
from scipy.linalg import hadamard


OUTPUT_DIR = "patterns"

class Projector:
    def __init__(self, display_number=1, square_size=40, delay_ms=200):
        self.delay = delay_ms / 1000

        # Configure projector interface
        pygame.init()
        monitors = get_monitors()

        # Select monitor
        self.proj = monitors[display_number]


        # Grid parameters
        self.square_size = square_size
        self.cols = self.proj.width // square_size
        self.rows = self.proj.height // square_size
        self.n_cells = self.cols * self.rows
        self.patterns = []

        # Create the pattern_marix 
        self.pattern_matrix = None 

        print(f"Configuring Projector: {self.proj.width}x{self.proj.height}")

        # Setup display window
        self.screen = pygame.display.set_mode(
            (self.proj.width, self.proj.height), pygame.NOFRAME
        )

        # Move window to correct monitor (Windows only)
        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.MoveWindow(
            hwnd,
            self.proj.x,
            self.proj.y,
            self.proj.width,
            self.proj.height,
            True,
        )

    # Create pygame surface 
    def surface_mask (self, mask):
        """Generate one pattern with a single white square at index = pattern_number"""
        surface = pygame.Surface((self.proj.width, self.proj.height))
        surface.fill((0, 0, 0))  # start black

        for cell in range(self.n_cells):
            if mask[cell]:
                # Compute which cell to turn white
                x_idx = cell % self.cols
                y_idx = cell // self.cols
                rect = (
                    x_idx * self.square_size,
                    y_idx * self.square_size,
                    self.square_size,
                    self.square_size,
                )

                pygame.draw.rect(surface, (255, 255, 255), rect)

        return surface

    def build_from_matrix(self, matrix):
        self.pattern_matrix = matrix.astype(np.uint8)
        self.patterns = [self.surface_mask(i) for i in self.pattern_matrix]
        return self.patterns
        

     # Generate raster pattern 
    def generate_rasters(self):
        """Pre-generate all raster patterns into an ordered list"""
        # total = self.cols * self.rows
        # print(f"Generating {total} raster patterns...")
        # return [self.raster_pattern(i) for i in range(total)]
        matrix = np.eye(self.n_cells, dtype=np.uint8)
        return self.build_from_matrix(matrix)
    
    # Generate hadamard pattern 
    def generate_hadamard(self):
        order = 1 << (self.n_cells - 1).bit_length()

        H = hadamard(order)
        pattern = H[:self.n_cells, :self.n_cells] 
        matrix = (pattern>0).astype(np.uint8)
        return self.build_from_matrix(matrix)

    def generate_random(self, pattern_number, fill_prob = 0.5, seed=0):
        random_pattern = np.random.default_rng(seed)
        matrix = (random_pattern.random((pattern_number, self.n_cells)) < fill_prob).astype(np.uint8)
        return self.build_from_matrix(matrix)
    
    def proj_pattern(self, pattern):
        """Generate one pattern with a single white square at index = pattern_number"""
        self.screen.blit(pattern, (0, 0))
        pygame.display.flip()

    def save_pattern_matrix(self, run_name="default_run", pattern="rasters"):
        patter_dir = os.path.join(OUTPUT_DIR, run_name, pattern) 
        os.makedirs(patter_dir, exist_ok=True)
        np.save(os.path.join(patter_dir, "pattern_matrix.npy"), self.pattern_matrix)
        meta = {
            "cols": self.cols,
            "rows": self.rows,
            "n_cells": self.n_cells,
            "square_size": self.square_size,
            "proj_width": self.proj.width,
            "proj_height": self.proj.height,
        }
        np.save(os.path.join(patter_dir, "grid_meta.npy"), meta)   

    def test_projection(self):
        """Cycle through all patterns, blocking until ESP32 responds"""
        if not self.patterns:
            self.patterns = self.generate_rasters()

        current = 0
        running = True

        while running and current < len(self.patterns):
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # Draw current pattern
            self.proj_pattern(self.patterns[current])

            time.sleep(self.delay)
            # TODO: Thread Camera triggers here

            current += 1

        pygame.quit()
