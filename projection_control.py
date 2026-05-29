import pygame
import ctypes
import serial
import time
from screeninfo import get_monitors
import csv
import pandas as pd

class Projector:
    def __init__(self, display_number=0, square_size=40, delay_ms=200):
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
        self.patterns = []

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


    def raster_pattern(self, pattern_number):
        """Generate one pattern with a single white square at index = pattern_number"""
        surface = pygame.Surface((self.proj.width, self.proj.height))
        surface.fill((0, 0, 0))  # start black

        # Compute which cell to turn white
        x_idx = pattern_number % self.cols
        y_idx = pattern_number // self.cols
        rect = (
            x_idx * self.square_size,
            y_idx * self.square_size,
            self.square_size,
            self.square_size,
        )
        pygame.draw.rect(surface, (255, 255, 255), rect)

        return surface


    def generate_rasters(self):
        """Pre-generate all raster patterns into an ordered list"""
        total = self.cols * self.rows
        print(f"Generating {total} raster patterns...")
        self.patterns = [self.raster_pattern(i) for i in range(total)]


    def run_patterns(self):
        """Cycle through all patterns, blocking until ESP32 responds"""
        if not self.patterns:
            self.generate_rasters()

        current = 0
        running = True

        while running and current < len(self.patterns):
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # Draw current pattern
            self.screen.blit(self.patterns[current], (0, 0))
            pygame.display.flip()

            # TODO: Thread Camera triggers here
            time.sleep(self.delay)
            current += 1

        pygame.quit()
