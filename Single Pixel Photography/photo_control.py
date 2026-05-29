import pygame
import ctypes
import serial
import time
from screeninfo import get_monitors
import csv
import pandas as pd

class PhotoControl:
    def __init__(self, display_number=1, square_size=40, delay_ms=500, logfile_name="test_readings.csv"):

        # Configure serial interface for triggering ESP32 ReadOuts
        self.ser = serial.Serial("COM4", 9600, timeout=1)

        # Configure projector interface
        pygame.init()
        monitors = get_monitors()

        # Select monitor
        self.proj = monitors[display_number]
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

        # time parameter
        self.delay = delay_ms / 1000

        # Grid parameters
        self.square_size = square_size
        self.cols = self.proj.width // square_size
        self.rows = self.proj.height // square_size
        self.patterns = {}

        # CSV logging setup
        self.logfile = open(logfile_name, "w", newline="")
        self.csvwriter = csv.writer(self.logfile)
        self.csvwriter.writerow(["pattern", "reading"])  # header


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
        """Pre-generate all raster patterns into a dictionary"""
        total = self.cols * self.rows
        print(f"Generating {total} raster patterns...")
        self.patterns = {i: self.raster_pattern(i) for i in range(total)}

    # Runs with blocking
    def project_and_read(self, pattern_idx):
        # Trigger ESP32
        msg = f"Projecting: {pattern_idx}\n"
        self.ser.write(msg.encode())

        # Wait for ESP32 response
        line = self.ser.readline().decode().strip()
        return line


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

            # Send + wait for response
            response = self.project_and_read(current)
            time.sleep(self.delay)
            print(f"PC sent pattern {current}, ESP32 responded: {response}")

            self.csvwriter.writerow([current, int(response.split(' ')[2])])
            self.logfile.flush() # likely not necessary but if i end up running long experiments, it could be useful

            # Advance AFTER ESP32 has read
            current += 1

        pygame.quit()
        self.ser.close()

        self.logfile.close()