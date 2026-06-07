import pygame
import numpy as np
from scipy.linalg import hadamard
from screeninfo import get_monitors

class Projector:
    def __init__(self, display_number=1, pattern_res_pxl=2, inverse=True):
        self.pattern_res_pxl = pattern_res_pxl
        self.inverse = inverse
        pygame.init()
        monitors = get_monitors()
        self.proj = monitors[display_number] if display_number < len(monitors) else monitors[0]
        self.grid_dimensions = int(2 ** np.floor(np.log2(min(self.proj.width, self.proj.height) / self.pattern_res_pxl)))
        self.n_cells = self.grid_dimensions ** 2
        self.grid_px = self.grid_dimensions * self.pattern_res_pxl
        self.grid_x = (self.proj.width - self.grid_px) // 2
        self.grid_y = (self.proj.height - self.grid_px) // 2
        
        self.screen = pygame.display.set_mode((self.proj.width, self.proj.height), pygame.NOFRAME, display=display_number)
        self.screen.fill((0, 0, 0))
        pygame.display.flip()

        self._tiny = pygame.Surface((self.grid_dimensions, self.grid_dimensions), depth=8)
        self._tiny.set_palette([(0, 0, 0), (255, 255, 255)])
        
        self.pattern_matrix = None
        self.patterns = []
        self.num_base_patterns = 0

    def _store_matrix(self, matrix):
        self.num_base_patterns = len(matrix)
        if self.inverse:
            interleaved = np.empty((2*len(matrix), matrix.shape[1]), dtype=matrix.dtype)
            interleaved[0::2] = matrix
            interleaved[1::2] = 1 - matrix
            self.pattern_matrix = interleaved
        else:
            self.pattern_matrix = matrix
            
        self.patterns = range(len(self.pattern_matrix))
        return self.patterns

    def generate_rasters(self):
        return self._store_matrix(np.eye(self.n_cells, dtype=np.uint8))

    def generate_hadamard(self):
        order = 1 << (self.n_cells - 1).bit_length()
        H = hadamard(order)
        sign_changes = np.diff(np.sign(H), axis=1)
        sequency = (sign_changes != 0).sum(axis=1)
        H = H[np.argsort(sequency)]
        return self._store_matrix((H[:self.n_cells, :] > 0).astype(np.uint8))

    def generate_structured_light(self):
        """ Takes advantage of having a multi photodiode array (camera)"""
        A = np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]], dtype=np.uint8)
        B = np.array([[0,0,1,0],[1,1,1,0],[0,1,1,1],[0,1,0,0]], dtype=np.uint8)
        g = self.grid_dimensions
        n_scales = int(np.log2(g // 4)) + 1
        patterns = []
        for i in range(n_scales + 1):
            block = 2 ** i
            scaled_size = 4 * block
            A_s = np.kron(A, np.ones((block, block), dtype=np.uint8))
            B_s = np.kron(B, np.ones((block, block), dtype=np.uint8))
            if scaled_size < g:
                reps = g // scaled_size
                A_s = np.tile(A_s, (reps, reps))
                B_s = np.tile(B_s, (reps, reps))
            pw, ph = max(0, g - A_s.shape[0]), max(0, g - A_s.shape[1])
            patterns.append(np.pad(A_s, ((0, pw), (0, ph)))[:g, :g].flatten())
            patterns.append(np.pad(B_s, ((0, pw), (0, ph)))[:g, :g].flatten())
        patterns.append(np.ones(g * g, dtype=np.uint8))
        return self._store_matrix(np.array(patterns)[::-1]) # corrects order

    def proj_pattern(self, pattern_index):
        row = self.pattern_matrix[pattern_index].reshape((self.grid_dimensions, self.grid_dimensions))
        pygame.surfarray.blit_array(self._tiny, row.T)
        scaled = pygame.transform.scale(self._tiny, (self.grid_px, self.grid_px))
        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled, (self.grid_x, self.grid_y))
        pygame.display.flip()
        
    def quit(self):
        pygame.quit()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=4, help="Pattern resolution in pixels.")
    parser.add_argument("--delay", type=int, default=500, help="Interval between projections (ms).")
    parser.add_argument("--pattern", choices=["raster", "hadamard", "structured"], 
                            default="structured", help="Type of pattern to project.")
    parser.add_argument("--inverse", action="store_true", help="Include inverse patterns.")
    args = parser.parse_args()

    p = Projector(inverse=args.inverse, pattern_res_pxl=args.res)

    if args.pattern == "raster":        p.generate_rasters()
    elif args.pattern == "hadamard":    p.generate_hadamard()
    elif args.pattern == "structured":  p.generate_structured_light()

    for i in range(len(p.patterns)):
        pygame.event.pump()  # keep pygame's event queue from clogging
        p.proj_pattern(i)
        pygame.time.wait(args.delay)   # yields to OS, doesn't burn CPU
        
    p.quit()