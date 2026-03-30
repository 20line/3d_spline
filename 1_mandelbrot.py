from typing import Optional, Tuple
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

class MandelbrotSetOnGrid():
    def __init__(self, resolution_x: int, max_iterations: int = 255, escape_radius: float = 1000):
        self.resolution_x = resolution_x
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def calculate_mandelbrot_set(self, pt_complex_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        c = self._complex_matrix(pt_complex_bounds)
        stability_map = np.empty(c.shape, dtype=np.float64)
        stability_map.fill(self.max_iterations)
        z = 0
        for iteration in range(self.max_iterations):
            z = z ** 2 + c
            z_norm = abs(z)
            subset = z_norm > self.escape_radius
            stability_map[subset] = iteration
            z[subset], c[subset] = 0, 0
        stability_map /= self.max_iterations
        self.comp_mode = "np"
        return stability_map

    def _complex_matrix(self, pt_complex_bounds: Tuple[float, float, float, float]) -> np.ndarray:
        dx = pt_complex_bounds[1] - pt_complex_bounds[0]
        dy = pt_complex_bounds[3] - pt_complex_bounds[2]
        x = np.linspace(pt_complex_bounds[0], pt_complex_bounds[1], self.resolution_x)
        y = np.linspace(pt_complex_bounds[2], pt_complex_bounds[3], int(np.abs(dy/dx) * self.resolution_x))
        return x[np.newaxis, :] + y[:, np.newaxis] * 1j

class MandelbrotDraw(MandelbrotSetOnGrid):
    def __init__(self, axs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if axs is None:
            self.fig, self.ax_main = plt.subplots()
            self.axs = (self.ax_main,)
        else:
            self.axs = axs
            self.ax_main = self.axs[0]
        self.pt_complex_bounds = None
        self.draw_mandelbrot_map((-2., 2., -1., 1.))

    def postproc(self, mandelbrot_map: np.ndarray):
        pass

    def draw_mandelbrot_map(self, pt_complex_bounds: Tuple[float, float, float, float]):
        start_time = time.time()
        for ax in self.axs:
            ax.clear()
        self.pt_complex_bounds = np.array(pt_complex_bounds)
        mandelbrot_map = self.calculate_mandelbrot_set(pt_complex_bounds)[::-1, :]
        self.ax_main.imshow(mandelbrot_map, extent=pt_complex_bounds)
        self.ax_main.callbacks.connect('ylim_changed', self._on_ylims_change)
        self.ax_main.set_title(
            "Используйте инструмент ПРИБЛИЖЕНИЯ (лупа) для выбора области.\n"
            "После выбора закройте окно — контур сохранится.")
        self.postproc(mandelbrot_map)
        print(f"Timer {time.time() - start_time:.3f} sec, mode: {self.comp_mode}", end="\r")

    def _on_ylims_change(self, event_ax):
        xlim = self.ax_main.get_xlim()
        ylim = self.ax_main.get_ylim()
        pt_complex_bounds = np.array((*xlim, *ylim))
        if self.pt_complex_bounds is None or np.any(pt_complex_bounds != self.pt_complex_bounds):
            self.draw_mandelbrot_map(pt_complex_bounds)

class MandelbrotContourDraw(MandelbrotDraw):
    def __init__(self, debug: bool = False, *args, **kwargs):
        self.debug = debug
        ax_config = (2, 2) if self.debug else (2, 1)
        self.fig, axs = plt.subplots(*ax_config, sharex=True, sharey=True)
        self.ax_contour = axs.flatten()[-1]
        self.fig.set_size_inches((15, 7))
        self.contours_draw_num = 10
        self.viz_step = 10
        super().__init__(axs.flatten(), *args, **kwargs)

    def postproc(self, mandelbrot_map: np.ndarray):
        image_blur = cv2.GaussianBlur(mandelbrot_map, (7, 7), sigmaX=2)
        _, image_bin = cv2.threshold(image_blur, 0.9, 1, cv2.THRESH_BINARY)
        self.contours = self._find_contours(image_bin)

        for contour in self.contours[:self.contours_draw_num]:
            t_grid = np.arange(0, contour.shape[1], self.viz_step, dtype=np.uint32)
            if t_grid.size < 2:
                continue
            self.ax_contour.scatter(contour[0, t_grid], contour[1, t_grid], s=10, marker="*")
            self.ax_contour.set_aspect(1)
            self.ax_contour.set_title(f"Точки {self.contours_draw_num} самых длинных контуров")

        np.savetxt("contour.txt", self.contours[0].T)

        if self.debug:
            self.axs[1].imshow(image_blur, extent=self.pt_complex_bounds)
            self.axs[1].set_title("Размытое")
            self.axs[2].imshow(image_bin, extent=self.pt_complex_bounds)
            self.axs[2].set_title("Бинарное")

    def _find_contours(self, image_bin: np.ndarray):
        contours_pix, _ = cv2.findContours(image_bin.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        h, w = image_bin.shape[:2]
        pad = 1
        contours_pix_clean = []
        for contour_pix in contours_pix:
            contour_pix = contour_pix[:, 0, :].T
            subset = (contour_pix[0] >= pad) & (contour_pix[1] >= pad) & (contour_pix[0] < w-pad) & (contour_pix[1] < h-pad)
            bounding_contour_ids = np.where(subset[:-1] != subset[1:])[0]
            if bounding_contour_ids.size > 0:
                splits = np.r_[0, bounding_contour_ids, contour_pix.shape[1]-1]
                for i in range(splits.size - 1):
                    if subset[splits[i]+1]:
                        contours_pix_clean.append(contour_pix[:, splits[i]:splits[i+1]+1])
            else:
                contours_pix_clean.append(contour_pix)

        b = self.pt_complex_bounds
        M_im2complex = np.array([
            [(b[1] - b[0])/w, 0, b[0]],
            [0, -(b[3] - b[2])/h, b[3]]
        ])
        contours_complex = []
        for contour_pix in contours_pix_clean:
            contour_pix_h = np.vstack([contour_pix, np.ones(contour_pix.shape[1])])
            contours_complex.append(M_im2complex @ contour_pix_h)
        return sorted(contours_complex, key=lambda c: c.shape[1], reverse=True)

if __name__ == '__main__':
    matplotlib.use("TKAgg")
    md = MandelbrotContourDraw(resolution_x=1024)
    plt.show()
