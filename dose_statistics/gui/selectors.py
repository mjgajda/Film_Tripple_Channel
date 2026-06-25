# gui/selectors.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector, RectangleSelector
from matplotlib.lines import Line2D
from skimage.draw import line as sk_line


class EllipseCollector:
    def __init__(self, image, n, title):
        self.image = image
        self.n = n
        self.selections = []
        self.current = None
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(image, cmap="bone", aspect="equal")
        self.ax.set_title(title)
        self.selector = EllipseSelector(self.ax, self.onselect, interactive=True, useblit=True)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def onselect(self, eclick, erelease):
        self.current = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)

    def on_key(self, event):
        if event.key == 'enter' and self.current is not None:
            x1, y1, x2, y2 = self.current
            cx, cy = (x1+x2)/2, (y1+y2)/2
            rx, ry = abs(x2-x1)/2, abs(y2-y1)/2
            yy, xx = np.indices(self.image.shape)
            mask = ((xx - cx)**2)/(rx**2 + 1e-12) + ((yy - cy)**2)/(ry**2 + 1e-12) <= 1.0
            self.selections.append(mask)
            self.ax.contour(mask, levels=[0.5], colors='r', linewidths=2)
            self.fig.canvas.draw_idle()
            if len(self.selections) >= self.n:
                plt.close(self.fig)

    def run(self):
        plt.show()
        return self.selections


class RectSelectorOnce:
    def __init__(self, image, title):
        self.image = image
        self.extent = None
        self.rectangle_patch = None
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(image, cmap="bone", aspect="equal")
        self.ax.set_title(title)
        self.selector = RectangleSelector(
            self.ax, self.onselect,
            interactive=True, useblit=True, button=[1], minspanx=5, minspany=5
        )
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def onselect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        self.extent = (x, y, w, h)
        if self.rectangle_patch:
            self.rectangle_patch.remove()
        self.rectangle_patch = plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=2)
        self.ax.add_patch(self.rectangle_patch)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'enter' and self.extent is not None:
            plt.close(self.fig)

    def run(self):
        plt.show()
        if self.extent is None:
            raise RuntimeError("No rectangle selected.")
        x, y, w, h = self.extent
        crop = self.image[y:y+h, x:x+w]
        return crop, (x, y, w, h)


class LineSelector:
    def __init__(self, image, title):
        self.image = image
        self.pts = []
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(image, cmap="bone", aspect="equal")
        self.ax.set_title(title)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        self.pts.append((event.xdata, event.ydata))
        self.ax.plot(event.xdata, event.ydata, 'ro')
        self.fig.canvas.draw_idle()
        if len(self.pts) == 2:
            x0, y0 = self.pts[0]
            x1, y1 = self.pts[1]
            rr, cc = sk_line(int(y0), int(x0), int(y1), int(x1))
            self.line_rrcc = (rr, cc)
            self.ax.add_line(Line2D([x0, x1], [y0, y1], color='r', linewidth=2))
            self.fig.canvas.draw_idle()
            plt.pause(0.3)
            plt.close(self.fig)

    def run(self):
        plt.show()
        return self.line_rrcc
