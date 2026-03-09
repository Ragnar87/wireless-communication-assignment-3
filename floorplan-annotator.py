import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class FloorplanGridAnnotator:
    """
    Keys:
      0 = open space
      1 = wall
      2 = window
      3 = door

      m = paint mode
      r = rectangle mode
      h = horizontal/vertical line mode

      u = undo
      g = toggle grid
      c = clear
      s = save
      l = load
      f = fullscreen
      q = quit
    """

    def __init__(self, image_path, unit_m=0.5, pixels_per_meter=None, door_width_px=None, door_width_m=1.0):
        if pixels_per_meter is None:
            if door_width_px is None:
                raise ValueError("Provide pixels_per_meter or door_width_px.")
            pixels_per_meter = float(door_width_px) / float(door_width_m)

        self.image_path = image_path
        self.img = np.flipud(mpimg.imread(image_path))
        self.unit_m = float(unit_m)
        self.ppm = float(pixels_per_meter)
        self.cell_px = max(1, int(round(self.ppm * self.unit_m)))

        self.h_px = self.img.shape[0]
        self.w_px = self.img.shape[1]

        self.h = self.h_px // self.cell_px
        self.w = self.w_px // self.cell_px

        self.grid = np.zeros((self.h, self.w), dtype=np.uint8)
        self.extent = (0, self.w, 0, self.h)

        self.paint_value = 1
        self.mode = "paint"
        self.show_grid = True
        self.mouse_down = False
        self.last_cell = None
        self.rect_start = None
        self.line_start = None
        self.undo_stack = []
        self.max_undo = 20
        self.save_path = "annotation_grid.npy"

        self.cmap = plt.cm.get_cmap("tab10", 4)
        self.alpha = 0.35
        self.preview_artist = None

    def _push_undo(self):
        self.undo_stack.append(self.grid.copy())
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def _autosave(self):
        np.save(self.save_path, self.grid)

    def _status_text(self):
        names = {0: "OPEN", 1: "WALL", 2: "WINDOW", 3: "DOOR"}
        return (
            f"Material: {self.paint_value} ({names[self.paint_value]}) | "
            f"Mode: {self.mode} | "
            f"Grid: {self.w}x{self.h} | "
            f"Keys: 0/1/2/3 m r h u s l g f c q"
        )

    def _update_overlay(self):
        self.overlay.set_data(self.grid)
        self.status.set_text(self._status_text())
        self.fig.canvas.draw_idle()

    def _pixel_to_cell(self, xdata, ydata):
        if xdata is None or ydata is None:
            return None
        col = int(np.floor(xdata))
        row = int(np.floor(ydata))
        if 0 <= row < self.h and 0 <= col < self.w:
            return row, col
        return None

    def _apply_gridlines(self):
        self.ax.set_xticks(np.arange(0, self.w + 1, 1), minor=True)
        self.ax.set_yticks(np.arange(0, self.h + 1, 1), minor=True)
        if self.show_grid:
            self.ax.grid(which="minor", linewidth=0.3)
        else:
            self.ax.grid(False)

    def _paint_cell(self, cell):
        if cell is None or cell == self.last_cell:
            return
        row, col = cell
        self.grid[row, col] = self.paint_value
        self.last_cell = cell

    def _fill_rectangle(self, start_cell, end_cell):
        if start_cell is None or end_cell is None:
            return

        r1, c1 = start_cell
        r2, c2 = end_cell

        r_min, r_max = sorted((r1, r2))
        c_min, c_max = sorted((c1, c2))

        self.grid[r_min:r_max + 1, c_min:c_max + 1] = self.paint_value

    def _snap_axis_line(self, start_cell, end_cell):
        """
        Constrain the line to either horizontal or vertical,
        depending on which movement is larger.
        """
        if start_cell is None or end_cell is None:
            return None, None

        r1, c1 = start_cell
        r2, c2 = end_cell

        if abs(c2 - c1) >= abs(r2 - r1):
            # horizontal
            r2 = r1
        else:
            # vertical
            c2 = c1

        return start_cell, (r2, c2)

    def _fill_axis_line(self, start_cell, end_cell):
        start_cell, end_cell = self._snap_axis_line(start_cell, end_cell)
        if start_cell is None or end_cell is None:
            return

        r1, c1 = start_cell
        r2, c2 = end_cell

        if r1 == r2:
            c_min, c_max = sorted((c1, c2))
            self.grid[r1, c_min:c_max + 1] = self.paint_value
        elif c1 == c2:
            r_min, r_max = sorted((r1, r2))
            self.grid[r_min:r_max + 1, c1] = self.paint_value

    def _clear_preview(self):
        if self.preview_artist is not None:
            self.preview_artist.remove()
            self.preview_artist = None

    def _draw_line_preview(self, start_cell, end_cell):
        self._clear_preview()

        start_cell, end_cell = self._snap_axis_line(start_cell, end_cell)
        if start_cell is None or end_cell is None:
            return

        r1, c1 = start_cell
        r2, c2 = end_cell

        x1 = c1 + 0.5
        y1 = r1 + 0.5
        x2 = c2 + 0.5
        y2 = r2 + 0.5

        self.preview_artist, = self.ax.plot(
            [x1, x2], [y1, y2],
            linestyle="--",
            linewidth=2
        )
        self.fig.canvas.draw_idle()

    def _toggle_fullscreen(self):
        manager = plt.get_current_fig_manager()
        try:
            manager.full_screen_toggle()
        except Exception:
            pass

    def _maximize_window(self):
        manager = plt.get_current_fig_manager()
        try:
            manager.window.state("zoomed")  # TkAgg on Windows
            return
        except Exception:
            pass
        try:
            manager.window.showMaximized()  # Qt
            return
        except Exception:
            pass

    def _on_key_press(self, event):
        key = event.key

        if key in ("0", "1", "2", "3"):
            self.paint_value = int(key)

        elif key == "m":
            self.mode = "paint"

        elif key == "r":
            self.mode = "rectangle"

        elif key == "h":
            self.mode = "line"

        elif key == "u":
            if self.undo_stack:
                self.grid[:] = self.undo_stack.pop()
                self._autosave()

        elif key == "g":
            self.show_grid = not self.show_grid
            self._apply_gridlines()

        elif key == "f":
            self._toggle_fullscreen()

        elif key == "c":
            self._push_undo()
            self.grid[:] = 0
            self._autosave()

        elif key == "s":
            np.save(self.save_path, self.grid)
            print(f"Saved: {self.save_path}")

        elif key == "l":
            if os.path.exists(self.save_path):
                loaded = np.load(self.save_path)
                if loaded.shape != self.grid.shape:
                    print(f"Cannot load {self.save_path}: shape {loaded.shape} != {self.grid.shape}")
                else:
                    self._push_undo()
                    self.grid[:] = loaded
                    print(f"Loaded: {self.save_path}")
            else:
                print(f"No file found: {self.save_path}")

        elif key == "q":
            plt.close(self.fig)
            return

        self._update_overlay()

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return

        cell = self._pixel_to_cell(event.xdata, event.ydata)
        if cell is None:
            return

        self.mouse_down = True
        self.last_cell = None
        self._push_undo()

        if self.mode == "paint":
            self._paint_cell(cell)
            self._autosave()
            self._update_overlay()

        elif self.mode == "rectangle":
            self.rect_start = cell

        elif self.mode == "line":
            self.line_start = cell
            self._draw_line_preview(self.line_start, cell)

    def _on_mouse_release(self, event):
        if not self.mouse_down:
            return

        self.mouse_down = False

        if self.mode == "rectangle":
            end_cell = self._pixel_to_cell(event.xdata, event.ydata)
            self._fill_rectangle(self.rect_start, end_cell)
            self.rect_start = None
            self._autosave()
            self._update_overlay()

        elif self.mode == "line":
            end_cell = self._pixel_to_cell(event.xdata, event.ydata)
            self._fill_axis_line(self.line_start, end_cell)
            self.line_start = None
            self._clear_preview()
            self._autosave()
            self._update_overlay()

        self.last_cell = None

    def _on_mouse_move(self, event):
        if not self.mouse_down:
            return
        if event.inaxes != self.ax:
            return

        cell = self._pixel_to_cell(event.xdata, event.ydata)

        if self.mode == "paint":
            self._paint_cell(cell)
            self._autosave()
            self._update_overlay()

        elif self.mode == "line":
            self._draw_line_preview(self.line_start, cell)

    def run(self, save_path="annotation_grid.npy", autoload=True, maximize=True):
        self.save_path = save_path

        if autoload and os.path.exists(self.save_path):
            loaded = np.load(self.save_path)
            if loaded.shape == self.grid.shape:
                self.grid[:] = loaded
                print(f"Loaded existing grid: {self.save_path}")

        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.ax.set_title("Floorplan Annotator")

        self.ax.imshow(self.img, origin="lower", extent=self.extent)

        self.overlay = self.ax.imshow(
            self.grid,
            origin="lower",
            extent=self.extent,
            cmap=self.cmap,
            alpha=self.alpha,
            interpolation="nearest",
            vmin=0,
            vmax=3,
        )

        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)
        self.ax.set_aspect("equal")

        self._apply_gridlines()

        self.status = self.ax.text(
            0.01,
            1.01,
            self._status_text(),
            transform=self.ax.transAxes,
            fontsize=10,
            va="bottom"
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        if maximize:
            self._maximize_window()

        plt.show()

        np.save(self.save_path, self.grid)
        with open("grid_python.txt", "w") as f:
            f.write("grid = ")
            f.write(str(self.grid.tolist()))
        print(f"Saved on exit: {self.save_path}")

        return self.grid
    
def measure_pixels(image_path):
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(36, 27))
    ax.imshow(img)
    ax.set_title("Click two points (door edges). Close window when done.")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    (x1, y1), (x2, y2) = pts
    dist = float(np.hypot(x2 - x1, y2 - y1))
    print(f"Pixel distance: {dist:.2f}px")
    return dist

def runner(image_path, unit_m, door_width_px, save_path, autoload, maximize):
    door_width_px = measure_pixels(image_path=image_path)
        
    annotator = FloorplanGridAnnotator(
        image_path=image_path,
        unit_m=unit_m,
        pixels_per_meter=door_width_px # assuming a door is 1 m
    )

    grid = annotator.run(save_path=save_path, autoload=autoload, maximize=maximize)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Floorplan annotator")
    args.add_argument("--image", type=str, default="entire_floorplan.png", help="Path to floorplan image")
    args.add_argument("--unit", type=float, default=0.25, help="Grid cell size in meters")
    args.add_argument("--door-px", type=float, default=None, help="Door width in pixels (used if ppm not provided)")
    args.add_argument("--save", type=str, default="my_floorplan_grid.npy", help="Path to save annotation grid")
    args.add_argument("--autoload", action="store_true", help="Automatically load existing grid if found")
    args.add_argument("--maximize", action="store_true", help="Start with maximized window")
    args = args.parse_args()

    runner(image_path=args.image, unit_m=args.unit, door_width_px=args.door_px, save_path=args.save, autoload=args.autoload, maximize=args.maximize)