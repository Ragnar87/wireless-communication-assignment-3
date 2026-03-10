import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np

# %% Section 1: Signal strength calculation functions

def manhattan_distance_grid(Z, tx):
    h, w = Z.shape
    tx_x, tx_y = tx

    y, x = np.indices((h, w))  # coordinate grids

    return np.abs(x - tx_x) + np.abs(y - tx_y)

def euclidean_distance(shape, tx, cell_size=0.25):
    h, w = shape
    tx_x, tx_y = tx

    x = np.arange(w)
    y = np.arange(h)

    dx = x - tx_x
    dy = y[:, None] - tx_y

    distance_cells = np.sqrt(dx**2 + dy**2)

    return distance_cells * cell_size

def path_loss(distance, exponent_n, PL_d0):
    distance = np.asarray(distance, dtype=float)

    result = np.zeros_like(distance)

    mask = distance > 0
    result[mask] = PL_d0 + 10 * exponent_n * np.log10(distance[mask])

    return result

def other_losses(n_walls, n_windows, n_doors, p_wall, p_window, p_door):
    return n_walls * p_wall + n_windows * p_window + n_doors * p_door

def power_received(distance, n_walls, n_windows, n_doors, exponent_n, PL_d0, p_wall, p_window, p_door):
    Pt = 0 #transmit power
    Gt = 0 #transmit antenna gain
    Gr = 0 #receive antenna gain
    return Pt + Gt + Gr - path_loss(distance, exponent_n, PL_d0) - other_losses(n_walls, n_windows, n_doors, p_wall, p_window, p_door)

def snr(p_r):
    B = 1000000 #bandwidth in MHz
    NF = 6 #Noise figure
    N = -174 + 10 * math.log10(B) + NF
    return p_r - N

# %% Section 2: Path tracing and obstacle counting functions

WALL = 1
WINDOW = 2
DOOR = 3

def cells_on_line(tx, target):
    """
    Returns grid cells crossed by the line from tx to target.
    tx and target are (x, y).
    """
    x0, y0 = tx
    x1, y1 = target

    x = x0 + 0.5
    y = y0 + 0.5
    end_x = x1 + 0.5
    end_y = y1 + 0.5

    dx = end_x - x
    dy = end_y - y

    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)

    if dx != 0:
        t_delta_x = abs(1.0 / dx)
        next_vert_grid = math.floor(x) + 1 if step_x > 0 else math.floor(x)
        t_max_x = abs((next_vert_grid - x) / dx)
    else:
        t_delta_x = float("inf")
        t_max_x = float("inf")

    if dy != 0:
        t_delta_y = abs(1.0 / dy)
        next_horiz_grid = math.floor(y) + 1 if step_y > 0 else math.floor(y)
        t_max_y = abs((next_horiz_grid - y) / dy)
    else:
        t_delta_y = float("inf")
        t_max_y = float("inf")

    cx, cy = x0, y0
    cells = [(cx, cy)]

    while (cx, cy) != (x1, y1):
        if t_max_x < t_max_y:
            cx += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x:
            cy += step_y
            t_max_y += t_delta_y
        else:
            cx += step_x
            cy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y

        cells.append((cx, cy))

    return cells


def count_obstacles_to_point(grid, tx, target):
    """
    Count crossings into contiguous obstacle regions instead of counting every cell.

    Returns:
        (n_walls, n_windows, n_doors)
    """
    h, w = grid.shape
    path = cells_on_line(tx, target)

    n_walls = 0
    n_windows = 0
    n_doors = 0

    prev_material = 0
    same_material_count = 0

    for x, y in path[1:]:  # skip tx cell
        if not (0 <= x < w and 0 <= y < h):
            continue

        material = grid[y, x]

        if(material > 0 and material == prev_material):
            same_material_count += 1
            #print(f"Same material {material} count: {same_material_count}")
        else:
            same_material_count = 0
        if(material == 0):
            same_material_count = 0
        # Count only when entering a new obstacle region
        if (same_material_count == 0 or same_material_count > 10):
            if material == WALL:
                n_walls += 1
            elif material == WINDOW:
                n_windows += 1
            elif material == DOOR:
                n_doors += 1

        prev_material = material


    return n_walls, n_windows, n_doors


def count_obstacles_grid(grid, tx):
    h, w = grid.shape

    wall_counts = np.zeros((h, w), dtype=int)
    window_counts = np.zeros((h, w), dtype=int)
    door_counts = np.zeros((h, w), dtype=int)

    for y in range(h):
        for x in range(w):
            n_walls, n_windows, n_doors = count_obstacles_to_point(grid, tx, (x, y))
            wall_counts[y, x] = n_walls
            window_counts[y, x] = n_windows
            door_counts[y, x] = n_doors

    return wall_counts, window_counts, door_counts

# %% Section 3: Visualization functions

def visualize_pr_heatmap_on_floorplan(
    pr_map,
    floorplan_path,
    transmitter=None,
    title="Power Received",
    heatmap_alpha=0.55,
    vmin=None,
    vmax=None,
    flip_floorplan=True,
):
    pr_map = np.asarray(pr_map, dtype=float)
    h, w = pr_map.shape

    floorplan = mpimg.imread(floorplan_path)
    if flip_floorplan:
        floorplan = np.flipud(floorplan)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Floorplan
    ax.imshow(
        floorplan,
        origin="lower",
        extent=(0, w, 0, h)
    )

    # Heatmap
    im = ax.imshow(
        pr_map,
        origin="lower",
        extent=(0, w, 0, h),
        cmap="inferno",
        interpolation="nearest",
        alpha=heatmap_alpha,
        vmin=vmin,
        vmax=vmax
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Power received (dBm)")

    # TX marker
    if transmitter is not None:
        tx_x, tx_y = transmitter
        ax.scatter(tx_x, tx_y, s=200, marker="*", edgecolors="black", label="TX")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if transmitter is not None:
        ax.legend()

    plt.tight_layout()
    plt.show()

def visualize_snr_contours_on_floorplan(
    snr_map,
    floorplan_path,
    thresholds,
    transmitter=None,
    title="SNR Coverage Classes",
    show_labels=True,
    flip_floorplan=True,
):
    snr_map = np.asarray(snr_map, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.ndim != 1 or thresholds.size != 2:
        raise ValueError("thresholds must contain exactly 2 values to create 3 classes")

    h, w = snr_map.shape

    floorplan = mpimg.imread(floorplan_path)
    if flip_floorplan:
        floorplan = np.flipud(floorplan)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Floorplan
    ax.imshow(
        floorplan,
        origin="lower",
        extent=(0, w, 0, h)
    )

    # Contours at cell centers
    x = np.arange(w) + 0.5
    y = np.arange(h) + 0.5
    X, Y = np.meshgrid(x, y)

    cs = ax.contour(
        X, Y, snr_map,
        levels=thresholds,
        linewidths=2
    )

    if show_labels:
        fmt = {t: f"{t:g} dB" for t in thresholds}
        ax.clabel(cs, inline=True, fontsize=9, fmt=fmt)

    # TX marker
    if transmitter is not None:
        tx_x, tx_y = transmitter
        ax.scatter(tx_x + 0.5, tx_y + 0.5, s=200, marker="*", edgecolors="black", label="TX")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    if transmitter is not None:
        ax.legend()

    plt.tight_layout()
    plt.show()

def select_tx_on_floorplan(
    floorplan_path,
    grid_shape,
    title="Click transmitter location",
    flip_floorplan=True,
):
    """
    Show the floorplan and let the user click once to select TX.

    Parameters
    ----------
    floorplan_path : str
        Path to the floorplan image.
    grid_shape : tuple[int, int]
        Shape of the simulation grid: (height, width).
    title : str
        Window title.
    flip_floorplan : bool
        Flip vertically to match origin='lower'.

    Returns
    -------
    tx : tuple[int, int]
        Transmitter position as (x, y) grid cell coordinates.
    """
    h, w = grid_shape

    floorplan = mpimg.imread(floorplan_path)
    if flip_floorplan:
        floorplan = np.flipud(floorplan)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        floorplan,
        origin="lower",
        extent=(0, w, 0, h)
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    print("Click once to place the transmitter...")
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)

    if not pts:
        raise ValueError("No transmitter point was selected.")

    x, y = pts[0]

    # Convert click position to grid cell
    tx = (int(np.floor(x)), int(np.floor(y)))

    print(f"Selected TX: {tx}")
    return tx

# %% Section 4: Main runner functions

def calculate_pr(grid, tx, exponent_n, pl_d0, p_wall, p_window, p_door, cell_size):
    grid = np.asarray(grid)

    distance = euclidean_distance(grid.shape, tx, cell_size=cell_size)

    wall_counts, window_counts, door_counts = count_obstacles_grid(grid, tx)

    return power_received(distance, wall_counts, window_counts, door_counts, exponent_n, pl_d0, p_wall, p_window, p_door)

def runner(floorplan, grid, cell_size, exponent_n, pl_d0, p_wall, p_window, p_door, thresholds):
    grid = np.load(grid)
    tx = select_tx_on_floorplan(floorplan, grid.shape)
    pr_map = calculate_pr(grid, tx, exponent_n, pl_d0, p_wall, p_window, p_door, cell_size)
    snr_map = snr(pr_map)


    visualize_pr_heatmap_on_floorplan(
        pr_map,
        floorplan,
        transmitter=tx,
        title="Power received (dBm)",
        vmin=-109,
        vmax=-0
    )

    visualize_snr_contours_on_floorplan(
        snr_map,
        floorplan,
        thresholds=thresholds,
        transmitter=tx,
        title="SNR class borders"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RF coverage heatmap on floorplan")
    parser.add_argument("--floorplan", type=str, default="entire_floorplan.png", help="Path to floorplan image")
    parser.add_argument("--grid", type=str, default="entire_floorplan_grid.npy", help="Path to floorplan grid numpy file")
    parser.add_argument("--unit", type=float, default=0.25,help="Grid cell size in meters (e.g. 0.25 metres)")
    parser.add_argument("--exponent_n", type=float, default=2.622, help="Path loss exponent (e.g. 2.5 for indoor)")
    parser.add_argument("--pl_d0", type=float, default=34.93, help="Path loss at reference distance d0 (e.g. 34 dB at 1 m)")
    parser.add_argument("--p_wall", type=float, default=19.54, help="Additional loss per wall (e.g. 20 dB)")
    parser.add_argument("--p_window", type=float, default=15.64, help="Additional loss per window (e.g. 15 dB)")
    parser.add_argument("--p_door", type=float, default=1, help="Additional loss per door (e.g. 5 dB)")
    parser.add_argument("--thresholds", type=float, nargs=2, default=[0, 7], help="SNR thresholds for class borders (e.g. 0, 7 dB)")
    args = parser.parse_args()
    runner(args.floorplan, args.grid, args.unit, args.exponent_n, args.pl_d0, args.p_wall, args.p_window, args.p_door, args.thresholds)