import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "build", "visualization")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(OUTPUT_DIR, ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def gaussian_model(params, x, y):
    dx = x - params[5]
    dy = y - params[6]
    ct = np.cos(params[4])
    st = np.sin(params[4])
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    q = u * u / (params[2] * params[2]) + v * v / (
        params[3] * params[3]
    )
    return params[0] + params[1] * np.exp(-0.5 * q)


def save_surface(path, title, x_grid, y_grid, z_grid, limits):
    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        x_grid, y_grid, z_grid, cmap="viridis", linewidth=0, antialiased=True
    )
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_xlim(limits[0])
    axis.set_ylim(limits[1])
    axis.set_zlim(limits[2])
    fig.colorbar(surface, ax=axis, shrink=0.65, pad=0.1)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    initial = np.array([0.10, 1.70, 1.60, 1.50, 0.20, -0.20, 0.10])
    truth = np.array([0.30, 2.50, 1.20, 2.00, 0.35, 0.40, -0.60])
    final = np.array([0.30, 2.50, 1.20, 2.00, 0.35, 0.40, -0.60])

    x = np.linspace(-5.0, 5.0, 31)
    y = np.linspace(-5.0, 5.0, 31)
    x_grid, y_grid = np.meshgrid(x, y)

    z_initial = gaussian_model(initial, x_grid, y_grid)
    z_truth = gaussian_model(truth, x_grid, y_grid)
    z_final = gaussian_model(final, x_grid, y_grid)
    z_error = z_final - z_truth

    rmse = np.sqrt(np.mean(z_error * z_error))
    max_error = np.max(np.abs(z_error))

    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    z_min = min(z_initial.min(), z_truth.min(), z_final.min())
    z_max = max(z_initial.max(), z_truth.max(), z_final.max())
    z_padding = 0.05 * (z_max - z_min)
    limits = ((-5.0, 5.0), (-5.0, 5.0), (z_min - z_padding, z_max + z_padding))

    save_surface(
        os.path.join(output_dir, "gaussian_truth_surface.png"),
        "Truth Gaussian Surface",
        x_grid,
        y_grid,
        z_truth,
        limits,
    )
    save_surface(
        os.path.join(output_dir, "gaussian_initial_surface.png"),
        "Initial Gaussian Surface",
        x_grid,
        y_grid,
        z_initial,
        limits,
    )
    save_surface(
        os.path.join(output_dir, "gaussian_final_surface.png"),
        "Fitted Gaussian Surface",
        x_grid,
        y_grid,
        z_final,
        limits,
    )

    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(
        z_error,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
    )
    axis.set_title(
        f"Fitted - Truth Error\nRMSE={rmse:.6g}, Max Error={max_error:.6g}"
    )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    fig.colorbar(image, ax=axis, label="Error")
    fig.savefig(
        os.path.join(output_dir, "gaussian_error_map.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    names = ["c0", "c1", "c2", "c3", "c4", "c5", "c6"]
    positions = np.arange(len(names))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - width, initial, width, label="Initial")
    axis.bar(positions, truth, width, label="Truth")
    axis.bar(positions + width, final, width, label="Final")
    axis.set_xticks(positions, names)
    axis.set_ylabel("Parameter value")
    axis.set_title("Gaussian Parameter Comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(
        os.path.join(output_dir, "gaussian_parameter_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    center_row = int(np.argmin(np.abs(y - truth[6])))
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x, z_initial[center_row], label="Initial")
    axis.plot(x, z_truth[center_row], label="Truth")
    axis.plot(x, z_final[center_row], "--", label="Final")
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_title("Gaussian Center Slice")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.savefig(
        os.path.join(output_dir, "gaussian_center_slice.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Python visualization check:")
    print(f"RMSE={rmse:.12g}")
    print(f"MaxError={max_error:.12g}")


if __name__ == "__main__":
    main()
