#!/usr/bin/env python3
"""Visualize the five Double Gaussian Golden Reference regression cases."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "build" / "visualization"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from double_gaussian_cases import (
    CASE_SPECS,
    CaseSpec,
    double_gaussian_model,
    generate_case_data,
)
from double_gaussian_minimize_reference import solve_reference_case


OLD_SINGLE_GAUSSIAN_OUTPUTS = (
    "gaussian_truth_surface.png",
    "gaussian_initial_surface.png",
    "gaussian_final_surface.png",
    "gaussian_error_map.png",
    "gaussian_parameter_comparison.png",
    "gaussian_center_slice.png",
)


def label_surface(axis: plt.Axes, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("h")
    axis.set_ylabel("v")
    axis.set_zlabel("response")


def plot_case(spec: CaseSpec) -> Path:
    data = generate_case_data(spec)
    reference = solve_reference_case(spec)
    reference_surface = double_gaussian_model(
        reference.params, data.h_full, data.v_full
    ).reshape(spec.ny, spec.nx)
    observed_surface = data.observed_full.reshape(spec.ny, spec.nx)
    valid_grid = data.valid.reshape(spec.ny, spec.nx)
    residual_surface = np.where(
        valid_grid, observed_surface - reference_surface, np.nan
    )
    signal_scale = float(np.max(np.abs(data.observed)))

    figure = plt.figure(figsize=(17, 5.5), constrained_layout=True)
    observed_axis = figure.add_subplot(131, projection="3d")
    fit_axis = figure.add_subplot(132, projection="3d")
    residual_axis = figure.add_subplot(133)

    if spec.mask_every:
        observed_axis.scatter(
            data.h,
            data.v,
            data.observed,
            c=data.observed,
            cmap="viridis",
            s=7,
            alpha=0.75,
            label="valid observed sample",
        )
        observed_axis.legend(loc="lower left", fontsize=8)
        observed_title = "Valid masked/noisy observations"
    else:
        observed_axis.plot_surface(
            data.h_grid,
            data.v_grid,
            observed_surface,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
        )
        observed_title = "Truth / observed surface"
    label_surface(observed_axis, observed_title)

    fit_surface = fit_axis.plot_surface(
        data.h_grid,
        data.v_grid,
        reference_surface,
        cmap="plasma",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    model_max = float(
        reference.params[0] + reference.params[1] + reference.params[2]
    )
    fit_axis.scatter(
        [reference.max_h],
        [reference.max_v],
        [model_max],
        color="black",
        marker="x",
        s=55,
        label="Reference Max Point",
    )
    fit_axis.legend(loc="lower left", fontsize=8)
    label_surface(fit_axis, "SciPy Golden Reference fit")
    figure.colorbar(fit_surface, ax=fit_axis, shrink=0.58, pad=0.08)

    image = residual_axis.imshow(
        residual_surface,
        extent=(spec.h_min, spec.h_max, spec.v_min, spec.v_max),
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
    )
    residual_axis.set_title("Observed - reference residual")
    residual_axis.set_xlabel("h")
    residual_axis.set_ylabel("v")
    figure.colorbar(image, ax=residual_axis, shrink=0.82, label="residual")

    figure.suptitle(
        f"{spec.name} {spec.title} - Double Gaussian\n"
        f"signal_scale={signal_scale:.6g}, valid={data.observed.size}"
    )
    output_path = OUTPUT_DIR / f"{spec.name}_{spec.slug}.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in OLD_SINGLE_GAUSSIAN_OUTPUTS:
        (OUTPUT_DIR / filename).unlink(missing_ok=True)
    outputs = [(spec.name, plot_case(spec)) for spec in CASE_SPECS]
    print("Double Gaussian visualization:")
    for name, path in outputs:
        print(f"{name}: generated {path.name}")
    print(f"output_dir={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
