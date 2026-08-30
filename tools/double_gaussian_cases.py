"""Shared Double Gaussian model and deterministic DG1-DG5 case data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PARAMETER_COUNT = 7


@dataclass(frozen=True)
class CaseSpec:
    name: str
    title: str
    slug: str
    truth: tuple[float, ...]
    initial: tuple[float, ...]
    h_min: float = -2.0
    h_max: float = 2.0
    v_min: float = -2.0
    v_max: float = 2.0
    nx: int = 41
    ny: int = 41
    mask_every: int = 0
    noise_scale: float = 0.0
    c3_bound: float = 1.0e-10
    prediction_rel_tol: float = 1.0e-6
    max_point_tol: float = 1.0e-5

    @property
    def lower(self) -> tuple[float, ...]:
        return (
            -np.inf,
            1.0e-10,
            1.0e-10,
            self.c3_bound,
            1.0e-10,
            -np.inf,
            -np.inf,
        )


@dataclass(frozen=True)
class CaseData:
    spec: CaseSpec
    h_axis: np.ndarray
    v_axis: np.ndarray
    h_grid: np.ndarray
    v_grid: np.ndarray
    h_full: np.ndarray
    v_full: np.ndarray
    observed_full: np.ndarray
    valid: np.ndarray
    h: np.ndarray
    v: np.ndarray
    observed: np.ndarray


CASE_SPECS = (
    CaseSpec(
        "DG1",
        "NORMAL",
        "normal",
        (0.20, 1.50, 1.00, 0.80, 1.10, -0.30, 0.40),
        (0.10, 1.00, 0.70, 1.10, 0.80, 0.00, 0.10),
    ),
    CaseSpec(
        "DG2",
        "SMALL MAGNITUDE",
        "small_magnitude",
        (1.0e-3, 7.5e-3, 5.0e-3, 0.80, 1.10, -0.30, 0.40),
        (0.5e-3, 5.0e-3, 3.5e-3, 1.10, 0.80, 0.00, 0.10),
        prediction_rel_tol=5.0e-6,
        max_point_tol=5.0e-5,
    ),
    CaseSpec(
        "DG3",
        "LARGE MAGNITUDE",
        "large_magnitude",
        (2.0e3, 1.5e4, 1.0e4, 0.80, 1.10, -0.30, 0.40),
        (1.0e3, 1.0e4, 0.7e4, 1.10, 0.80, 0.00, 0.10),
        prediction_rel_tol=2.0e-6,
        max_point_tol=2.0e-4,
    ),
    CaseSpec(
        "DG4",
        "COORDINATE SCALING",
        "coordinate_scaling",
        (0.20, 1.50, 1.00, 80.0, 110.0, -0.30, 0.40),
        (0.10, 1.00, 0.70, 110.0, 80.0, 0.00, 0.10),
        h_min=-0.02,
        h_max=0.02,
        v_min=-0.02,
        v_max=0.02,
        prediction_rel_tol=2.0e-6,
        max_point_tol=2.0e-5,
    ),
    CaseSpec(
        "DG5",
        "ACTIVE BOUND + MASK + NOISE",
        "active_bound_mask_noise",
        (0.20, 1.50, 1.00, 0.65, 1.10, -0.30, 0.40),
        (0.10, 1.00, 0.70, 1.10, 0.80, 0.00, 0.10),
        mask_every=7,
        noise_scale=1.0,
        c3_bound=0.90,
        prediction_rel_tol=2.0e-5,
        max_point_tol=5.0e-4,
    ),
)


def double_gaussian_model(
    params: np.ndarray | tuple[float, ...],
    h: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    c0, c1, c2, c3, c4, c5, c6 = np.asarray(params, dtype=np.float64)
    a = c3 * h + c4 * v + c5
    b = c3 * h - c4 * v + c6
    return c0 + c1 * np.exp(-(a * a)) + c2 * np.exp(-(b * b))


def generate_case_data(spec: CaseSpec) -> CaseData:
    h_axis = np.linspace(spec.h_min, spec.h_max, spec.nx, dtype=np.float64)
    v_axis = np.linspace(spec.v_min, spec.v_max, spec.ny, dtype=np.float64)
    h_grid, v_grid = np.meshgrid(h_axis, v_axis, indexing="xy")
    h_full = h_grid.ravel()
    v_full = v_grid.ravel()
    observed_full = double_gaussian_model(spec.truth, h_full, v_full)
    full_index = np.arange(observed_full.size, dtype=np.float64)
    if spec.noise_scale != 0.0:
        observed_full = observed_full + spec.noise_scale * (
            0.01 * np.sin(0.37 * full_index)
            + 0.005 * np.cos(0.11 * full_index)
        )
    valid = np.ones(observed_full.size, dtype=bool)
    if spec.mask_every:
        valid[np.arange(observed_full.size) % spec.mask_every == 0] = False
    return CaseData(
        spec,
        h_axis,
        v_axis,
        h_grid,
        v_grid,
        h_full,
        v_full,
        observed_full,
        valid,
        h_full[valid],
        v_full[valid],
        observed_full[valid],
    )


def maximum_point(params: np.ndarray | tuple[float, ...]) -> tuple[float, float]:
    values = np.asarray(params, dtype=np.float64)
    return (
        float(-(values[5] + values[6]) / (2.0 * values[3])),
        float(-(values[5] - values[6]) / (2.0 * values[4])),
    )
