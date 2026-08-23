#!/usr/bin/env python3
"""Independent SciPy reference solves for the Gaussian bounds test cases.

The C tests solve these problems with Augmented Lagrangian constraints and
CMinpack.  This module deliberately uses SciPy's native bound-constrained TRF
solver so it can serve as an independent numerical reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Mapping

import numpy as np
import scipy
from scipy.optimize import least_squares


PARAMETER_COUNT = 7
GRID_SIZE = 31
MAX_NFEV = 2000

NORMAL_TRUTH = np.array([0.3, 2.5, 1.2, 2.0, 0.35, 0.4, -0.6])
NORMAL_INITIAL = np.array([0.1, 1.7, 1.6, 1.5, 0.2, -0.2, 0.1])
# Still deliberately far from the truth, but avoids a known degenerate local
# representation reached by the more extreme illustrative guess in the task.
POOR_INITIAL = np.array([0.6, 1.2, 2.0, 1.2, 0.6, -0.7, 0.3])

NORMAL_LOWER = np.array(
    [-np.inf, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, -np.inf, -np.inf]
)
NORMAL_UPPER = np.full(PARAMETER_COUNT, np.inf)


@dataclass(frozen=True)
class GaussianCase:
    """Complete definition of one Gaussian fitting problem."""

    name: str
    truth: np.ndarray
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    add_noise: bool = False
    grid_nx: int = GRID_SIZE
    grid_ny: int = GRID_SIZE
    x_limits: tuple[float, float] = (-5.0, 5.0)
    y_limits: tuple[float, float] = (-5.0, 5.0)
    special_condition: str = "standard_bound_constrained_gaussian"
    expected_failure: bool = False
    feasibility_mode: str = "solve"
    general_lower: tuple[tuple[int, float], ...] = ()
    general_upper: tuple[tuple[int, float], ...] = ()
    x_scale: str | None = None
    ftol: float = 1.0e-12
    xtol: float = 1.0e-12
    gtol: float = 1.0e-12
    max_nfev: int = MAX_NFEV


def _array(values: Iterable[float]) -> np.ndarray:
    """Create an owned float64 parameter vector for a case definition."""

    result = np.asarray(tuple(values), dtype=np.float64)
    if result.shape != (PARAMETER_COUNT,):
        raise ValueError(f"expected {PARAMETER_COUNT} parameters, got {result.shape}")
    return result


def _make_case(
    name: str,
    *,
    truth: Iterable[float] = NORMAL_TRUTH,
    initial: Iterable[float] = NORMAL_INITIAL,
    lower: Iterable[float] = NORMAL_LOWER,
    upper: Iterable[float] = NORMAL_UPPER,
    add_noise: bool = False,
    grid_nx: int = GRID_SIZE,
    grid_ny: int = GRID_SIZE,
    x_limits: tuple[float, float] = (-5.0, 5.0),
    y_limits: tuple[float, float] = (-5.0, 5.0),
    special_condition: str = "standard_bound_constrained_gaussian",
    expected_failure: bool = False,
    feasibility_mode: str = "solve",
    general_lower: tuple[tuple[int, float], ...] = (),
    general_upper: tuple[tuple[int, float], ...] = (),
    x_scale: str | None = None,
    ftol: float = 1.0e-12,
    xtol: float = 1.0e-12,
    gtol: float = 1.0e-12,
    max_nfev: int = MAX_NFEV,
) -> GaussianCase:
    if grid_nx < 2 or grid_ny < 2:
        raise ValueError("Gaussian grids need at least two points per axis")
    if x_limits[0] >= x_limits[1] or y_limits[0] >= y_limits[1]:
        raise ValueError("Gaussian grid limits must be strictly increasing")
    return GaussianCase(
        name=name,
        truth=_array(truth),
        initial=_array(initial),
        lower=_array(lower),
        upper=_array(upper),
        add_noise=add_noise,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        x_limits=x_limits,
        y_limits=y_limits,
        special_condition=special_condition,
        expected_failure=expected_failure,
        feasibility_mode=feasibility_mode,
        general_lower=general_lower,
        general_upper=general_upper,
        x_scale=x_scale,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
    )


_infeasible_initial = NORMAL_INITIAL.copy()
_infeasible_initial[2] = 0.2
_infeasible_lower = NORMAL_LOWER.copy()
_infeasible_lower[2] = 0.5

_active_lower_truth = NORMAL_TRUTH.copy()
_active_lower_truth[2] = 0.6
_active_lower_bounds = NORMAL_LOWER.copy()
_active_lower_bounds[2] = 0.9

_active_upper_bounds = NORMAL_UPPER.copy()
_active_upper_bounds[1] = 2.0

_box_initial = NORMAL_INITIAL.copy()
_box_initial[2] = 1.1
_box_lower = NORMAL_LOWER.copy()
_box_lower[2] = 1.0
_box_upper = NORMAL_UPPER.copy()
_box_upper[2] = 1.4

_bad_scaling_truth = np.array(
    [1.0e3, 1.0e6, 1.0e-2, 2.0, 0.30, 0.01, -0.50]
)
_bad_scaling_initial = np.array(
    [900.0, 9.0e5, 1.0e-2, 2.0, 0.30, 0.01, -0.50]
)

_near_lower_truth = NORMAL_TRUTH.copy()
_near_lower_truth[2] = 0.90000001
_near_lower_bounds = NORMAL_LOWER.copy()
_near_lower_bounds[2] = 0.9

_two_active_truth = NORMAL_TRUTH.copy()
_two_active_truth[2] = 0.6
_two_active_lower = NORMAL_LOWER.copy()
_two_active_lower[2] = 0.9
_two_active_upper = NORMAL_UPPER.copy()
_two_active_upper[1] = 2.0

_infeasible_box_lower = NORMAL_LOWER.copy()
_infeasible_box_lower[2] = 2.0
_infeasible_box_upper = NORMAL_UPPER.copy()
_infeasible_box_upper[2] = 1.0

_rank_deficient_truth = NORMAL_TRUTH.copy()
_rank_deficient_truth[2:4] = 1.5
_rank_deficient_initial = np.array([0.1, 1.7, 1.2, 1.8, 1.1, -0.2, 0.1])

_narrow_truth = np.array([0.1, 2.0, 1.0e-3, 5.0e-3, 0.3, 0.0, 0.0])
_narrow_initial = np.array([0.05, 1.5, 2.0e-3, 8.0e-3, 0.2, 1.0e-3, -2.0e-3])
_narrow_lower = NORMAL_LOWER.copy()
_narrow_lower[2] = 1.0e-5
_narrow_lower[3] = 1.0e-5


# Insertion order is the reporting order used by the C test and cross-validator.
CASE_SPECS: Mapping[str, GaussianCase] = {
    "gaussian_normal": _make_case("gaussian_normal"),
    "gaussian_infeasible_initial": _make_case(
        "gaussian_infeasible_initial",
        initial=_infeasible_initial,
        lower=_infeasible_lower,
    ),
    "gaussian_active_lower": _make_case(
        "gaussian_active_lower",
        truth=_active_lower_truth,
        lower=_active_lower_bounds,
    ),
    "gaussian_active_upper": _make_case(
        "gaussian_active_upper",
        upper=_active_upper_bounds,
    ),
    "gaussian_box_bound": _make_case(
        "gaussian_box_bound",
        initial=_box_initial,
        lower=_box_lower,
        upper=_box_upper,
    ),
    "gaussian_noisy": _make_case("gaussian_noisy", add_noise=True),
    "gaussian_poor_initial": _make_case(
        "gaussian_poor_initial", initial=POOR_INITIAL
    ),
    "gaussian_bad_scaling": _make_case(
        "gaussian_bad_scaling",
        truth=_bad_scaling_truth,
        initial=_bad_scaling_initial,
        grid_nx=101,
        grid_ny=51,
        x_limits=(-0.05, 0.07),
        y_limits=(-4.0, 3.0),
        special_condition="severe_parameter_scaling",
        x_scale="jac",
    ),
    "gaussian_near_lower_bound": _make_case(
        "gaussian_near_lower_bound",
        truth=_near_lower_truth,
        lower=_near_lower_bounds,
        special_condition="mathematically_interior_1e-8_above_lower_bound",
        gtol=1.0e-14,
    ),
    "gaussian_two_active_bounds": _make_case(
        "gaussian_two_active_bounds",
        truth=_two_active_truth,
        lower=_two_active_lower,
        upper=_two_active_upper,
        special_condition="two_simultaneously_active_bounds",
    ),
    "gaussian_infeasible_box": _make_case(
        "gaussian_infeasible_box",
        lower=_infeasible_box_lower,
        upper=_infeasible_box_upper,
        special_condition="contradictory_native_box_bounds",
        expected_failure=True,
        feasibility_mode="scipy_rejected_bounds",
    ),
    "gaussian_infeasible_constraints": _make_case(
        "gaussian_infeasible_constraints",
        special_condition="disjoint_general_inequality_intervals",
        expected_failure=True,
        feasibility_mode="independent_interval_check",
        general_lower=((1, 3.0),),
        general_upper=((1, 2.0),),
    ),
    "gaussian_large_grid": _make_case(
        "gaussian_large_grid",
        grid_nx=101,
        grid_ny=101,
        special_condition="10201_residual_large_grid",
    ),
    "gaussian_rank_deficient": _make_case(
        "gaussian_rank_deficient",
        truth=_rank_deficient_truth,
        initial=_rank_deficient_initial,
        special_condition="circular_gaussian_theta_non_identifiable",
    ),
    "gaussian_narrow_width": _make_case(
        "gaussian_narrow_width",
        truth=_narrow_truth,
        initial=_narrow_initial,
        lower=_narrow_lower,
        grid_nx=101,
        grid_ny=101,
        x_limits=(-0.04, 0.04),
        y_limits=(-0.12, 0.12),
        special_condition="narrow_width_and_exponential_underflow",
        x_scale="jac",
    ),
}

# A compact iterable is convenient for callers that only need the canonical names.
CASES = tuple(CASE_SPECS)

_CASE_ALIASES = {
    "normal": "gaussian_normal",
    "infeasible_initial": "gaussian_infeasible_initial",
    "active_lower": "gaussian_active_lower",
    "active_upper": "gaussian_active_upper",
    "box": "gaussian_box_bound",
    "box_bound": "gaussian_box_bound",
    "noisy": "gaussian_noisy",
    "poor_initial": "gaussian_poor_initial",
    "bad_scaling": "gaussian_bad_scaling",
    "near_lower_bound": "gaussian_near_lower_bound",
    "two_active_bounds": "gaussian_two_active_bounds",
    "infeasible_box": "gaussian_infeasible_box",
    "infeasible_constraints": "gaussian_infeasible_constraints",
    "large_grid": "gaussian_large_grid",
    "rank_deficient": "gaussian_rank_deficient",
    "narrow_width": "gaussian_narrow_width",
}


def gaussian_model(
    params: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Evaluate the same seven-parameter rotated Gaussian used by the C test."""

    c = np.asarray(params, dtype=np.float64)
    return c[0] + c[1] * gaussian_exponential(c, x, y)


def gaussian_exponential(
    params: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Return exp(-q/2), including legitimate IEEE underflow zeros."""

    c = np.asarray(params, dtype=np.float64)
    dx = np.asarray(x, dtype=np.float64) - c[5]
    dy = np.asarray(y, dtype=np.float64) - c[6]
    ct = np.cos(c[4])
    st = np.sin(c[4])
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    q = u * u / (c[2] * c[2]) + v * v / (c[3] * c[3])
    return np.exp(-0.5 * q)


def gaussian_residual(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Return model minus observations, matching the C residual sign."""

    return gaussian_model(params, x, y) - observed


def gaussian_jacobian(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic row-major (m, 7) Jacobian used by the C implementation."""

    del observed  # The model Jacobian is independent of the observations.
    c = np.asarray(params, dtype=np.float64)
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)

    amplitude = c[1]
    sx = c[2]
    sy = c[3]
    ct = np.cos(c[4])
    st = np.sin(c[4])
    inv_sx2 = 1.0 / (sx * sx)
    inv_sy2 = 1.0 / (sy * sy)

    dx = x_values - c[5]
    dy = y_values - c[6]
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    q = u * u * inv_sx2 + v * v * inv_sy2
    exponential = np.exp(-0.5 * q)
    scaled = amplitude * exponential

    jacobian = np.empty((x_values.size, PARAMETER_COUNT), dtype=np.float64)
    jacobian[:, 0] = 1.0
    jacobian[:, 1] = exponential
    jacobian[:, 2] = scaled * u * u / (sx * sx * sx)
    jacobian[:, 3] = scaled * v * v / (sy * sy * sy)
    jacobian[:, 4] = scaled * u * v * (inv_sy2 - inv_sx2)
    jacobian[:, 5] = scaled * (u * ct * inv_sx2 - v * st * inv_sy2)
    jacobian[:, 6] = scaled * (u * st * inv_sx2 + v * ct * inv_sy2)
    return jacobian


def gaussian_grid(
    size: int = GRID_SIZE,
    *,
    nx: int | None = None,
    ny: int | None = None,
    x_limits: tuple[float, float] = (-5.0, 5.0),
    y_limits: tuple[float, float] = (-5.0, 5.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return the C-compatible grid ordering (x changes fastest)."""

    nx = size if nx is None else nx
    ny = size if ny is None else ny
    x_coordinates = np.linspace(*x_limits, nx, dtype=np.float64)
    y_coordinates = np.linspace(*y_limits, ny, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    return x_grid.ravel(), y_grid.ravel()


def case_grid(case: str | GaussianCase) -> tuple[np.ndarray, np.ndarray]:
    """Build the deterministic grid declared by a case specification."""

    spec = get_case(case) if isinstance(case, str) else case
    return gaussian_grid(
        nx=spec.grid_nx,
        ny=spec.grid_ny,
        x_limits=spec.x_limits,
        y_limits=spec.y_limits,
    )


def deterministic_noise(count: int) -> np.ndarray:
    """Return the reproducible non-random perturbation used in the noisy case."""

    index = np.arange(count, dtype=np.float64)
    return 0.01 * np.sin(0.37 * index) * np.cos(0.11 * index)


def project_initial_to_bounds(
    initial: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Make an initial vector strictly feasible for SciPy's native bounds.

    The raw initial vector remains part of the result.  In particular, the C
    infeasible-initial case starts with sigma_x=0.2 while its lower bound is
    0.5; SciPy receives 0.50000001 because ``least_squares`` requires a
    feasible x0.
    """

    projected = np.asarray(initial, dtype=np.float64).copy()
    lower_values = np.asarray(lower, dtype=np.float64)
    upper_values = np.asarray(upper, dtype=np.float64)

    if np.any(lower_values >= upper_values):
        raise ValueError("each lower bound must be strictly below its upper bound")

    for index, (lo, hi) in enumerate(zip(lower_values, upper_values)):
        scale = max(
            1.0,
            abs(lo) if np.isfinite(lo) else 0.0,
            abs(hi) if np.isfinite(hi) else 0.0,
        )
        margin = 1.0e-8 * scale
        if np.isfinite(lo) and np.isfinite(hi):
            margin = min(margin, 0.25 * (hi - lo))
        if np.isfinite(lo) and projected[index] <= lo:
            projected[index] = np.nextafter(lo + margin, np.inf)
        if np.isfinite(hi) and projected[index] >= hi:
            projected[index] = np.nextafter(hi - margin, -np.inf)

    changed = not np.array_equal(projected, np.asarray(initial, dtype=np.float64))
    return projected, changed


def constraint_violation(
    params: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Return max positive violation across all finite lower/upper bounds."""

    values = np.asarray(params, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return float("inf")
    lower_violation = np.where(np.isfinite(lower), lower - values, -np.inf)
    upper_violation = np.where(np.isfinite(upper), values - upper, -np.inf)
    return float(max(0.0, np.max(lower_violation), np.max(upper_violation)))


def active_bounds(
    params: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float = 1.0e-7,
) -> list[dict[str, object]]:
    """Describe native bounds active at the SciPy solution."""

    values = np.asarray(params, dtype=np.float64)
    active: list[dict[str, object]] = []
    for index, (value, lo, hi) in enumerate(zip(values, lower, upper)):
        if np.isfinite(lo) and value - lo <= tolerance:
            active.append(
                {
                    "index": index,
                    "type": "lower",
                    "value": float(value),
                    "target": float(lo),
                    "slack": float(value - lo),
                }
            )
        if np.isfinite(hi) and hi - value <= tolerance:
            active.append(
                {
                    "index": index,
                    "type": "upper",
                    "value": float(value),
                    "target": float(hi),
                    "slack": float(hi - value),
                }
            )
    return active


def get_case(name: str) -> GaussianCase:
    """Look up a canonical case, accepting short aliases for convenience."""

    canonical_name = _CASE_ALIASES.get(name, name)
    try:
        return CASE_SPECS[canonical_name]
    except KeyError as exc:
        choices = ", ".join(CASE_SPECS)
        raise KeyError(f"unknown Gaussian case {name!r}; expected one of {choices}") from exc


def general_constraint_violation(spec: GaussianCase, params: np.ndarray) -> float:
    """Evaluate separately registered lower/upper inequalities."""

    values = np.asarray(params, dtype=np.float64)
    violations = [0.0]
    violations.extend(bound - values[index] for index, bound in spec.general_lower)
    violations.extend(values[index] - bound for index, bound in spec.general_upper)
    return float(max(violations))


def parameter_scale_ratio(params: np.ndarray) -> float:
    """Return max magnitude divided by the smallest nonzero magnitude."""

    magnitudes = np.abs(np.asarray(params, dtype=np.float64))
    nonzero = magnitudes[(magnitudes > 0.0) & np.isfinite(magnitudes)]
    if nonzero.size == 0:
        return 1.0
    return float(np.max(nonzero) / np.min(nonzero))


def normalize_theta(theta: float) -> float:
    """Normalize the pi-periodic Gaussian orientation to [-pi/2, pi/2)."""

    return float((theta + 0.5 * np.pi) % np.pi - 0.5 * np.pi)


def _problem_arrays(
    spec: GaussianCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_values, y_values = case_grid(spec)
    truth_prediction = gaussian_model(spec.truth, x_values, y_values)
    observed = truth_prediction.copy()
    if spec.add_noise:
        observed += deterministic_noise(observed.size)
    return x_values, y_values, truth_prediction, observed


def _make_result(
    spec: GaussianCase,
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
    truth_prediction: np.ndarray,
    observed: np.ndarray,
    solver_initial: np.ndarray,
    initial_projected: bool,
    final: np.ndarray,
    status: int,
    success: bool,
    message: str,
    nfev: int,
    njev: int | None,
    cost: float,
    optimality: float | None,
    runtime_seconds: float,
    solver_executed: bool,
    feasibility_status: str,
) -> dict[str, object]:
    """Build the stable result schema shared by solved and infeasible cases."""

    final_values = np.asarray(final, dtype=np.float64).copy()
    prediction = gaussian_model(final_values, x_values, y_values)
    residual = prediction - observed
    jacobian = gaussian_jacobian(final_values, x_values, y_values)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    max_prediction_error = float(np.max(np.abs(prediction - truth_prediction)))
    native_violation = constraint_violation(
        final_values, spec.lower, spec.upper
    )
    general_violation = general_constraint_violation(spec, final_values)
    violation = max(native_violation, general_violation)
    active = (
        active_bounds(final_values, spec.lower, spec.upper)
        if np.all(spec.lower < spec.upper)
        else []
    )
    primary_active = active[0] if active else None
    signal_scale = float(max(1.0, np.max(np.abs(observed))))
    jacobian_norm = float(np.linalg.norm(jacobian))
    theta_jacobian_norm = float(np.linalg.norm(jacobian[:, 4]))
    finite_check = bool(
        np.all(np.isfinite(final_values))
        and np.all(np.isfinite(truth_prediction))
        and np.all(np.isfinite(observed))
        and np.all(np.isfinite(prediction))
        and np.all(np.isfinite(residual))
        and np.all(np.isfinite(jacobian))
        and np.isfinite(rmse)
        and np.isfinite(max_prediction_error)
        and np.isfinite(cost)
        and np.isfinite(violation)
    )

    result: dict[str, object] = {
        "name": spec.name,
        "status": int(status),
        "success": bool(success),
        "message": message,
        "nfev": int(nfev),
        "njev": int(njev) if njev is not None else None,
        "cost": float(cost),
        "optimality": float(optimality) if optimality is not None else None,
        "initial": spec.initial.copy(),
        "solver_initial": np.asarray(solver_initial, dtype=np.float64).copy(),
        "initial_projected": bool(initial_projected),
        "truth": spec.truth.copy(),
        "final": final_values,
        "rmse": rmse,
        "max_prediction_error": max_prediction_error,
        "constraint_violation": violation,
        "native_bound_violation": native_violation,
        "general_constraint_violation": general_violation,
        "active_bounds": active,
        "active_bound_count": len(active),
        "active_bound_index": primary_active["index"] if primary_active else -1,
        "active_bound_type": primary_active["type"] if primary_active else "none",
        "active_bound_value": primary_active["value"] if primary_active else None,
        "active_bound_target": primary_active["target"] if primary_active else None,
        "active_bound_slack": primary_active["slack"] if primary_active else None,
        "lower": spec.lower.copy(),
        "upper": spec.upper.copy(),
        "general_lower": tuple(spec.general_lower),
        "general_upper": tuple(spec.general_upper),
        "x": x_values,
        "y": y_values,
        "observed": observed,
        "truth_prediction": truth_prediction,
        "prediction": prediction,
        "predictions": prediction,
        "residual": residual,
        "m": int(observed.size),
        "n": PARAMETER_COUNT,
        "grid_shape": (spec.grid_ny, spec.grid_nx),
        "grid_x_limits": spec.x_limits,
        "grid_y_limits": spec.y_limits,
        "runtime_seconds": float(runtime_seconds),
        "special_condition": spec.special_condition,
        "expected_failure": spec.expected_failure,
        "feasibility_status": feasibility_status,
        "solver_executed": solver_executed,
        "solver_method": "trf" if solver_executed else "not_run",
        "solver_x_scale": spec.x_scale if spec.x_scale is not None else "default",
        "solver_ftol": spec.ftol,
        "solver_xtol": spec.xtol,
        "solver_gtol": spec.gtol,
        "solver_max_nfev": spec.max_nfev,
        "finite_check": finite_check,
        "signal_scale": signal_scale,
        "relative_fit_rmse": rmse / signal_scale,
        "relative_max_prediction_error": max_prediction_error / signal_scale,
        "parameter_scale_ratio": parameter_scale_ratio(final_values),
        "theta_jacobian_column_norm": theta_jacobian_norm,
        "theta_jacobian_relative_norm": (
            theta_jacobian_norm / jacobian_norm if jacobian_norm > 0.0 else 0.0
        ),
        "normalized_theta": normalize_theta(final_values[4]),
        "theta_periodic_difference": normalize_theta(
            final_values[4] - spec.truth[4]
        ),
        "width_difference": float(abs(final_values[2] - final_values[3])),
        "truth_underflow_zero_count": int(
            np.count_nonzero(gaussian_exponential(spec.truth, x_values, y_values) == 0.0)
        ),
        "final_underflow_zero_count": int(
            np.count_nonzero(gaussian_exponential(final_values, x_values, y_values) == 0.0)
        ),
        "underflow_zero_count": int(
            np.count_nonzero(gaussian_exponential(final_values, x_values, y_values) == 0.0)
        ),
        "scipy_version": scipy.__version__,
    }
    return result


def _solve_infeasible_box(
    spec: GaussianCase,
    x_values: np.ndarray,
    y_values: np.ndarray,
    truth_prediction: np.ndarray,
    observed: np.ndarray,
) -> dict[str, object]:
    """Ask SciPy to validate contradictory native bounds and capture rejection."""

    started = perf_counter()
    exception: ValueError | None = None
    unexpected_result = None
    try:
        unexpected_result = least_squares(
            gaussian_residual,
            spec.initial,
            jac=gaussian_jacobian,
            bounds=(spec.lower, spec.upper),
            method="trf",
            ftol=spec.ftol,
            xtol=spec.xtol,
            gtol=spec.gtol,
            max_nfev=spec.max_nfev,
            args=(x_values, y_values, observed),
        )
    except ValueError as caught:
        exception = caught
    runtime_seconds = perf_counter() - started
    rejected = exception is not None
    final = (
        spec.initial.copy()
        if unexpected_result is None
        else np.asarray(unexpected_result.x, dtype=np.float64)
    )
    residual = gaussian_residual(final, x_values, y_values, observed)
    result = _make_result(
        spec,
        x_values=x_values,
        y_values=y_values,
        truth_prediction=truth_prediction,
        observed=observed,
        solver_initial=spec.initial,
        initial_projected=False,
        final=final,
        status=-1 if rejected else int(unexpected_result.status),
        success=False if rejected else bool(unexpected_result.success),
        message=str(exception) if rejected else str(unexpected_result.message),
        nfev=0 if rejected else int(unexpected_result.nfev),
        njev=0 if rejected else unexpected_result.njev,
        cost=float(0.5 * np.dot(residual, residual)) if rejected else float(unexpected_result.cost),
        optimality=None if rejected else float(unexpected_result.optimality),
        runtime_seconds=runtime_seconds,
        solver_executed=not rejected,
        feasibility_status="rejected_bounds" if rejected else "unexpectedly_accepted_bounds",
    )
    result["scipy_exception_type"] = type(exception).__name__ if exception else None
    result["scipy_exception_message"] = str(exception) if exception else None
    result["pass"] = rejected and result["finite_check"]
    result["test_result"] = "PASS" if result["pass"] else "FAIL"
    return result


def _solve_infeasible_constraints(
    spec: GaussianCase,
    x_values: np.ndarray,
    y_values: np.ndarray,
    truth_prediction: np.ndarray,
    observed: np.ndarray,
) -> dict[str, object]:
    """Independently prove that registered inequality intervals are disjoint."""

    started = perf_counter()
    required_lower = np.full(PARAMETER_COUNT, -np.inf)
    required_upper = np.full(PARAMETER_COUNT, np.inf)
    for index, value in spec.general_lower:
        required_lower[index] = max(required_lower[index], value)
    for index, value in spec.general_upper:
        required_upper[index] = min(required_upper[index], value)
    contradictory = bool(np.any(required_lower > required_upper))
    runtime_seconds = perf_counter() - started
    residual = gaussian_residual(spec.initial, x_values, y_values, observed)
    result = _make_result(
        spec,
        x_values=x_values,
        y_values=y_values,
        truth_prediction=truth_prediction,
        observed=observed,
        solver_initial=spec.initial,
        initial_projected=False,
        final=spec.initial,
        status=-2,
        success=False,
        message=(
            "independent feasibility check found disjoint intervals"
            if contradictory
            else "independent feasibility check found no contradiction"
        ),
        nfev=0,
        njev=0,
        cost=float(0.5 * np.dot(residual, residual)),
        optimality=None,
        runtime_seconds=runtime_seconds,
        solver_executed=False,
        feasibility_status=(
            "infeasible_by_interval_check" if contradictory else "feasible_intervals"
        ),
    )
    result["feasibility_required_lower"] = required_lower
    result["feasibility_required_upper"] = required_upper
    result["feasibility_interval_width_c1"] = float(
        required_upper[1] - required_lower[1]
    )
    result["pass"] = contradictory and result["finite_check"]
    result["test_result"] = "PASS" if result["pass"] else "FAIL"
    return result


def solve_case(case: str | GaussianCase) -> dict[str, object]:
    """Solve one case and return a cross-validator-friendly result mapping."""

    spec = get_case(case) if isinstance(case, str) else case
    x_values, y_values, truth_prediction, observed = _problem_arrays(spec)

    if spec.feasibility_mode == "scipy_rejected_bounds":
        return _solve_infeasible_box(
            spec, x_values, y_values, truth_prediction, observed
        )
    if spec.feasibility_mode == "independent_interval_check":
        return _solve_infeasible_constraints(
            spec, x_values, y_values, truth_prediction, observed
        )

    solver_initial, initial_projected = project_initial_to_bounds(
        spec.initial, spec.lower, spec.upper
    )
    options: dict[str, object] = {}
    if spec.x_scale is not None:
        options["x_scale"] = spec.x_scale
    started = perf_counter()
    scipy_result = least_squares(
        gaussian_residual,
        solver_initial,
        jac=gaussian_jacobian,
        bounds=(spec.lower, spec.upper),
        method="trf",
        ftol=spec.ftol,
        xtol=spec.xtol,
        gtol=spec.gtol,
        max_nfev=spec.max_nfev,
        args=(x_values, y_values, observed),
        **options,
    )
    runtime_seconds = perf_counter() - started

    final = np.asarray(scipy_result.x, dtype=np.float64).copy()
    result = _make_result(
        spec,
        x_values=x_values,
        y_values=y_values,
        truth_prediction=truth_prediction,
        observed=observed,
        solver_initial=solver_initial,
        initial_projected=initial_projected,
        final=final,
        status=int(scipy_result.status),
        success=bool(scipy_result.success),
        message=str(scipy_result.message),
        nfev=int(scipy_result.nfev),
        njev=int(scipy_result.njev) if scipy_result.njev is not None else None,
        cost=float(scipy_result.cost),
        optimality=float(scipy_result.optimality),
        runtime_seconds=runtime_seconds,
        solver_executed=True,
        feasibility_status="feasible_solution",
    )

    if spec.name == "gaussian_near_lower_bound":
        slack = float(final[2] - spec.lower[2])
        classification_margin = float(
            16.0
            * np.finfo(np.float64).eps
            * max(1.0, abs(final[2]), abs(spec.lower[2]))
        )

        def active_at(tolerance: float) -> bool:
            return slack <= tolerance + classification_margin

        variants: dict[str, dict[str, object]] = {}
        for label, tolerance in (("standard", 1.0e-8), ("strict", 1.0e-10)):
            variants[label] = {
                "constraint_tol": tolerance,
                "final": final.copy(),
                "bound_target": float(spec.lower[2]),
                "bound_slack": slack,
                "constraint_violation": result["constraint_violation"],
                "active_by_tolerance": active_at(tolerance),
                "mathematically_interior": bool(final[2] > spec.lower[2]),
            }
        result.update(
            {
                "constraint_tol": 1.0e-10,
                "bound_target": float(spec.lower[2]),
                "bound_slack": slack,
                "truth_bound_slack": float(spec.truth[2] - spec.lower[2]),
                "mathematically_interior": bool(spec.truth[2] > spec.lower[2]),
                "active_by_tolerance": active_at(1.0e-10),
                "active_by_tolerance_1e-8": active_at(1.0e-8),
                "active_by_tolerance_1e-10": active_at(1.0e-10),
                "active_classification_margin": classification_margin,
                "near_bound_variants": variants,
                "strict_final": final.copy(),
                "strict_bound_slack": slack,
                "strict_constraint_violation": result["constraint_violation"],
            }
        )

    passed = bool(
        result["success"]
        and result["finite_check"]
        and result["constraint_violation"] <= 1.0e-8
    )
    if spec.name == "gaussian_two_active_bounds":
        passed = passed and result["active_bound_count"] == 2
    if spec.name in {
        "gaussian_bad_scaling",
        "gaussian_large_grid",
        "gaussian_rank_deficient",
        "gaussian_narrow_width",
    }:
        passed = passed and result["relative_max_prediction_error"] < 1.0e-6
    result["pass"] = passed
    result["test_result"] = "PASS" if passed else "FAIL"
    return result


def _format_float(value: object) -> str:
    if value is None:
        return "none"
    return format(float(value), ".17g")


def _format_vector(values: np.ndarray) -> str:
    return "[" + ",".join(_format_float(value) for value in values) + "]"


def _format_active_bounds(bounds: object) -> str:
    entries = []
    for bound in bounds:
        entries.append(
            f"{bound['index']}:{bound['type']}:"
            f"{_format_float(bound['value'])}:"
            f"{_format_float(bound['target'])}:"
            f"{_format_float(bound['slack'])}"
        )
    return "[" + ";".join(entries) + "]"


def print_case_result(result: Mapping[str, object]) -> None:
    """Print the stable line-oriented PYTHON CASE protocol."""

    print(f"PYTHON CASE {result['name']}")
    print(f"m={result['m']}")
    print(f"n={result['n']}")
    print(f"status={result['status']}")
    print(f"success={str(result['success']).lower()}")
    print(f"pass={str(result['pass']).lower()}")
    print(f"expected_failure={str(result['expected_failure']).lower()}")
    print(f"feasibility_status={result['feasibility_status']}")
    print(f"solver_executed={str(result['solver_executed']).lower()}")
    print(f"special_condition={result['special_condition']}")
    print(f"nfev={result['nfev']}")
    print(f"njev={result['njev']}")
    print(f"cost={_format_float(result['cost'])}")
    print(f"optimality={_format_float(result['optimality'])}")
    print(f"initial={_format_vector(result['initial'])}")
    print(f"solver_initial={_format_vector(result['solver_initial'])}")
    print(f"initial_projected={str(result['initial_projected']).lower()}")
    print(f"truth={_format_vector(result['truth'])}")
    print(f"final={_format_vector(result['final'])}")
    print(f"rmse={_format_float(result['rmse'])}")
    print(
        "max_prediction_error="
        f"{_format_float(result['max_prediction_error'])}"
    )
    print(
        "constraint_violation="
        f"{_format_float(result['constraint_violation'])}"
    )
    print(f"finite_check={str(result['finite_check']).lower()}")
    print(f"active_bound_count={result['active_bound_count']}")
    print(f"active_bounds={_format_active_bounds(result['active_bounds'])}")
    print(f"signal_scale={_format_float(result['signal_scale'])}")
    print(f"relative_fit_rmse={_format_float(result['relative_fit_rmse'])}")
    print(
        "relative_max_prediction_error="
        f"{_format_float(result['relative_max_prediction_error'])}"
    )
    print(f"parameter_scale_ratio={_format_float(result['parameter_scale_ratio'])}")
    print(f"runtime_seconds={_format_float(result['runtime_seconds'])}")
    print(
        "theta_jacobian_column_norm="
        f"{_format_float(result['theta_jacobian_column_norm'])}"
    )
    print(
        "theta_jacobian_relative_norm="
        f"{_format_float(result['theta_jacobian_relative_norm'])}"
    )
    print(f"normalized_theta={_format_float(result['normalized_theta'])}")
    print(f"width_difference={_format_float(result['width_difference'])}")
    print(f"underflow_zero_count={result['underflow_zero_count']}")
    if result["name"] == "gaussian_near_lower_bound":
        print(f"bound_target={_format_float(result['bound_target'])}")
        print(f"bound_slack={_format_float(result['bound_slack'])}")
        print(
            "mathematically_interior="
            f"{str(result['mathematically_interior']).lower()}"
        )
        print(
            "active_by_tolerance_1e-8="
            f"{str(result['active_by_tolerance_1e-8']).lower()}"
        )
        print(
            "active_by_tolerance_1e-10="
            f"{str(result['active_by_tolerance_1e-10']).lower()}"
        )
    if result.get("scipy_exception_type") is not None:
        print(f"scipy_exception_type={result['scipy_exception_type']}")
        print(f"scipy_exception_message={result['scipy_exception_message']}")


def run_all_cases(print_output: bool = True) -> dict[str, dict[str, object]]:
    """Solve every reference problem in reporting order."""

    results: dict[str, dict[str, object]] = {}
    for name in CASE_SPECS:
        result = solve_case(name)
        results[name] = result
        if print_output:
            print_case_result(result)
            print()
    return results


def main() -> int:
    print(f"scipy_version={scipy.__version__}")
    results = run_all_cases(print_output=True)
    return 0 if all(result["pass"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
