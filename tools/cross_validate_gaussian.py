#!/usr/bin/env python3
"""Run and report C AugLag versus independent SciPy Gaussian validation."""

from __future__ import annotations

import importlib.util
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
C_EXECUTABLE = ROOT / "build" / "test_auglag"
REFERENCE_SCRIPT = ROOT / "tools" / "verify_gaussian_scipy.py"
REPORT_PATH = ROOT / "build" / "test_report_gaussian.txt"
SUCCESS = 1
VIOLATION_LIMIT = 1.0e-8
ACTIVE_TOL = 1.0e-4
MISSING = object()

BASE_LOWER = np.array(
    [-np.inf, 1e-10, 1e-10, 1e-10, 1e-10, -np.inf, -np.inf]
)
BASE_UPPER = np.full(7, np.inf)


def updated_bounds(
    lower_updates: Mapping[int, float] | None = None,
    upper_updates: Mapping[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = BASE_LOWER.copy(), BASE_UPPER.copy()
    for index, value in (lower_updates or {}).items():
        lower[index] = value
    for index, value in (upper_updates or {}).items():
        upper[index] = value
    return lower, upper


def case(
    title: str,
    special: str,
    *,
    lower_updates: Mapping[int, float] | None = None,
    upper_updates: Mapping[int, float] | None = None,
    m: int = 961,
    bound_count: int = 4,
    active: tuple[tuple[int, str, float], ...] = (),
    mode: str = "absolute",
    cross_limit: float = 1e-6,
) -> dict[str, Any]:
    lower, upper = updated_bounds(lower_updates, upper_updates)
    return {
        "title": title,
        "special": special,
        "lower": lower,
        "upper": upper,
        "m": m,
        "bound_count": bound_count,
        "active": active,
        "mode": mode,
        "cross_limit": cross_limit,
    }


CASES = {
    "gaussian_normal": case("Gaussian Normal", "large_m_gt_n_inactive_bounds"),
    "gaussian_infeasible_initial": case(
        "Infeasible Initial", "infeasible_initial_recovery", lower_updates={2: 0.5}
    ),
    "gaussian_active_lower": case(
        "Active Lower Bound", "one_active_lower_bound",
        lower_updates={2: 0.9}, active=((2, "lower", 0.9),),
    ),
    "gaussian_active_upper": case(
        "Active Upper Bound", "one_active_upper_bound",
        upper_updates={1: 2.0}, bound_count=5, active=((1, "upper", 2.0),),
    ),
    "gaussian_box_bound": case(
        "Box Bound", "interior_box_optimum", lower_updates={2: 1.0},
        upper_updates={2: 1.4}, bound_count=5,
    ),
    "gaussian_noisy": case("Noisy Gaussian", "deterministic_nonzero_residual"),
    "gaussian_poor_initial": case("Poor Initial Guess", "poor_initial_guess"),
    "gaussian_bad_scaling": case(
        "Severe Parameter Scaling", "severe_parameter_scaling",
        m=5151, mode="relative",
    ),
    "gaussian_near_lower_bound": case(
        "Near Lower Bound", "mathematically_interior_near_lower_bound",
        lower_updates={2: 0.9}, mode="near",
    ),
    "gaussian_two_active_bounds": case(
        "Two Active Bounds", "two_simultaneously_active_bounds",
        lower_updates={2: 0.9}, upper_updates={1: 2.0}, bound_count=5,
        active=((1, "upper", 2.0), (2, "lower", 0.9)), cross_limit=1e-5,
    ),
    "gaussian_infeasible_box": case(
        "Contradictory Box", "contradictory_infeasible_box",
        lower_updates={2: 2.0}, upper_updates={2: 1.0}, bound_count=5,
        mode="infeasible",
    ),
    "gaussian_infeasible_constraints": case(
        "General Infeasible Constraints", "disjoint_general_constraints",
        lower_updates={1: 3.0}, upper_updates={1: 2.0}, bound_count=5,
        mode="infeasible",
    ),
    "gaussian_large_grid": case(
        "Large Grid", "10201_residual_large_scale_problem", m=10201,
    ),
    "gaussian_rank_deficient": case(
        "Rank Deficient", "theta_non_identifiable_circular_gaussian", mode="rank",
    ),
    "gaussian_narrow_width": case(
        "Narrow Gaussian", "narrow_width_exponential_underflow",
        lower_updates={2: 1e-5, 3: 1e-5}, m=10201, mode="relative",
    ),
}

C_REQUIRED = {
    "m", "n", "m_aug", "grid_nx", "grid_ny", "x_min", "x_max", "y_min",
    "y_max", "special_condition", "reference_expectation", "expected_failure",
    "constraint_tol", "status", "initial", "truth", "final", "jac_error",
    "jac_error_kind", "jac_error_limit",
    "rmse", "prediction_rmse", "relative_prediction_rmse",
    "max_prediction_error", "constraint_violation", "outer_callback_count",
    "finite_check", "rho", "rho_max", "parameter_scale_ratio",
    "theta_jacobian_norm", "underflow_zero_count", "bound_target",
    "active_bound_count", "active_bound_index", "active_bound_type",
    "active_bound_value", "active_bound_target", "active_bound_slack",
    "active_bounds", "active_bound_indices", "active_bound_kinds",
    "active_bound_values", "active_bound_targets", "active_bound_slacks",
    "initial_constraint_violation", "test_result", "pass",
}
PY_REQUIRED = {
    "name", "m", "n", "status", "success", "pass", "expected_failure",
    "feasibility_status", "solver_executed", "special_condition", "nfev", "njev",
    "cost", "optimality", "initial", "solver_initial", "truth", "final", "rmse",
    "max_prediction_error", "constraint_violation", "finite_check",
    "active_bound_count", "active_bounds", "signal_scale", "relative_fit_rmse",
    "relative_max_prediction_error", "parameter_scale_ratio", "runtime_seconds",
    "theta_jacobian_column_norm", "theta_jacobian_relative_norm",
    "normalized_theta", "width_difference", "underflow_zero_count", "x", "y",
    "observed", "prediction",
}
NEAR_C_REQUIRED = {
    "bound_slack", "tol_1e8_constraint_tol", "tol_1e8_status",
    "tol_1e8_final", "tol_1e8_constraint_violation", "tol_1e8_bound_slack",
    "strict_constraint_tol", "strict_status", "strict_final",
    "strict_constraint_violation", "strict_bound_slack", "strict_prediction_rmse",
    "strict_relative_prediction_rmse", "strict_finite_check",
    "active_by_tolerance_1e8", "active_by_tolerance_1e10",
    "mathematically_interior",
}
NEAR_PY_REQUIRED = {
    "constraint_tol", "bound_target", "bound_slack", "strict_final",
    "strict_bound_slack", "strict_constraint_violation", "near_bound_variants",
    "active_by_tolerance_1e-8", "active_by_tolerance_1e-10",
    "mathematically_interior",
}

C_HEADER = re.compile(r"^C CASE ([A-Za-z0-9_]+)$")
KEY_VALUE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def parse_value(text: str) -> Any:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        return [] if not body else [parse_value(item) for item in body.split(",")]
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
    except ValueError:
        return text


def parse_c_cases(output: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    lines, cases, errors = output.splitlines(), {}, []
    index = 0
    while index < len(lines):
        header = C_HEADER.match(lines[index].strip())
        if not header:
            index += 1
            continue
        name, fields = header.group(1), {}
        index += 1
        while index < len(lines) and not C_HEADER.match(lines[index].strip()):
            line = lines[index].strip()
            index += 1
            if not line:
                continue
            match = KEY_VALUE.match(line)
            if not match:
                errors.append(f"C CASE {name}: malformed line {line!r}")
                continue
            key, raw = match.groups()
            if key in fields:
                errors.append(f"C CASE {name}: duplicate field {key}")
                continue
            try:
                fields[key] = parse_value(raw)
            except ValueError as exc:
                errors.append(f"C CASE {name}: invalid {key}: {exc}")
        if name in cases:
            errors.append(f"duplicate C CASE {name}")
        else:
            cases[name] = fields
    return cases, errors


def run_c() -> tuple[str, int, str | None]:
    if not C_EXECUTABLE.is_file():
        return "", 127, f"missing executable: {C_EXECUTABLE}"
    try:
        completed = subprocess.run(
            [str(C_EXECUTABLE)], cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=600, check=False,
        )
        return completed.stdout, completed.returncode, None
    except (OSError, subprocess.SubprocessError) as exc:
        return "", 127, str(exc)


def run_reference() -> tuple[dict[str, Mapping[str, Any]], str, str | None]:
    try:
        spec = importlib.util.spec_from_file_location(
            "verify_gaussian_scipy", REFERENCE_SCRIPT
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {REFERENCE_SCRIPT}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        captured = StringIO()
        with redirect_stdout(captured):
            results = module.run_all_cases(print_output=True)
        if not isinstance(results, Mapping) or any(
            not isinstance(value, Mapping) for value in results.values()
        ):
            raise TypeError("run_all_cases() returned an invalid result mapping")
        return dict(results), captured.getvalue(), None
    except Exception as exc:  # Always preserve diagnostics in a FAIL report.
        return {}, "", f"{type(exc).__name__}: {exc}"


def number(data: Mapping[str, Any], key: str, issues: list[str], source: str) -> float:
    if key not in data:
        return math.nan
    try:
        value = float(data[key])
    except (TypeError, ValueError):
        issues.append(f"{source} field is not numeric: {key}={data[key]!r}")
        return math.nan
    if not math.isfinite(value):
        issues.append(f"{source} field is not finite: {key}={data[key]!r}")
        return math.nan
    return value


def vector(
    data: Mapping[str, Any], key: str, issues: list[str], source: str,
) -> np.ndarray | None:
    if key not in data:
        return None
    try:
        value = np.asarray(data[key], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        value = np.empty(0)
    if value.size != 7 or not np.all(np.isfinite(value)):
        issues.append(f"{source} {key} must be a finite 7-parameter vector")
        return None
    return value


def gaussian_model(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx, dy = x - params[5], y - params[6]
    ct, st = np.cos(params[4]), np.sin(params[4])
    u, v = ct * dx + st * dy, -st * dx + ct * dy
    q = u * u / params[2] ** 2 + v * v / params[3] ** 2
    return params[0] + params[1] * np.exp(-0.5 * q)


def check_bounds(
    source: str, params: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    issues: list[str],
) -> None:
    for index in range(7):
        if params[index] < lower[index] - VIOLATION_LIMIT:
            issues.append(f"{source} c{index} violates lower bound {lower[index]:.12g}")
        if params[index] > upper[index] + VIOLATION_LIMIT:
            issues.append(f"{source} c{index} violates upper bound {upper[index]:.12g}")


def validate_active_bounds(
    source: str, entries: Sequence[Mapping[str, Any]],
    expected: tuple[tuple[int, str, float], ...], final: np.ndarray | None,
    issues: list[str],
) -> None:
    found: dict[tuple[int, str], Mapping[str, Any]] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            issues.append(f"{source} active bound entry is not a mapping")
            continue
        try:
            found[(int(item["index"]), str(item["type"]))] = item
        except (KeyError, TypeError, ValueError):
            issues.append(f"{source} active bound entry is missing index/type")
    if len(entries) != len(expected):
        issues.append(f"{source} active_bound_count={len(entries)}, expected {len(expected)}")
    for index, kind, target in expected:
        entry = found.get((index, kind))
        if entry is None:
            issues.append(f"{source} missing active {kind} bound on c{index}")
            continue
        try:
            value = float(entry["value"])
            reported_target = float(entry["target"])
            slack = float(entry["slack"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{source} active bound c{index} has invalid fields")
            continue
        if not all(math.isfinite(item) for item in (value, reported_target, slack)):
            issues.append(f"{source} active bound c{index} is not finite")
        if abs(reported_target - target) > 1e-12 or abs(slack) >= ACTIVE_TOL:
            issues.append(f"{source} active {kind} c{index} target/slack mismatch")
        if final is not None and (
            abs(final[index] - target) >= ACTIVE_TOL
            or abs(final[index] - value) > 1e-10
        ):
            issues.append(f"{source} active bound c{index} does not match final")


def c_active_entries(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    indices = data.get("active_bound_indices", [])
    kinds = data.get("active_bound_kinds", [])
    values = data.get("active_bound_values", [])
    targets = data.get("active_bound_targets", [])
    slacks = data.get("active_bound_slacks", [])
    if not all(isinstance(item, Sequence) and not isinstance(item, str) for item in (
        indices, kinds, values, targets, slacks
    )) or len({len(indices), len(kinds), len(values), len(targets), len(slacks)}) != 1:
        return []
    return [
        {
            "index": indices[i],
            "type": "lower" if int(kinds[i]) == -1 else "upper",
            "value": values[i], "target": targets[i], "slack": slacks[i],
        }
        for i in range(len(indices))
    ]


def compare_infeasible(
    name: str, c_data: Mapping[str, Any], py_data: Mapping[str, Any],
    result: dict[str, Any],
) -> None:
    issues = result["issues"]
    status = number(c_data, "status", issues, "C")
    violation = number(c_data, "constraint_violation", issues, "C")
    tolerance = number(c_data, "constraint_tol", issues, "C")
    rho = number(c_data, "rho", issues, "C")
    rho_max = number(c_data, "rho_max", issues, "C")
    vector(c_data, "final", issues, "C")
    if math.isfinite(status) and int(status) == SUCCESS:
        issues.append("C incorrectly returned AUGLAG_SUCCESS for an infeasible problem")
    if math.isfinite(violation) and math.isfinite(tolerance) and violation <= tolerance:
        issues.append("C infeasible problem did not retain violation above tolerance")
    if math.isfinite(rho) and math.isfinite(rho_max) and not (0 <= rho <= rho_max):
        issues.append(f"C rho={rho:.12g} exceeds rho_max={rho_max:.12g}")
    if int(c_data.get("finite_check", 0)) != 1:
        issues.append("C infeasible final state is not finite")
    if int(c_data.get("expected_failure", 0)) != 1:
        issues.append("C expected_failure is not set")
    expected_status = (
        "rejected_bounds" if name == "gaussian_infeasible_box"
        else "infeasible_by_interval_check"
    )
    if py_data.get("success") is not False or py_data.get("solver_executed") is not False:
        issues.append("Python infeasible reference unexpectedly executed/succeeded")
    if py_data.get("feasibility_status") != expected_status:
        issues.append(
            f"Python feasibility_status={py_data.get('feasibility_status')!r}, "
            f"expected {expected_status!r}"
        )
    if py_data.get("expected_failure") is not True or py_data.get("pass") is not True:
        issues.append("Python did not classify expected infeasibility as PASS")
    for key in (
        "parameter_difference", "parameter_difference_excluding_theta", "cross_rmse",
        "relative_cross_rmse", "signal_scale", "c_cost", "cost_difference",
        "rmse_difference", "max_error_difference",
    ):
        result[key] = None
    result["cross_status"] = "EXPECTED_INFEASIBLE_PASS" if not issues else "FAIL"


def compare_case(
    name: str, c_data: Mapping[str, Any] | None,
    py_data: Mapping[str, Any] | None,
) -> dict[str, Any]:
    meta = CASES[name]
    result: dict[str, Any] = {
        "name": name, "meta": meta, "c": c_data, "py": py_data, "issues": [],
        "c_final_used": None, "parameter_difference": None,
        "parameter_difference_excluding_theta": None, "cross_rmse": math.nan,
        "relative_cross_rmse": math.nan, "signal_scale": math.nan,
        "c_cost": math.nan, "cost_difference": math.nan,
        "rmse_difference": math.nan, "max_error_difference": math.nan,
        "cross_status": "NOT_RUN",
    }
    issues: list[str] = result["issues"]
    if c_data is None:
        issues.append(f"C CASE block missing: {name}")
    if py_data is None:
        issues.append(f"Python result missing: {name}")
    if c_data is None or py_data is None:
        return result

    for source, data, required in (
        ("C", c_data, C_REQUIRED), ("Python", py_data, PY_REQUIRED)
    ):
        for key in sorted(required - set(data)):
            issues.append(f"{source} field missing: {key}")
    if meta["mode"] == "near":
        for key in sorted(NEAR_C_REQUIRED - set(c_data)):
            issues.append(f"C field missing: {key}")
        for key in sorted(NEAR_PY_REQUIRED - set(py_data)):
            issues.append(f"Python field missing: {key}")

    if int(c_data.get("pass", 0)) != 1 or c_data.get("test_result") != "PASS":
        issues.append("C case self-check did not report PASS")
    if py_data.get("pass") is not True:
        issues.append("Python case self-check did not report PASS")
    if int(c_data.get("finite_check", 0)) != 1 or py_data.get("finite_check") is not True:
        issues.append("C or Python finite_check failed")

    m = number(c_data, "m", issues, "C")
    py_m = number(py_data, "m", issues, "Python")
    n = number(c_data, "n", issues, "C")
    py_n = number(py_data, "n", issues, "Python")
    m_aug = number(c_data, "m_aug", issues, "C")
    if math.isfinite(m) and int(m) != meta["m"]:
        issues.append(f"C m={int(m)}, expected {meta['m']}")
    if math.isfinite(py_m) and int(py_m) != meta["m"]:
        issues.append(f"Python m={int(py_m)}, expected {meta['m']}")
    if any(math.isfinite(value) and int(value) != 7 for value in (n, py_n)):
        issues.append("C or Python n is not 7")
    expected_m_aug = meta["m"] + meta["bound_count"]
    if math.isfinite(m_aug) and int(m_aug) != expected_m_aug:
        issues.append(f"C m_aug={int(m_aug)}, expected {expected_m_aug}")
    if int(c_data.get("grid_nx", 0)) * int(c_data.get("grid_ny", 0)) != meta["m"]:
        issues.append("C grid dimensions do not match m")

    for source, data in (("C", c_data), ("Python", py_data)):
        vector(data, "initial", issues, source)
        vector(data, "truth", issues, source)
    if meta["mode"] == "infeasible":
        compare_infeasible(name, c_data, py_data, result)
        return result

    c_status = number(c_data, "status", issues, "C")
    py_status = number(py_data, "status", issues, "Python")
    if math.isfinite(c_status) and int(c_status) != SUCCESS:
        issues.append(f"C status={int(c_status)}, expected AUGLAG_SUCCESS")
    if py_data.get("success") is not True or (math.isfinite(py_status) and py_status <= 0):
        issues.append("Python solver did not report success")
    if py_data.get("solver_executed") is not True:
        issues.append("Python feasible solver was not executed")
    if bool(c_data.get("expected_failure")) or bool(py_data.get("expected_failure")):
        issues.append("feasible case incorrectly marked expected_failure")

    c_final_key = "strict_final" if meta["mode"] == "near" else "final"
    c_final = vector(c_data, c_final_key, issues, "C")
    py_final = vector(py_data, "strict_final" if meta["mode"] == "near" else "final", issues, "Python")
    result["c_final_used"] = c_final
    c_violation_key = "strict_constraint_violation" if meta["mode"] == "near" else "constraint_violation"
    c_tol_key = "strict_constraint_tol" if meta["mode"] == "near" else "constraint_tol"
    c_violation = number(c_data, c_violation_key, issues, "C")
    c_tolerance = number(c_data, c_tol_key, issues, "C")
    py_violation = number(py_data, "constraint_violation", issues, "Python")
    if math.isfinite(c_violation) and math.isfinite(c_tolerance) and c_violation > c_tolerance:
        issues.append(f"C constraint violation {c_violation:.12g} exceeds configured tolerance")
    if math.isfinite(py_violation) and py_violation > VIOLATION_LIMIT:
        issues.append(f"Python constraint violation {py_violation:.12g} exceeds 1e-8")
    if meta["mode"] == "near":
        primary_violation = number(c_data, "constraint_violation", issues, "C")
        primary_tol = number(c_data, "constraint_tol", issues, "C")
        if primary_violation > primary_tol:
            issues.append("C standard-tolerance near-bound solve is infeasible")
        if int(c_data.get("strict_status", 0)) != SUCCESS or int(c_data.get("strict_finite_check", 0)) != 1:
            issues.append("C strict-tolerance near-bound solve failed")
        if not bool(c_data.get("mathematically_interior")) or not bool(py_data.get("mathematically_interior")):
            issues.append("near-bound case was not classified mathematically interior")
        if not bool(c_data.get("active_by_tolerance_1e8")) or bool(c_data.get("active_by_tolerance_1e10")):
            issues.append("C near-bound active-by-tolerance classification is inconsistent")
        if not bool(py_data.get("active_by_tolerance_1e-8")) or bool(py_data.get("active_by_tolerance_1e-10")):
            issues.append("Python near-bound active-by-tolerance classification is inconsistent")

    jac_error = number(c_data, "jac_error", issues, "C")
    if name not in {"gaussian_bad_scaling", "gaussian_narrow_width"} and (
        math.isfinite(jac_error) and jac_error >= 1e-5
    ):
        issues.append(f"C jac_error={jac_error:.12g} is not below 1e-5")
    outer = number(c_data, "outer_callback_count", issues, "C")
    if math.isfinite(outer) and outer <= 0:
        issues.append("C outer_callback_count must be positive")

    c_rmse = number(c_data, "rmse", issues, "C")
    py_rmse = number(py_data, "rmse", issues, "Python")
    c_max = number(c_data, "max_prediction_error", issues, "C")
    py_max = number(py_data, "max_prediction_error", issues, "Python")
    py_cost = number(py_data, "cost", issues, "Python")
    if math.isfinite(c_rmse) and math.isfinite(py_rmse):
        result["rmse_difference"] = abs(c_rmse - py_rmse)
        result["c_cost"] = 0.5 * meta["m"] * c_rmse * c_rmse
    if math.isfinite(result["c_cost"]) and math.isfinite(py_cost):
        result["cost_difference"] = abs(result["c_cost"] - py_cost)
    if math.isfinite(c_max) and math.isfinite(py_max):
        result["max_error_difference"] = abs(c_max - py_max)

    if c_final is not None and py_final is not None:
        check_bounds("C", c_final, meta["lower"], meta["upper"], issues)
        check_bounds("Python", py_final, meta["lower"], meta["upper"], issues)
        difference = c_final - py_final
        if meta["mode"] == "rank":
            result["parameter_difference_excluding_theta"] = np.delete(difference, 4)
            if abs(c_final[2] - c_final[3]) >= 1e-4 or abs(py_final[2] - py_final[3]) >= 1e-4:
                issues.append("rank-deficient circular Gaussian widths are not equal")
        else:
            result["parameter_difference"] = difference

        try:
            x = np.asarray(py_data["x"], dtype=float).reshape(-1)
            y = np.asarray(py_data["y"], dtype=float).reshape(-1)
            observed = np.asarray(py_data["observed"], dtype=float).reshape(-1)
            py_prediction = np.asarray(py_data["prediction"], dtype=float).reshape(-1)
        except (KeyError, TypeError, ValueError):
            x = y = observed = py_prediction = np.empty(0)
        arrays = (x, y, observed, py_prediction)
        if any(array.size != meta["m"] or not np.all(np.isfinite(array)) for array in arrays):
            issues.append(f"Python grid/observed/prediction must contain {meta['m']} finite values")
        else:
            prediction_difference = gaussian_model(c_final, x, y) - py_prediction
            result["cross_rmse"] = float(np.sqrt(np.mean(prediction_difference ** 2)))
            result["signal_scale"] = float(max(1.0, np.max(np.abs(observed))))
            result["relative_cross_rmse"] = result["cross_rmse"] / result["signal_scale"]
            metric = (
                result["relative_cross_rmse"]
                if meta["mode"] == "relative" else result["cross_rmse"]
            )
            if not math.isfinite(metric) or metric >= meta["cross_limit"]:
                metric_name = "relative_prediction_rmse" if meta["mode"] == "relative" else "prediction_cross_rmse"
                issues.append(
                    f"{metric_name}={metric:.12g} is not below {meta['cross_limit']:.1e}"
                )

    if meta["mode"] != "near":
        c_entries = c_active_entries(c_data)
        py_entries = py_data.get("active_bounds", [])
        if not isinstance(py_entries, Sequence) or isinstance(py_entries, (str, bytes)):
            issues.append("Python active_bounds must be a sequence")
            py_entries = []
        validate_active_bounds("C", c_entries, meta["active"], c_final, issues)
        validate_active_bounds("Python", py_entries, meta["active"], py_final, issues)
    if name == "gaussian_box_bound":
        for source, final in (("C", c_final), ("Python", py_final)):
            if final is not None and not (1.0 + ACTIVE_TOL < final[2] < 1.4 - ACTIVE_TOL):
                issues.append(f"{source} box optimum c2={final[2]:.12g} is not interior")
    result["cross_status"] = "PASS" if not issues else "FAIL"
    return result


def fmt(value: Any) -> str:
    if value is MISSING:
        return "MISSING"
    if value is None:
        return "not_applicable"
    if isinstance(value, Mapping):
        return repr(dict(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            items = np.asarray(value, dtype=float).reshape(-1)
            return "[" + ", ".join(f"{item:.12g}" for item in items) + "]"
        except (TypeError, ValueError):
            return repr(value)
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def report_line(data: Mapping[str, Any] | None, key: str) -> str:
    value = MISSING if data is None else data.get(key, MISSING)
    return f"{key}={fmt(value)}"


def make_report(
    comparisons: list[dict[str, Any]], c_returncode: int, global_issues: list[str],
) -> str:
    lines = [
        "==================================================", "Gaussian AugLag Cross Validation Report",
        "==================================================", "", "Environment:",
        f"    platform={platform.platform()}", f"    python={platform.python_version()}",
        f"    numpy={np.__version__}", f"    scipy={scipy.__version__}",
        "C solver:", "    AugLag + CMinpack lmder",
        f"    executable={C_EXECUTABLE}", f"    returncode={c_returncode}",
        "Python reference:", "    scipy.optimize.least_squares",
        "    method=TRF", "    native bounds (independent feasibility check for Case 12)", "",
    ]
    if global_issues:
        lines.append("Global diagnostics:")
        lines.extend(f"    FAIL: {issue}" for issue in global_issues)
        lines.append("")
    c_fields = (
        "status", "pass", "test_result", "expected_failure", "finite_check",
        "m_aug", "constraint_tol", "outer_callback_count", "rho", "rho_max",
        "initial", "truth", "final", "jac_error", "jac_error_kind",
        "jac_error_limit", "rmse", "prediction_rmse",
        "relative_prediction_rmse", "max_prediction_error", "constraint_violation",
        "active_bound_count", "active_bounds", "active_bound_indices",
        "active_bound_kinds", "active_bound_values", "active_bound_targets",
        "active_bound_slacks", "parameter_scale_ratio", "theta_jacobian_norm",
        "underflow_zero_count",
    )
    py_fields = (
        "status", "success", "pass", "expected_failure", "feasibility_status",
        "solver_executed", "finite_check", "runtime_seconds", "nfev", "njev",
        "initial", "solver_initial", "truth", "final", "cost", "optimality", "rmse",
        "relative_fit_rmse", "max_prediction_error", "relative_max_prediction_error",
        "constraint_violation", "active_bound_count", "active_bounds", "signal_scale",
        "parameter_scale_ratio", "theta_jacobian_column_norm",
        "theta_jacobian_relative_norm", "normalized_theta", "width_difference",
        "underflow_zero_count",
    )
    for case_number, comparison in enumerate(comparisons, 1):
        name, meta = comparison["name"], comparison["meta"]
        c_data, py_data = comparison["c"], comparison["py"]
        lines.extend([
            "--------------------------------------------------", f"Case {case_number}: {meta['title']}",
            "--------------------------------------------------", "", "Problem:", f"name={name}",
            f"m={meta['m']}", "n=7",
            f"bounds/constraints=lower={fmt(meta['lower'])}, upper={fmt(meta['upper'])}",
            f"special_condition={meta['special']}",
            f"grid={report_line(c_data, 'grid_nx').split('=', 1)[1]}x"
            f"{report_line(c_data, 'grid_ny').split('=', 1)[1]}",
        ])
        if name == "gaussian_infeasible_initial":
            lines.append("note=C recovers from infeasible x0; SciPy uses projected x0")
        lines.extend(["", "C Result:"])
        lines.extend(report_line(c_data, key) for key in c_fields)
        lines.append(f"cost_from_rmse={fmt(comparison['c_cost'])}")
        if meta["mode"] == "near":
            lines.extend(report_line(c_data, key) for key in sorted(NEAR_C_REQUIRED))
            lines.append("classification=mathematically interior; activity reported per tolerance")
        lines.extend(["", "Python Result:"])
        lines.extend(report_line(py_data, key) for key in py_fields)
        if meta["mode"] == "near":
            lines.extend(report_line(py_data, key) for key in sorted(NEAR_PY_REQUIRED))

        difference = comparison["parameter_difference"]
        l2 = None if difference is None else float(np.linalg.norm(difference))
        maximum = None if difference is None else float(np.max(np.abs(difference)))
        lines.extend([
            "", "Cross Validation:", f"parameter_difference={fmt(difference)}",
            f"parameter_difference_excluding_theta={fmt(comparison['parameter_difference_excluding_theta'])}",
            f"parameter_l2_difference={fmt(l2)}",
            f"parameter_max_abs_difference={fmt(maximum)}",
            f"cost_difference={fmt(comparison['cost_difference'])}",
            f"rmse_difference={fmt(comparison['rmse_difference'])}",
            f"max_prediction_error_difference={fmt(comparison['max_error_difference'])}",
            f"signal_scale={fmt(comparison['signal_scale'])}",
            f"prediction_cross_rmse={fmt(comparison['cross_rmse'])}",
            f"relative_prediction_rmse={fmt(comparison['relative_cross_rmse'])}",
            f"cross_status={comparison['cross_status']}",
        ])
        if meta["mode"] == "rank":
            lines.append("theta_parameter_comparison=not_applicable_non_identifiable")
        if meta["mode"] == "infeasible":
            lines.append("final_parameter_comparison=not_applicable_expected_infeasible")
        lines.extend(f"FAIL_REASON={issue}" for issue in comparison["issues"])
        lines.extend(["PASS" if not comparison["issues"] else "FAIL", ""])

    lines.extend([
        "==================================================", "Test Coverage Summary",
        "==================================================",
        "- large m>>n nonlinear least squares and analytic Gaussian Jacobian",
        "- inactive, active lower, active upper, and interior box bounds",
        "- infeasible initial recovery, noisy data, and poor initial guess",
        "- severe parameter scaling / conditioning",
        "- optimum extremely close to a lower bound with 1e-8 and 1e-10 tolerances",
        "- multiple simultaneously active bounds",
        "- contradictory/infeasible box constraints",
        "- infeasible general constraint set",
        "- 10201-residual large-scale Gaussian problem",
        "- rank-deficient / non-identifiable Gaussian parameterization",
        "- narrow Gaussian and exponential underflow",
        "- finite-value and numerical stability paths",
        "- same nls_solver reused across AugLag outer solves",
        "- independent SciPy cross validation", "",
    ])
    overall = not global_issues and all(not item["issues"] for item in comparisons)
    lines.extend(["Overall:", "PASS" if overall else "FAIL", ""])
    return "\n".join(lines)


def main() -> int:
    global_issues: list[str] = []
    c_output, c_returncode, c_error = run_c()
    print("========== C RESULTS ==========")
    if c_output:
        print(c_output, end="" if c_output.endswith("\n") else "\n")
    if c_error:
        print(f"ERROR: {c_error}")
        global_issues.append(f"could not run C executable: {c_error}")
    if c_returncode != 0:
        global_issues.append(f"C executable returned {c_returncode}")
    c_cases, parse_errors = parse_c_cases(c_output)
    global_issues.extend(parse_errors)
    unexpected_c = sorted(set(c_cases) - set(CASES))
    if unexpected_c:
        global_issues.append(f"unexpected C CASE blocks: {unexpected_c}")

    py_cases, py_output, py_error = run_reference()
    print("========== PYTHON RESULTS ==========")
    if py_output:
        print(py_output, end="" if py_output.endswith("\n") else "\n")
    if py_error:
        print(f"ERROR: {py_error}")
        global_issues.append(f"Python reference failed: {py_error}")
    unexpected_py = sorted(set(py_cases) - set(CASES))
    if unexpected_py:
        global_issues.append(f"unexpected Python results: {unexpected_py}")

    comparisons = [
        compare_case(name, c_cases.get(name), py_cases.get(name)) for name in CASES
    ]
    print("========== CROSS VALIDATION ==========")
    for comparison in comparisons:
        mode = comparison["meta"]["mode"]
        metric = None if mode == "infeasible" else (
            comparison["relative_cross_rmse"] if mode == "relative"
            else comparison["cross_rmse"]
        )
        print(
            f"{comparison['name']}: metric={fmt(metric)} "
            f"cross_status={comparison['cross_status']} "
            f"verdict={'PASS' if not comparison['issues'] else 'FAIL'}"
        )
        for issue in comparison["issues"]:
            print(f"  - {issue}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        make_report(comparisons, c_returncode, global_issues), encoding="utf-8"
    )
    overall = not global_issues and all(not item["issues"] for item in comparisons)
    print("========== REPORT ==========")
    print(f"path={REPORT_PATH}")
    print(f"Overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
