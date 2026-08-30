#!/usr/bin/env python3
"""Double Gaussian SciPy reference and C/Python cross-validation report.

The input coordinates are already transformed Mahalanobis-space points.  This
module intentionally does not attempt to reproduce the unavailable original
MahalanobisTransformer.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
    import scipy
    from scipy.optimize import minimize
except ModuleNotFoundError as exc:  # Kept explicit for build.sh diagnostics.
    dependency_error = f"MISSING_PYTHON_DEPENDENCY: {exc}"
    print(dependency_error, file=sys.stderr)
    if "--report" in sys.argv:
        report_index = sys.argv.index("--report") + 1
        if report_index < len(sys.argv):
            report_path = Path(sys.argv[report_index])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                "Double Gaussian Minimize Cross-Validation Report\n"
                f"{dependency_error}\n"
                "Overall FAIL\n",
                encoding="utf-8",
            )
    raise SystemExit(2) from exc


PARAMETER_COUNT = 7
REFERENCE_LEVEL = "mahalanobis_space_optimization_core"
FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1
DEFAULT_LOWER = np.array(
    [-np.inf, 1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10, -np.inf, -np.inf],
    dtype=np.float64,
)


class DoubleGaussianSurfaceFitterByMinimizeInMahalanobis:
    """Optimization-compatible core of the user's fitter class."""

    reference_level = REFERENCE_LEVEL

    def __init__(
        self,
        transformed_points: np.ndarray,
        observed: np.ndarray,
        weights: np.ndarray,
        initial_guess: Iterable[float],
        bounds: list[tuple[float | None, float | None]] | None = None,
        *,
        max_iter: int = 1000,
        ftol: float = 1.0e-10,
        gtol: float = 1.0e-8,
    ) -> None:
        points = np.asarray(transformed_points, dtype=np.float64)
        self.observed = np.asarray(observed, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.initial_guess = np.asarray(tuple(initial_guess), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("transformed_points must have shape (m, 2)")
        if self.observed.shape != (points.shape[0],) or self.weights.shape != (
            points.shape[0],
        ):
            raise ValueError("observed and weights must match transformed_points")
        if self.initial_guess.shape != (PARAMETER_COUNT,):
            raise ValueError("initial_guess must contain seven parameters")
        self.h = points[:, 0]
        self.v = points[:, 1]
        self.bounds = bounds or [
            (None, None),
            (1.0e-10, None),
            (1.0e-10, None),
            (1.0e-10, None),
            (1.0e-10, None),
            (None, None),
            (None, None),
        ]
        self.max_iter = max_iter
        self.ftol = ftol
        self.gtol = gtol
        self.step_history: list[np.ndarray] = []
        self.result: Any | None = None

    @staticmethod
    def _model_at(params: np.ndarray, h: np.ndarray, v: np.ndarray) -> np.ndarray:
        c0, c1, c2, c3, c4, c5, c6 = np.asarray(params, dtype=np.float64)
        a = c3 * h + c4 * v + c5
        b = c3 * h - c4 * v + c6
        return c0 + c1 * np.exp(-(a * a)) + c2 * np.exp(-(b * b))

    def _model(self, params: np.ndarray) -> np.ndarray:
        return self._model_at(params, self.h, self.v)

    def objective(self, params: np.ndarray) -> float:
        # The original class uses weights only as an inclusion mask.
        residuals = self.observed - self._model(params)
        return float(np.sum(residuals[self.weights > 0] ** 2))

    def _callback(self, xk: np.ndarray) -> None:
        self.step_history.append(np.array(xk, dtype=np.float64, copy=True))

    def fit(self) -> Any:
        self.step_history.clear()
        self.result = minimize(
            fun=self.objective,
            x0=self.initial_guess,
            method="L-BFGS-B",
            bounds=self.bounds,
            options={
                "maxiter": self.max_iter,
                "ftol": self.ftol,
                "gtol": self.gtol,
            },
            callback=self._callback,
        )
        return self.result

    def output(self, params: np.ndarray | None = None) -> tuple[float, float]:
        if params is None:
            if self.result is None:
                raise RuntimeError("fit() has not been called")
            params = np.asarray(self.result.x, dtype=np.float64)
        c3, c4, c5, c6 = params[3], params[4], params[5], params[6]
        return (-(c5 + c6) / (2.0 * c3), -(c5 - c6) / (2.0 * c4))


@dataclass(frozen=True)
class CaseSpec:
    name: str
    title: str
    truth: tuple[float, ...]
    initial: tuple[float, ...]
    h_limits: tuple[float, float] = (-2.0, 2.0)
    v_limits: tuple[float, float] = (-2.0, 2.0)
    nx: int = 41
    ny: int = 41
    mask_every: int | None = None
    noise_scale: float = 0.0
    c3_bound: float = 1.0e-10
    prediction_limit: float = 1.0e-6
    max_point_limit: float = 1.0e-5


@dataclass
class CaseData:
    spec: CaseSpec
    h: np.ndarray
    v: np.ndarray
    observed: np.ndarray
    weights: np.ndarray
    valid: np.ndarray
    lower: np.ndarray

    @property
    def truth(self) -> np.ndarray:
        return np.asarray(self.spec.truth, dtype=np.float64)

    @property
    def initial(self) -> np.ndarray:
        return np.asarray(self.spec.initial, dtype=np.float64)


CASE_SPECS = (
    CaseSpec(
        "DG1",
        "NORMAL",
        (0.20, 1.50, 1.00, 0.80, 1.10, -0.30, 0.40),
        (0.10, 1.00, 0.70, 1.10, 0.80, 0.00, 0.10),
    ),
    CaseSpec(
        "DG2",
        "SMALL MAGNITUDE",
        (1.0e-3, 7.5e-3, 5.0e-3, 0.80, 1.10, -0.30, 0.40),
        (0.5e-3, 5.0e-3, 3.5e-3, 1.10, 0.80, 0.00, 0.10),
        prediction_limit=5.0e-6,
        max_point_limit=5.0e-5,
    ),
    CaseSpec(
        "DG3",
        "LARGE MAGNITUDE",
        (2.0e3, 1.5e4, 1.0e4, 0.80, 1.10, -0.30, 0.40),
        (1.0e3, 1.0e4, 0.7e4, 1.10, 0.80, 0.00, 0.10),
        prediction_limit=2.0e-6,
        max_point_limit=2.0e-4,
    ),
    CaseSpec(
        "DG4",
        "COORDINATE SCALING",
        (0.20, 1.50, 1.00, 80.0, 110.0, -0.30, 0.40),
        (0.10, 1.00, 0.70, 110.0, 80.0, 0.00, 0.10),
        h_limits=(-0.02, 0.02),
        v_limits=(-0.02, 0.02),
        prediction_limit=2.0e-6,
        max_point_limit=2.0e-5,
    ),
    CaseSpec(
        "DG5",
        "ACTIVE BOUND + MASK + NOISE",
        (0.20, 1.50, 1.00, 0.65, 1.10, -0.30, 0.40),
        (0.10, 1.00, 0.70, 1.10, 0.80, 0.00, 0.10),
        mask_every=7,
        noise_scale=1.0,
        c3_bound=0.90,
        prediction_limit=2.0e-5,
        max_point_limit=5.0e-4,
    ),
)


def model(params: np.ndarray, h: np.ndarray, v: np.ndarray) -> np.ndarray:
    return DoubleGaussianSurfaceFitterByMinimizeInMahalanobis._model_at(
        params, h, v
    )


def make_case(spec: CaseSpec) -> CaseData:
    h_axis = np.linspace(spec.h_limits[0], spec.h_limits[1], spec.nx)
    v_axis = np.linspace(spec.v_limits[0], spec.v_limits[1], spec.ny)
    h_grid, v_grid = np.meshgrid(h_axis, v_axis)
    h = np.asarray(h_grid.ravel(), dtype=np.float64)
    v = np.asarray(v_grid.ravel(), dtype=np.float64)
    truth = np.asarray(spec.truth, dtype=np.float64)
    observed = model(truth, h, v)
    indices = np.arange(observed.size, dtype=np.float64)
    if spec.noise_scale != 0.0:
        observed = observed + spec.noise_scale * (
            0.01 * np.sin(0.37 * indices) + 0.005 * np.cos(0.11 * indices)
        )
    weights = np.ones(observed.size, dtype=np.float64)
    if spec.mask_every is not None:
        weights = np.resize(
            np.array([0.1, 1.0, 100.0], dtype=np.float64), observed.size
        )
        weights[:: spec.mask_every] = 0.0
    valid = weights > 0
    lower = DEFAULT_LOWER.copy()
    lower[3] = spec.c3_bound
    return CaseData(spec, h, v, observed, weights, valid, lower)


def fnv_u64(value: int, hash_value: int) -> int:
    return ((hash_value ^ (value & UINT64_MASK)) * FNV_PRIME) & UINT64_MASK


def double_bits(value: float) -> int:
    return struct.unpack("=Q", struct.pack("=d", float(value)))[0]


def input_hash(case: CaseData) -> str:
    valid_indices = np.flatnonzero(case.valid)
    hash_value = FNV_OFFSET
    for value in (
        1,
        case.spec.nx,
        case.spec.ny,
        case.h.size,
        case.h.size - valid_indices.size,
        valid_indices.size,
    ):
        hash_value = fnv_u64(int(value), hash_value)
    for vector in (case.truth, case.initial, case.lower):
        for value in vector:
            hash_value = fnv_u64(double_bits(float(value)), hash_value)
    for index in valid_indices:
        hash_value = fnv_u64(int(index), hash_value)
        hash_value = fnv_u64(double_bits(case.h[index]), hash_value)
        hash_value = fnv_u64(double_bits(case.v[index]), hash_value)
        hash_value = fnv_u64(double_bits(case.observed[index]), hash_value)
    return f"{hash_value:016x}"


def format_float(value: float) -> str:
    return format(float(value), ".17g")


def format_vector(values: Iterable[float]) -> str:
    return "[" + ", ".join(format_float(value) for value in values) + "]"


def write_c_input(case: CaseData, path: Path) -> None:
    valid_indices = np.flatnonzero(case.valid)
    lines = [
        "NLS_DG_INPUT_V1",
        f"name {case.spec.name}",
        f"grid_nx {case.spec.nx}",
        f"grid_ny {case.spec.ny}",
        f"full_count {case.h.size}",
        f"mask_count {case.h.size - valid_indices.size}",
        f"m {valid_indices.size}",
        "truth " + " ".join(format_float(x) for x in case.truth),
        "initial " + " ".join(format_float(x) for x in case.initial),
        "lower " + " ".join(format_float(x) for x in case.lower),
        "data",
    ]
    lines.extend(
        f"{int(i)} {format_float(case.h[i])} {format_float(case.v[i])} "
        f"{format_float(case.observed[i])}"
        for i in valid_indices
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        body = raw[1:-1].strip()
        return [] if not body else [float(item) for item in body.split(",")]
    if raw in ("PASS", "FAIL"):
        return raw
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_c_output(output: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == "C_RESULT_BEGIN":
            if current is not None:
                raise ValueError("nested C_RESULT_BEGIN")
            current = {}
        elif line == "C_RESULT_END":
            if current is None:
                raise ValueError("C_RESULT_END without begin")
            algorithm = str(current.get("algorithm", ""))
            if algorithm not in ("LM", "GN") or algorithm in results:
                raise ValueError(f"invalid/duplicate C algorithm {algorithm!r}")
            results[algorithm] = current
            current = None
        elif current is not None:
            if "=" not in line:
                raise ValueError(f"malformed C result line: {line!r}")
            key, value = line.split("=", 1)
            current[key] = parse_value(value)
    if current is not None or set(results) != {"LM", "GN"}:
        raise ValueError("incomplete C result output")
    return results


def run_c_case(runner: Path, input_path: Path) -> dict[str, dict[str, Any]]:
    completed = subprocess.run(
        [str(runner), str(input_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"C runner failed ({completed.returncode}):\n{completed.stdout}"
        )
    return parse_c_output(completed.stdout)


def bounds_from_lower(lower: np.ndarray) -> list[tuple[float | None, None]]:
    return [(None if not np.isfinite(value) else float(value), None) for value in lower]


def maximum_point(params: np.ndarray) -> np.ndarray:
    return np.array(
        [
            -(params[5] + params[6]) / (2.0 * params[3]),
            -(params[5] - params[6]) / (2.0 * params[4]),
        ],
        dtype=np.float64,
    )


def constraint_violation(params: np.ndarray, lower: np.ndarray) -> float:
    finite = np.isfinite(lower)
    return float(max(0.0, np.max(lower[finite] - params[finite])))


def prediction_metrics(
    first: np.ndarray,
    second: np.ndarray,
    h: np.ndarray,
    v: np.ndarray,
    signal_scale: float,
) -> tuple[float, float]:
    difference = model(first, h, v) - model(second, h, v)
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return rmse, rmse / signal_scale


def optimizer_metrics(
    params: np.ndarray,
    case: CaseData,
) -> dict[str, Any]:
    valid = case.valid
    prediction = model(params, case.h[valid], case.v[valid])
    residual = prediction - case.observed[valid]
    sse = float(np.sum(residual * residual))
    return {
        "final": params,
        "sse": sse,
        "rmse": math.sqrt(sse / int(np.count_nonzero(valid))),
        "constraint_violation": constraint_violation(params, case.lower),
        "finite": bool(
            np.all(np.isfinite(params))
            and np.all(np.isfinite(prediction))
            and math.isfinite(sse)
        ),
        "max_point": maximum_point(params),
    }


def run_python_case(case: CaseData) -> tuple[dict[str, Any], Any]:
    points = np.column_stack((case.h, case.v))
    fitter = DoubleGaussianSurfaceFitterByMinimizeInMahalanobis(
        points,
        case.observed,
        case.weights,
        case.initial,
        bounds=bounds_from_lower(case.lower),
    )
    result = fitter.fit()
    metrics = optimizer_metrics(np.asarray(result.x, dtype=np.float64), case)
    metrics["max_point"] = np.asarray(fitter.output(result.x), dtype=np.float64)
    metrics.update(
        {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message).replace("\n", " "),
            "fun": float(result.fun),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "njev": int(getattr(result, "njev", -1)),
            "callback_iteration_count": len(fitter.step_history),
        }
    )
    return metrics, result


def c_metrics(record: dict[str, Any], case: CaseData) -> dict[str, Any]:
    params = np.asarray(record["final"], dtype=np.float64)
    recomputed = optimizer_metrics(params, case)
    result = dict(record)
    result["reported_sse"] = float(record["sse"])
    result["reported_rmse"] = float(record["rmse"])
    result["reported_constraint_violation"] = float(record["constraint_violation"])
    result["reported_max_point"] = np.asarray(record["max_point"], dtype=np.float64)
    result.update(recomputed)
    result["final"] = params
    result["finite"] = record.get("finite_check") == "PASS" and bool(
        recomputed["finite"]
    )
    return result


def compare_case(
    case: CaseData,
    c_records: dict[str, dict[str, Any]],
    python: dict[str, Any],
) -> dict[str, Any]:
    lm = c_metrics(c_records["LM"], case)
    gn = c_metrics(c_records["GN"], case)
    valid_h = case.h[case.valid]
    valid_v = case.v[case.valid]
    signal_scale = max(float(np.max(np.abs(case.observed[case.valid]))), 1.0e-300)
    valid_count = int(np.count_nonzero(case.valid))
    numeric_sse_floor = (
        signal_scale * signal_scale * valid_count
        * np.finfo(np.float64).eps ** 2 * 100.0
    )
    lm_py = prediction_metrics(lm["final"], python["final"], valid_h, valid_v, signal_scale)
    gn_py = prediction_metrics(gn["final"], python["final"], valid_h, valid_v, signal_scale)
    lm_gn = prediction_metrics(lm["final"], gn["final"], valid_h, valid_v, signal_scale)
    lm_py_max = float(np.linalg.norm(lm["max_point"] - python["max_point"]))
    gn_py_max = float(np.linalg.norm(gn["max_point"] - python["max_point"]))
    lm_gn_max = float(np.linalg.norm(lm["max_point"] - gn["max_point"]))
    expected_hash = input_hash(case)
    input_ok = all(
        str(c_records[name].get("input_hash")) == expected_hash
        and str(c_records[name].get("case")) == case.spec.name
        for name in ("LM", "GN")
    )
    constraint_limit = 1.1e-8
    jac_limit = 5.0e-6
    jac_ok = all(
        math.isfinite(float(c_records[name]["jac_relative_error"]))
        and float(c_records[name]["jac_relative_error"]) <= jac_limit
        for name in ("LM", "GN")
    )
    status_ok = (
        int(lm["status"]) == 1
        and int(gn["status"]) == 1
        and bool(python["success"])
    )
    finite_ok = bool(lm["finite"] and gn["finite"] and python["finite"])
    constraints_ok = max(
        float(lm["constraint_violation"]),
        float(gn["constraint_violation"]),
        float(python["constraint_violation"]),
    ) <= constraint_limit
    prediction_ok = max(lm_py[1], gn_py[1], lm_gn[1]) <= case.spec.prediction_limit
    max_point_ok = max(lm_py_max, gn_py_max, lm_gn_max) <= case.spec.max_point_limit
    rmse_agreement = max(
        abs(float(lm["rmse"]) - float(python["rmse"])),
        abs(float(gn["rmse"]) - float(python["rmse"])),
    ) / signal_scale
    objective_ok = rmse_agreement <= case.spec.prediction_limit
    c_reporting_ok = all(
        math.isclose(
            float(result["reported_sse"]),
            float(result["sse"]),
            rel_tol=1.0e-9,
            abs_tol=numeric_sse_floor,
        )
        and math.isclose(
            float(result["reported_rmse"]),
            float(result["rmse"]),
            rel_tol=1.0e-9,
            abs_tol=math.sqrt(numeric_sse_floor / valid_count),
        )
        and math.isclose(
            float(result["reported_constraint_violation"]),
            float(result["constraint_violation"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
        and bool(
            np.allclose(
                result["reported_max_point"],
                result["max_point"],
                rtol=1.0e-12,
                atol=1.0e-14,
            )
        )
        for result in (lm, gn)
    )
    python_objective_ok = math.isclose(
        float(python["fun"]),
        float(python["sse"]),
        rel_tol=1.0e-12,
        abs_tol=numeric_sse_floor,
    )
    metric_consistency_ok = c_reporting_ok and python_objective_ok
    callback_ok = abs(int(python["nit"]) - int(python["callback_iteration_count"])) <= 1
    active_bound_ok = case.spec.name != "DG5" or max(
        abs(float(lm["final"][3]) - case.spec.c3_bound),
        abs(float(gn["final"][3]) - case.spec.c3_bound),
        abs(float(python["final"][3]) - case.spec.c3_bound),
    ) <= 2.0e-6
    result_ok = all(
        (
            status_ok,
            finite_ok,
            constraints_ok,
            prediction_ok,
            max_point_ok,
            objective_ok,
            metric_consistency_ok,
            jac_ok,
            callback_ok,
            active_bound_ok,
        )
    )
    return {
        "lm": lm,
        "gn": gn,
        "python": python,
        "signal_scale": signal_scale,
        "input_hash": expected_hash,
        "input_ok": input_ok,
        "jac_limit": jac_limit,
        "jac_ok": jac_ok,
        "status_ok": status_ok,
        "finite_ok": finite_ok,
        "constraints_ok": constraints_ok,
        "prediction_ok": prediction_ok,
        "max_point_ok": max_point_ok,
        "objective_ok": objective_ok,
        "metric_consistency_ok": metric_consistency_ok,
        "callback_ok": callback_ok,
        "active_bound_ok": active_bound_ok,
        "lm_py_prediction": lm_py,
        "gn_py_prediction": gn_py,
        "lm_gn_prediction": lm_gn,
        "lm_py_max": lm_py_max,
        "gn_py_max": gn_py_max,
        "lm_gn_max": lm_gn_max,
        "lm_py_param": float(np.max(np.abs(lm["final"] - python["final"]))),
        "gn_py_param": float(np.max(np.abs(gn["final"] - python["final"]))),
        "lm_gn_param": float(np.max(np.abs(lm["final"] - gn["final"]))),
        "rmse_agreement_relative": rmse_agreement,
        "result_ok": result_ok,
        "pass": input_ok and result_ok,
    }


def command_first_line(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            check=False,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        for line in lines:
            if "clang version" in line.lower() or line.lower().startswith("cmake version"):
                return line
        return lines[0] if lines else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def result_lines(prefix: str, result: dict[str, Any], is_c: bool) -> list[str]:
    lines = [prefix]
    if is_c:
        lines.extend(
            [
                f"status={result['status']}",
                f"final={format_vector(result['final'])}",
                f"SSE={format_float(result['sse'])}",
                f"RMSE={format_float(result['rmse'])}",
                f"C_reported_SSE={format_float(result['reported_sse'])}",
                f"C_reported_RMSE={format_float(result['reported_rmse'])}",
                f"constraint_violation={format_float(result['constraint_violation'])}",
                f"finite_check={'PASS' if result['finite'] else 'FAIL'}",
                f"max_point={format_vector(result['max_point'])}",
                f"rho_final={format_float(result['rho_final'])}",
                f"outer_iterations={result['outer_iterations']}",
                f"inner_iterations={result['inner_iterations']}",
                f"inner_nfev={result['inner_nfev']}",
                f"inner_njev={result['inner_njev']}",
                "inner_iteration_metric="
                + ("nfev" if prefix.startswith("C LM") else "GN_iterations"),
            ]
        )
    else:
        lines.extend(
            [
                f"success={result['success']}",
                f"status={result['status']}",
                f"message={result['message']}",
                f"final={format_vector(result['final'])}",
                f"fun={format_float(result['fun'])}",
                f"SSE={format_float(result['sse'])}",
                f"RMSE={format_float(result['rmse'])}",
                f"constraint_violation={format_float(result['constraint_violation'])}",
                f"finite_check={'PASS' if result['finite'] else 'FAIL'}",
                f"max_point={format_vector(result['max_point'])}",
                f"nit={result['nit']}",
                f"nfev={result['nfev']}",
                f"njev={result['njev']}",
                f"callback_iteration_count={result['callback_iteration_count']}",
            ]
        )
    return lines


def build_report(
    cases: list[tuple[CaseData, dict[str, Any]]],
    overall: bool,
) -> str:
    lines = [
        "Double Gaussian Minimize Cross-Validation Report",
        "",
        "Environment:",
        f"OS={platform.platform()}",
        f"architecture={platform.machine()}",
        f"compiler={command_first_line(['cc', '--version'])}",
        f"CMake={command_first_line(['cmake', '--version'])}",
        f"Python={sys.version.split()[0]}",
        f"NumPy={np.__version__}",
        f"SciPy={scipy.__version__}",
        "",
        "C Solver:",
        "LM implementation=CMinpack lmder",
        "GN implementation=project Gauss-Newton",
        "LLS implementation=LM uses CMinpack internal QR; GN uses Cholesky",
        "unsupported combination=GN + QR returns allocation failure",
        "",
        "Python Reference:",
        "scipy.optimize.minimize",
        "method=L-BFGS-B",
        "reference_class=DoubleGaussianSurfaceFitterByMinimizeInMahalanobis-compatible core",
        f"reference_level={REFERENCE_LEVEL}",
        "MahalanobisTransformer=not available; comparison starts from transformed_points",
        "maximum_point_space=Mahalanobis transformed space",
        "objective=sum((observed-model)^2) over weights>0; weights are a mask only",
        "",
        "C Core Tests:",
        f"nls_core={os.environ.get('NLS_CTEST_STATUS', 'NOT_RECORDED')}",
        f"auglag_core_api={os.environ.get('NLS_CTEST_STATUS', 'NOT_RECORDED')}",
        "",
        "Test Suite Scope:",
        "existing_auglag_top_level_cases_before=19",
        "existing_auglag_top_level_cases_after=6 (5 numeric cores + 1 API matrix)",
        "new_double_gaussian_cases=5",
        "kept_CORE1=overdetermined nonlinear NLS; LM/GN/public APIs/Jacobian/GN+QR rejection",
        "kept_CORE2=mixed equality and active inequality from an infeasible initial point",
        "kept_CORE3=minimal active lower-bound convergence",
        "kept_CORE4=contradictory inequalities; prevents false constrained success",
        "kept_CORE5A=rank-deficient model judged by prediction and identifiable sum",
        "kept_CORE5B=large-m scaled numerical coverage",
        "kept_API_matrix=default/custom/invalid options and multiplier copy/validation",
        "removed_cases=unconstrained,equality-only,inequality-only,gaussian_normal,gaussian_infeasible_initial,gaussian_active_lower,gaussian_active_upper,gaussian_box_bound,gaussian_noisy,gaussian_poor_initial,gaussian_bad_scaling,gaussian_near_lower_bound,gaussian_two_active_bounds,gaussian_infeasible_box,gaussian_infeasible_constraints,gaussian_large_grid,gaussian_rank_deficient,gaussian_narrow_width",
        "removal_reason=duplicate coverage replaced by focused CORE cases or DG1-DG5",
        "",
    ]
    for case, comparison in cases:
        valid_observed = case.observed[case.valid]
        mask_count = int(case.h.size - np.count_nonzero(case.valid))
        noise = (
            "none"
            if case.spec.noise_scale == 0.0
            else "scale*(0.01*sin(0.37*i)+0.005*cos(0.11*i)), scale=1"
        )
        lines.extend(
            [
                f"CASE {case.spec.name} {case.spec.title}",
                "",
                "INPUT",
                f"m={np.count_nonzero(case.valid)}",
                f"n={PARAMETER_COUNT}",
                f"grid={case.spec.nx}x{case.spec.ny}",
                f"h_range={format_vector([case.h.min(), case.h.max()])}",
                f"v_range={format_vector([case.v.min(), case.v.max()])}",
                f"signal_scale={format_float(comparison['signal_scale'])}",
                f"data_min={format_float(valid_observed.min())}",
                f"data_max={format_float(valid_observed.max())}",
                f"data_mean={format_float(valid_observed.mean())}",
                f"data_l2_norm={format_float(np.linalg.norm(valid_observed))}",
                f"data_checksum={comparison['input_hash']}",
                f"mask_count={mask_count}",
                "positive_weight_values=[0.1, 1, 100]"
                if case.spec.mask_every is not None
                else "positive_weight_values=[1]",
                "weights_semantics=mask_only (all positive values have equal objective weight)",
                f"noise={noise}",
                f"bounds={format_vector(case.lower)}",
                f"truth={format_vector(case.truth)}",
                f"initial={format_vector(case.initial)}",
                "",
            ]
        )
        lines.extend(result_lines("C LM + AUGLAG", comparison["lm"], True))
        lines.append("")
        lines.extend(result_lines("C GN + AUGLAG", comparison["gn"], True))
        lines.append("")
        lines.extend(result_lines("PYTHON L-BFGS-B", comparison["python"], False))
        lines.extend(
            [
                "",
                "JACOBIAN CHECK",
                f"jac_error={format_float(comparison['lm']['jac_error'])}",
                f"jac_relative_error={format_float(comparison['lm']['jac_relative_error'])}",
                f"jac_error_limit={format_float(comparison['jac_limit'])}",
                "fd_step_strategy=sqrt(DBL_EPSILON)*max(1,abs(param[j]))",
                f"jacobian_check={'PASS' if comparison['jac_ok'] else 'FAIL'}",
                "",
                "INPUT CONSISTENCY CHECK",
                f"python_input_hash={comparison['input_hash']}",
                f"C_LM_input_hash={comparison['lm']['input_hash']}",
                f"C_GN_input_hash={comparison['gn']['input_hash']}",
                f"same_m={'true' if comparison['input_ok'] else 'false'}",
                f"same_h={'true' if comparison['input_ok'] else 'false'}",
                f"same_v={'true' if comparison['input_ok'] else 'false'}",
                f"same_observed={'true' if comparison['input_ok'] else 'false'}",
                f"same_mask={'true' if comparison['input_ok'] else 'false'}",
                f"same_initial_guess={'true' if comparison['input_ok'] else 'false'}",
                f"same_bounds={'true' if comparison['input_ok'] else 'false'}",
                f"input_consistency={'PASS' if comparison['input_ok'] else 'FAIL'}",
                "",
                "CROSS",
                f"LM_vs_Python_param_max_abs_diff={format_float(comparison['lm_py_param'])}",
                f"GN_vs_Python_param_max_abs_diff={format_float(comparison['gn_py_param'])}",
                f"LM_vs_GN_param_max_abs_diff={format_float(comparison['lm_gn_param'])}",
                f"LM_vs_Python_prediction_rmse={format_float(comparison['lm_py_prediction'][0])}",
                f"GN_vs_Python_prediction_rmse={format_float(comparison['gn_py_prediction'][0])}",
                f"LM_vs_GN_prediction_rmse={format_float(comparison['lm_gn_prediction'][0])}",
                f"LM_vs_Python_relative_prediction_rmse={format_float(comparison['lm_py_prediction'][1])}",
                f"GN_vs_Python_relative_prediction_rmse={format_float(comparison['gn_py_prediction'][1])}",
                f"LM_vs_GN_relative_prediction_rmse={format_float(comparison['lm_gn_prediction'][1])}",
                f"LM_vs_Python_max_point_error={format_float(comparison['lm_py_max'])}",
                f"GN_vs_Python_max_point_error={format_float(comparison['gn_py_max'])}",
                f"LM_vs_GN_max_point_error={format_float(comparison['lm_gn_max'])}",
                f"relative_prediction_limit={format_float(case.spec.prediction_limit)}",
                f"max_point_limit={format_float(case.spec.max_point_limit)}",
                f"recomputed_metric_consistency={'PASS' if comparison['metric_consistency_ok'] else 'FAIL'}",
                f"result_consistency={'PASS' if comparison['result_ok'] else 'FAIL'}",
                f"case_result={'PASS' if comparison['pass'] else 'FAIL'}",
                "",
            ]
        )

    lines.extend(
        [
            "Solver Comparison Summary",
            "Case | Signal Scale | LM Result | GN Result | L-BFGS-B Result | LM Iter Metric | GN Iter | Python nit | Relative Prediction Error | PASS",
        ]
    )
    for case, comparison in cases:
        lm = comparison["lm"]
        gn = comparison["gn"]
        py = comparison["python"]
        rel = max(comparison["lm_py_prediction"][1], comparison["gn_py_prediction"][1])
        lines.append(
            f"{case.spec.name} | {comparison['signal_scale']:.6g} | {lm['status']} | "
            f"{gn['status']} | {py['success']} | nfev={lm['inner_nfev']} | "
            f"{gn['inner_iterations']} | {py['nit']} | {rel:.6g} | "
            f"{'PASS' if comparison['pass'] else 'FAIL'}"
        )
    lines.extend(
        [
            "",
            "Iteration Comparison",
            "Case | C LM outer_iter | C LM inner metric | C GN outer_iter | C GN inner metric | Python nit | Python nfev",
        ]
    )
    for case, comparison in cases:
        lm = comparison["lm"]
        gn = comparison["gn"]
        py = comparison["python"]
        lines.append(
            f"{case.spec.name} | {lm['outer_iterations']} | nfev={lm['inner_nfev']} | "
            f"{gn['outer_iterations']} | iterations={gn['inner_iterations']} | "
            f"{py['nit']} | {py['nfev']}"
        )
    lines.extend(
        [
            "",
            "Iteration Metric Note:",
            "AugLag outer_iterations counts constrained outer solves.",
            "LM inner metric is CMinpack nfev (not an iteration count).",
            "GN inner_iterations counts completed Gauss-Newton steps.",
            "L-BFGS-B nit and nfev are SciPy metrics; these measures are not strictly equivalent.",
            "",
            "Multi-Magnitude Summary:",
        ]
    )
    for case, comparison in cases:
        lines.append(
            f"{case.spec.name} signal_scale={comparison['signal_scale']:.12g} "
            f"relative_prediction_error="
            f"{max(comparison['lm_py_prediction'][1], comparison['gn_py_prediction'][1]):.12g} "
            f"result={'PASS' if comparison['pass'] else 'FAIL'}"
        )
    dg5 = next(comp for case, comp in cases if case.spec.name == "DG5")
    lines.extend(
        [
            "",
            "Active-Bound Result:",
            f"bound_c3={next(case.spec.c3_bound for case, _ in cases if case.spec.name == 'DG5')}",
            f"LM_c3={dg5['lm']['final'][3]:.17g}",
            f"GN_c3={dg5['gn']['final'][3]:.17g}",
            f"Python_c3={dg5['python']['final'][3]:.17g}",
            f"active_bound_result={'PASS' if dg5['active_bound_ok'] else 'FAIL'}",
            "",
            "PRODUCTION BUG FOUND",
            "file=src/nls_solver.c",
            "function=nls_solver_alloc",
            "case=CORE1 GN+QR",
            "root_cause=unsupported GN+QR allocation was accepted and failed only at solve time",
            "fix=reject GN+QR during allocation without fallback or algorithm switching",
            "before=allocation succeeded; nls_solve returned NLS_ERR_INVALID",
            "after=allocation returns NULL explicitly",
            "",
            "PRODUCTION BUG FOUND",
            "file=src/auglag_constrain.c",
            "function=auglag_init",
            "case=constraint descriptor ownership",
            "root_cause=constraint descriptor arrays were shallow-copied into the context",
            "fix=allocate and copy equality/inequality descriptor arrays; callback data remains explicitly borrowed",
            "before=caller descriptor arrays had to outlive AugLagContext and could otherwise dangle",
            "after=AugLagContext owns descriptor copies and frees them in auglag_destroy",
            "",
            f"Overall {'PASS' if overall else 'FAIL'}",
            "",
        ]
    )
    return "\n".join(lines)


def failure_report(error: BaseException) -> str:
    return "\n".join(
        [
            "Double Gaussian Minimize Cross-Validation Report",
            "",
            "Environment:",
            f"OS={platform.platform()}",
            f"architecture={platform.machine()}",
            f"Python={sys.version.split()[0]}",
            f"NumPy={np.__version__}",
            f"SciPy={scipy.__version__}",
            "",
            f"ERROR={type(error).__name__}: {error}",
            "Overall FAIL",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-runner", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not args.c_runner.is_file() or not os.access(args.c_runner, os.X_OK):
            raise FileNotFoundError(f"C runner is missing/not executable: {args.c_runner}")
        args.work_dir.mkdir(parents=True, exist_ok=True)
        results: list[tuple[CaseData, dict[str, Any]]] = []
        for spec in CASE_SPECS:
            case = make_case(spec)
            input_path = args.work_dir / f"{spec.name.lower()}.txt"
            write_c_input(case, input_path)
            c_records = run_c_case(args.c_runner, input_path)
            python_result, _ = run_python_case(case)
            comparison = compare_case(case, c_records, python_result)
            results.append((case, comparison))
            print(
                f"{spec.name}: input={'PASS' if comparison['input_ok'] else 'FAIL'} "
                f"result={'PASS' if comparison['result_ok'] else 'FAIL'} "
                f"case={'PASS' if comparison['pass'] else 'FAIL'}"
            )
        overall = all(comparison["pass"] for _, comparison in results)
        args.report.write_text(build_report(results, overall), encoding="utf-8")
        print(f"report={args.report}")
        print(f"Overall {'PASS' if overall else 'FAIL'}")
        return 0 if overall else 1
    except BaseException as exc:  # Preserve a report for any reference failure.
        args.report.write_text(failure_report(exc), encoding="utf-8")
        print(f"Double Gaussian cross-validation failed: {exc}", file=sys.stderr)
        print(f"report={args.report}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
