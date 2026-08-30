#!/usr/bin/env python3
"""Generate committed SciPy L-BFGS-B Golden Reference data for DG1-DG5."""

from __future__ import annotations

import argparse
import math
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

from double_gaussian_cases import (
    CASE_SPECS,
    CaseSpec,
    double_gaussian_model,
    generate_case_data,
    maximum_point,
)


REFERENCE_LEVEL = "mahalanobis_space_optimization_core"


@dataclass(frozen=True)
class ReferenceResult:
    params: np.ndarray
    sse: float
    rmse: float
    max_h: float
    max_v: float
    nit: int
    nfev: int


def solve_reference_case(spec: CaseSpec) -> ReferenceResult:
    data = generate_case_data(spec)

    def objective(params: np.ndarray) -> float:
        residual = data.observed - double_gaussian_model(params, data.h, data.v)
        return float(np.sum(residual * residual))

    bounds = [
        (None if not math.isfinite(lower) else lower, None)
        for lower in spec.lower
    ]
    result = minimize(
        objective,
        np.asarray(spec.initial, dtype=np.float64),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1.0e-10, "gtol": 1.0e-8},
    )
    params = np.asarray(result.x, dtype=np.float64)
    residual = data.observed - double_gaussian_model(params, data.h, data.v)
    sse = float(np.sum(residual * residual))
    max_h, max_v = maximum_point(params)
    if not result.success or not np.all(np.isfinite(params)) or not math.isfinite(sse):
        raise RuntimeError(f"{spec.name} reference solve failed: {result.message}")
    return ReferenceResult(
        params,
        sse,
        math.sqrt(sse / data.observed.size),
        max_h,
        max_v,
        int(result.nit),
        int(result.nfev),
    )


def c_float(value: float) -> str:
    if math.isinf(value):
        return "-INFINITY" if value < 0.0 else "INFINITY"
    return format(float(value), ".17g")


def c_vector(values: tuple[float, ...] | np.ndarray) -> str:
    return "{ " + ", ".join(c_float(value) for value in values) + " }"


def render_header(results: list[tuple[CaseSpec, ReferenceResult]]) -> str:
    lines = [
        "#ifndef DOUBLE_GAUSSIAN_REFERENCE_H",
        "#define DOUBLE_GAUSSIAN_REFERENCE_H",
        "",
        "/*",
        " * Generated reference. DO NOT EDIT MANUALLY.",
        " *",
        " * Reference: scipy.optimize.minimize, method=L-BFGS-B",
        " * maxiter=1000, ftol=1e-10, gtol=1e-8",
        f" * Python: {platform.python_version()}",
        f" * NumPy: {np.__version__}",
        f" * SciPy: {scipy.__version__}",
        f" * Reference level: {REFERENCE_LEVEL}",
        " */",
        "static const DoubleGaussianCase g_double_gaussian_cases[] = {",
    ]
    for spec, reference in results:
        lines.extend(
            [
                "    {",
                f'        .name = "{spec.name}",',
                f'        .title = "{spec.title}",',
                f"        .nx = {spec.nx},",
                f"        .ny = {spec.ny},",
                f"        .h_min = {c_float(spec.h_min)},",
                f"        .h_max = {c_float(spec.h_max)},",
                f"        .v_min = {c_float(spec.v_min)},",
                f"        .v_max = {c_float(spec.v_max)},",
                f"        .truth = {c_vector(spec.truth)},",
                f"        .initial = {c_vector(spec.initial)},",
                f"        .lower = {c_vector(spec.lower)},",
                f"        .mask_every = {spec.mask_every},",
                f"        .noise_scale = {c_float(spec.noise_scale)},",
                f"        .reference_params = {c_vector(reference.params)},",
                f"        .reference_sse = {c_float(reference.sse)},",
                f"        .reference_rmse = {c_float(reference.rmse)},",
                f"        .reference_max_h = {c_float(reference.max_h)},",
                f"        .reference_max_v = {c_float(reference.max_v)},",
                f"        .reference_nit = {reference.nit},",
                f"        .reference_nfev = {reference.nfev},",
                f"        .prediction_rel_tol = {c_float(spec.prediction_rel_tol)},",
                f"        .max_point_tol = {c_float(spec.max_point_tol)},",
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            "#define DOUBLE_GAUSSIAN_CASE_COUNT \\",
            "    (sizeof(g_double_gaussian_cases) / sizeof(g_double_gaussian_cases[0]))",
            "",
            "#endif",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the generated C header instead of printing it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = [(spec, solve_reference_case(spec)) for spec in CASE_SPECS]
    header = render_header(results)
    if args.output is None:
        print(header, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(header, encoding="utf-8")
        print(f"GOLDEN_REFERENCE_WRITTEN={args.output}")
        print(
            f"Python={platform.python_version()} NumPy={np.__version__} "
            f"SciPy={scipy.__version__}"
        )
        for spec, reference in results:
            print(
                f"{spec.name}: sse={reference.sse:.17g} "
                f"rmse={reference.rmse:.17g} nit={reference.nit} "
                f"nfev={reference.nfev}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
