#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_JOBS="${BUILD_JOBS:-4}"
REPORT_PATH="$BUILD_DIR/test_report_double_gaussian_minimize.txt"

mkdir -p "$BUILD_DIR"
printf '%s\n' \
    'Double Gaussian Minimize Cross-Validation Report' \
    'status=BUILD_OR_TEST_IN_PROGRESS' \
    'Overall FAIL' > "$REPORT_PATH"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"
ctest --test-dir "$BUILD_DIR" --output-on-failure

PYTHON_CMD=()
if [[ -n "${PYTHON:-}" ]]; then
    if "$PYTHON" -c 'import numpy, scipy' >/dev/null 2>&1; then
        PYTHON_CMD=("$PYTHON")
    fi
elif python3 -c 'import numpy, scipy' >/dev/null 2>&1; then
    PYTHON_CMD=(python3)
elif [[ -x /opt/homebrew/Caskroom/miniconda/base/envs/nls-verify/bin/python ]] &&
     /opt/homebrew/Caskroom/miniconda/base/envs/nls-verify/bin/python \
        -c 'import numpy, scipy' >/dev/null 2>&1; then
    PYTHON_CMD=(/opt/homebrew/Caskroom/miniconda/base/envs/nls-verify/bin/python)
elif command -v conda >/dev/null 2>&1 &&
     conda run -n nls-verify python -c 'import numpy, scipy' >/dev/null 2>&1; then
    PYTHON_CMD=(conda run -n nls-verify python)
fi

if [[ ${#PYTHON_CMD[@]} -eq 0 ]]; then
    echo "MISSING_PYTHON_DEPENDENCY: Python with NumPy and SciPy is required"
    printf '%s\n' \
        'Double Gaussian Minimize Cross-Validation Report' \
        'MISSING_PYTHON_DEPENDENCY: Python with NumPy and SciPy is required' \
        'Overall FAIL' > "$REPORT_PATH"
    exit 1
fi

NLS_CTEST_STATUS=PASS \
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/tools/double_gaussian_minimize_reference.py" \
    --c-runner "$BUILD_DIR/test_double_gaussian" \
    --work-dir "$BUILD_DIR/double_gaussian_inputs" \
    --report "$REPORT_PATH"

echo "Build, CTest, SciPy reference, cross-validation, and report: PASS"
echo "Report: $REPORT_PATH"
