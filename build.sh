#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_JOBS="${BUILD_JOBS:-4}"
NLS_PYTHON_ENV="${NLS_PYTHON_ENV:-nls-verify}"
COMMAND="${1:-build}"
PYTHON_CMD=()

select_project_python()
{
    if [[ -n "${PYTHON:-}" ]]; then
        PYTHON_CMD=("$PYTHON")
    elif command -v conda >/dev/null 2>&1; then
        PYTHON_CMD=(conda run --no-capture-output -n "$NLS_PYTHON_ENV" python)
    else
        echo "MISSING_PROJECT_PYTHON: set PYTHON or install Conda env $NLS_PYTHON_ENV" >&2
        return 1
    fi
}

require_python_modules()
{
    local import_statement="$1"

    select_project_python
    if ! "${PYTHON_CMD[@]}" -c "$import_statement" >/dev/null 2>&1; then
        echo "MISSING_PYTHON_DEPENDENCY in $NLS_PYTHON_ENV: $import_statement" >&2
        return 1
    fi
}

build_project()
{
    mkdir -p "$BUILD_DIR"
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"
    ctest --test-dir "$BUILD_DIR" --output-on-failure
    echo "Build and CTest: PASS"
}

case "$COMMAND" in
    build)
        build_project
        ;;
    update-reference)
        require_python_modules "import numpy, scipy"
        "${PYTHON_CMD[@]}" tools/double_gaussian_minimize_reference.py \
            --output tests/double_gaussian_reference.h
        echo "GOLDEN_REFERENCE_UPDATED"
        echo "Review the reference diff, then run ./build.sh"
        ;;
    visualize)
        require_python_modules "import numpy, scipy, matplotlib"
        "${PYTHON_CMD[@]}" tools/plot_gaussian_fit.py
        ;;
    *)
        echo "usage: $0 [build|update-reference|visualize]" >&2
        exit 2
        ;;
esac
