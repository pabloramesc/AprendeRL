#!/usr/bin/env bash

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
notebook_dir="$repo_root/examples"
cell_timeout="${NOTEBOOK_CELL_TIMEOUT:-900}"

if command -v uv >/dev/null 2>&1; then
    jupyter=(uv run jupyter)
elif [[ -x "$repo_root/.venv/bin/jupyter" ]]; then
    jupyter=("$repo_root/.venv/bin/jupyter")
elif command -v jupyter >/dev/null 2>&1; then
    jupyter=(jupyter)
else
    echo "error: Jupyter is not available; install the project dependencies first" >&2
    exit 1
fi

shopt -s nullglob
notebooks=("$notebook_dir"/*.ipynb)

if ((${#notebooks[@]} == 0)); then
    echo "error: no example notebooks found in $notebook_dir" >&2
    exit 1
fi

failed=()

for notebook in "${notebooks[@]}"; do
    relative_path="${notebook#"$repo_root/"}"
    echo "Executing $relative_path"

    if ! (
        cd "$repo_root"
        "${jupyter[@]}" nbconvert \
            --to notebook \
            --execute \
            --inplace \
            --ExecutePreprocessor.timeout="$cell_timeout" \
            "$notebook"
    ); then
        failed+=("$relative_path")
    fi
done

if ((${#failed[@]} > 0)); then
    echo >&2
    echo "Failed notebooks:" >&2
    printf '  - %s\n' "${failed[@]}" >&2
    exit 1
fi

echo
echo "Regenerated outputs for ${#notebooks[@]} example notebooks."
