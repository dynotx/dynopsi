#!/bin/bash
set -euo pipefail

CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean|-c) CLEAN=true ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

dynopsi check "$REPO_ROOT/tests/test_files/test_simulation.yaml"

dynopsi sample "$REPO_ROOT/tests/test_files/test_simulation.yaml" --num_samples 2 --batch_size 2

final_state_count="$(
    find ./tmp/test_cli_output \
        -type f \
        \( -name 'X_T_final_state_*.cif' -o -name 'X_T_final_state_*.pdb' \) \
        -size +0c \
        -print |
        wc -l
)"
if [ "$final_state_count" -ne 8 ]; then
    echo "Expected 8 non-empty final-state structures under ./tmp/test_cli_output; found $final_state_count" >&2
    exit 1
fi

if [ "$CLEAN" = true ]; then
    rm -rf ./tmp/test_cli_output
    echo "Cleaned up ./tmp/test_cli_output"
fi
