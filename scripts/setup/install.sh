#!/usr/bin/env bash

set -euo pipefail

AMBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

git -C "$AMBENCH_ROOT" submodule update --init --recursive ext/pyroki ext/acados

uv pip install -e "$AMBENCH_ROOT/ext/pyroki"
uv pip install "numpy==1.26.0" "jax==0.4.28" "jaxlib==0.4.28"

cmake -S "$AMBENCH_ROOT/ext/acados" \
  -B "$AMBENCH_ROOT/ext/acados/build" \
  -DACADOS_WITH_QPOASES=ON
cmake --build "$AMBENCH_ROOT/ext/acados/build" --target install --parallel 4
uv pip install -e "$AMBENCH_ROOT/ext/acados/interfaces/acados_template"

uv pip install -e "$AMBENCH_ROOT/source/ambench"
uv pip install -e "$AMBENCH_ROOT/source/ambench_learn"

echo "AM-Bench and its Pyroki and acados dependencies are installed."
