#!/usr/bin/env bash

AMBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export JAX_PLATFORMS=cpu
export ACADOS_SOURCE_DIR="$AMBENCH_ROOT/ext/acados"
if [[ -n "${LD_LIBRARY_PATH-}" ]]; then
  export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/build:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:$ACADOS_SOURCE_DIR/build"
fi

unset AMBENCH_ROOT
