#!/bin/sh
# Launch one closed agent cycle. Does not install packages or download data.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
export JEPA_ENV_SH="$ROOT/scripts/env.sh"
# shellcheck disable=SC1091
. "$JEPA_ENV_SH"
exec python "$ROOT/scripts/cycle.py" launch "$@"
