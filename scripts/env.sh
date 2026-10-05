# Source this file. It loads the OpenRouter env, sets STABLEWM_HOME, and
# activates le-wm/.venv when that venv exists. It does not print secrets.
# shellcheck shell=sh

if [ -z "${HOME:-}" ]; then
  echo "HOME is unset" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck disable=SC1091
. "$HOME/.config/openrouter/env"
set +a

export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.stable-wm}"

_src="${JEPA_ENV_SH:-}"
if [ -z "$_src" ]; then
  if [ -n "${BASH_VERSION:-}" ] && [ -n "${BASH_SOURCE:-}" ]; then
    _src=$BASH_SOURCE
  elif [ -n "${ZSH_VERSION:-}" ]; then
    _src=${(%):-%x}
  else
    _src=$0
  fi
fi
_here=$(CDPATH= cd -- "$(dirname -- "$_src")" && pwd)
_root=$(CDPATH= cd -- "$_here/.." && pwd)

if [ -f "$_root/le-wm/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$_root/le-wm/.venv/bin/activate"
fi
unset _src _here _root
