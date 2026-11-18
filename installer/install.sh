#!/usr/bin/env bash
# FaceGen installer for macOS / Linux. Run from the repository folder: ./installer/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (the Python manager FaceGen uses) from https://astral.sh/uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv python install 3.12
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[heic]"
exec .venv/bin/python installer/install.py "$@"
