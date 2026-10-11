#!/usr/bin/env bash
# Kian launcher for Linux/macOS.
set -e
cd "$(dirname "$0")/.."
python3 -m venv .venv 2>/dev/null || true
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet numpy
if [ ! -f kian_home/being.json ]; then
  ./.venv/bin/python -m kian --home kian_home birth
fi
exec ./.venv/bin/python -m kian --home kian_home "${@:-live}"
