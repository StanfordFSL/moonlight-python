#!/usr/bin/env bash
set -euo pipefail

# Build manylinux wheels locally using cibuildwheel.
# Requires Docker to be running.

pip install cibuildwheel
cibuildwheel --output-dir wheelhouse
