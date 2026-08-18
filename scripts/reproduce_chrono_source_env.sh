#!/usr/bin/env bash
set -euo pipefail

# Create a fresh conda env and build the exact source Chrono stack used here.
# This wrapper is the clean-room reproduction loop: it keeps all Chrono/VSG
# source, build, and install directories away from the working chrono-src build.

ENV_NAME="${1:-chrono-src-test}"
BUILD_ROOT="${2:-$HOME/${ENV_NAME}_builds}"
CLEAN_ENV="${CLEAN_ENV:-0}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/environment.yml"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: cannot find environment.yml at $ENV_FILE" >&2
  exit 1
fi

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_ROOT="$(dirname "$(dirname "$CONDA_EXE")")"
  if [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
  fi
fi

if ! type conda 2>/dev/null | grep -q "function"; then
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
fi

if ! type conda 2>/dev/null | grep -q "function"; then
  echo "ERROR: conda shell activation function is unavailable." >&2
  echo "Run: source <conda>/etc/profile.d/conda.sh" >&2
  exit 1
fi

if [[ "$CLEAN_ENV" == "1" ]] && conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Removing existing conda env: $ENV_NAME"
  conda env remove -y -n "$ENV_NAME"
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Creating conda env: $ENV_NAME"
  conda env create -f "$ENV_FILE" -n "$ENV_NAME"
else
  echo "Updating existing conda env: $ENV_NAME"
  conda env update -f "$ENV_FILE" -n "$ENV_NAME" --prune
fi

conda activate "$ENV_NAME"

case "$BUILD_ROOT" in
  "$HOME/chrono_builds"|"$HOME/chrono_builds/"*)
    echo "ERROR: refusing to use working build root as clean-room BUILD_ROOT: $BUILD_ROOT" >&2
    echo "Use a separate root such as: $HOME/chrono_repro_test" >&2
    exit 1
    ;;
esac

if [[ "$CLEAN_BUILD" == "1" ]]; then
  echo "Cleaning isolated build root: $BUILD_ROOT"
  rm -rf "$BUILD_ROOT"
else
  export CLEAN_CHRONO_BUILD="${CLEAN_CHRONO_BUILD:-0}"
fi

mkdir -p "$BUILD_ROOT"

export CHRONO_ROOT="$BUILD_ROOT"
export CHRONO_SRC="$BUILD_ROOT/chrono"
export CHRONO_BUILD="$BUILD_ROOT/chrono-build"
export CHRONO_INSTALL="$BUILD_ROOT/chrono-install"
export VSG_BUILD="$BUILD_ROOT/vsg_build"
export VSG_INSTALL="$BUILD_ROOT/vsg-install"

echo "Running source Chrono setup in isolated root:"
echo "  env:        $ENV_NAME"
echo "  build root: $BUILD_ROOT"
echo

bash "$REPO_ROOT/scripts/setup_chrono_source.sh"

echo
echo "Clean-room source Chrono environment is ready:"
echo "  conda activate $ENV_NAME"
echo "  build root: $BUILD_ROOT"
