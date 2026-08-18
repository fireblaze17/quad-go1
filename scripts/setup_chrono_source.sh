#!/usr/bin/env bash
set -euo pipefail

# Rebuild the source Chrono stack used by this project.
#
# Usage:
#   conda activate chrono-src
#   bash scripts/setup_chrono_source.sh
#
# For clean-room reproduction, prefer:
#   bash scripts/reproduce_chrono_source_env.sh chrono-src-test
#
# This intentionally does not install packaged pychrono. The project uses a
# source build because the packaged VSG bindings hit an ImGui/VSG ABI mismatch
# on this setup.

CHRONO_COMMIT="${CHRONO_COMMIT:-9faf13dd8f1128dd75ed233a9627027b0422c3f7}"
CHRONO_ROOT="${CHRONO_ROOT:-$HOME/chrono_builds}"
CHRONO_SRC="${CHRONO_SRC:-$CHRONO_ROOT/chrono}"
CHRONO_BUILD="${CHRONO_BUILD:-$CHRONO_ROOT/chrono-build}"
CHRONO_INSTALL="${CHRONO_INSTALL:-$CHRONO_ROOT/chrono-install}"
VSG_BUILD="${VSG_BUILD:-$CHRONO_ROOT/vsg_build}"
VSG_INSTALL="${VSG_INSTALL:-$CHRONO_ROOT/vsg-install}"
# Match the successful source build workflow. Using all WSL cores can exhaust
# memory during the full Chrono C++/CUDA build.
BUILD_JOBS="${BUILD_JOBS:-8}"
CLEAN_CHRONO_BUILD="${CLEAN_CHRONO_BUILD:-1}"

if [[ -z "${CONDA_PREFIX:-}" || -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "ERROR: activate the target conda env first." >&2
  echo "For normal setup:" >&2
  echo "  conda activate chrono-src" >&2
  echo "For clean-room testing:" >&2
  echo "  conda activate chrono-src-test" >&2
  exit 1
fi

CONDA_CMD="${CONDA_EXE:-conda}"
echo "Matching known-working chrono-src conda package builds..."
"$CONDA_CMD" install -y -c conda-forge \
  "git=2.55.0=pl5321h5685339_1" \
  "cmake=4.4.1=hc85cc9f_0" \
  "ninja=1.13.2=h171cf75_0" \
  "pkg-config=0.29.2=h4bc722e_1009" \
  "xorg-libxau=1.0.12=hb03c661_1" \
  "xorg-libxdmcp=1.1.5=hb03c661_1" \
  "xorg-xproto=7.0.31=hb9d3cd8_1008"
hash -r

for cmd in git cmake ninja python pkg-config swig; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: missing '$cmd' in PATH. Recreate/update the env from environment.yml." >&2
    exit 1
  fi
done

PKG_CONFIG_EXECUTABLE="$(command -v pkg-config)"
export PKG_CONFIG="$PKG_CONFIG_EXECUTABLE"

if ! pkg-config --exists xcb; then
  echo "ERROR: pkg-config cannot resolve xcb after applying the chrono-src package pins." >&2
  echo "Compare this env against the working chrono-src env with:" >&2
  echo "  conda list | grep -Ei 'pkg-config|libxcb|xorg-libxau|xorg-libxdmcp|xorg-xproto'" >&2
  exit 1
fi

if [[ ! -f /usr/include/irrlicht/irrlicht.h || ! -f /usr/lib/x86_64-linux-gnu/libIrrlicht.so ]]; then
  if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    echo "Installing system Irrlicht development files with apt..."
    sudo apt-get update
    sudo apt-get install -y libirrlicht-dev
  fi
fi

if [[ ! -f /usr/include/irrlicht/irrlicht.h || ! -f /usr/lib/x86_64-linux-gnu/libIrrlicht.so ]]; then
  echo "ERROR: system Irrlicht development files are still missing." >&2
  echo "Expected:" >&2
  echo "  /usr/include/irrlicht/irrlicht.h" >&2
  echo "  /usr/lib/x86_64-linux-gnu/libIrrlicht.so" >&2
  exit 1
fi

if [[ ! -d "$CONDA_PREFIX/targets/x86_64-linux/include/thrust" ]]; then
  echo "ERROR: Thrust headers were not found in the active conda environment." >&2
  echo "Expected:" >&2
  echo "  $CONDA_PREFIX/targets/x86_64-linux/include/thrust" >&2
  echo "Recreate/update the env from environment.yml so cuda-cccl_linux-64 is installed." >&2
  exit 1
fi

mkdir -p "$CHRONO_ROOT"

if [[ ! -d "$CHRONO_SRC/.git" ]]; then
  git clone https://github.com/projectchrono/chrono.git "$CHRONO_SRC"
fi

git -C "$CHRONO_SRC" fetch --tags origin
git -C "$CHRONO_SRC" checkout "$CHRONO_COMMIT"
git -C "$CHRONO_SRC" submodule update --init --recursive

mkdir -p "$VSG_BUILD"
cp "$CHRONO_SRC/contrib/build-scripts/linux/buildVSG.sh" "$VSG_BUILD/buildVSG.sh"

# Build only the libraries Chrono needs. The upstream helper also builds
# vsgExamples and writes to ~/.bashrc; neither is required for this project and
# the examples stage was one of the places setup failed while debugging.
python - "$VSG_BUILD/buildVSG.sh" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("#!/bin/bash", "#!/bin/bash\nset -euo pipefail", 1)
text = text.replace(
    'cmake -G "${BUILDSYSTEM}"',
    'cmake -G "${BUILDSYSTEM}" -DPKG_CONFIG_EXECUTABLE="${PKG_CONFIG}"',
)
text = text.replace(
    'cmake  -G "${BUILDSYSTEM}"',
    'cmake  -G "${BUILDSYSTEM}" -DPKG_CONFIG_EXECUTABLE="${PKG_CONFIG}"',
)
text = text.replace("BUILDDEBUG=ON", "BUILDDEBUG=OFF")
head = text.split("# --- vsgExamples", 1)[0]
head += '''

echo -e "\\n------------------------ Project VSG build complete\\n"
echo "VSG installed in: ${VSG_INSTALL_DIR}"
'''
path.write_text(head)
PY

(
  cd "$VSG_BUILD"
  bash buildVSG.sh "$VSG_INSTALL"
)

for required in \
  "$VSG_INSTALL/lib/cmake/vsg/vsgConfig.cmake" \
  "$VSG_INSTALL/lib/cmake/vsgXchange/vsgXchangeConfig.cmake" \
  "$VSG_INSTALL/lib/cmake/vsgImGui/vsgImGuiConfig.cmake"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: VSG dependency build did not install expected CMake config:" >&2
    echo "  $required" >&2
    echo "Check the first failing CMake/build error above, then remove the failed VSG build dir and rerun." >&2
    exit 1
  fi
done

if [[ "$CLEAN_CHRONO_BUILD" == "1" ]]; then
  rm -rf "$CHRONO_BUILD"
fi
cmake -S "$CHRONO_SRC" -B "$CHRONO_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CHRONO_INSTALL" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$VSG_INSTALL" \
  -DPKG_CONFIG_EXECUTABLE="$PKG_CONFIG_EXECUTABLE" \
  -DIrrlicht_INCLUDE_DIR="/usr/include/irrlicht" \
  -DIrrlicht_LIBRARY="/usr/lib/x86_64-linux-gnu/libIrrlicht.so" \
  -DTHRUST_INCLUDE_DIR="$CONDA_PREFIX/targets/x86_64-linux/include" \
  -Dvsg_DIR="$VSG_INSTALL/lib/cmake/vsg" \
  -DvsgXchange_DIR="$VSG_INSTALL/lib/cmake/vsgXchange" \
  -DvsgImGui_DIR="$VSG_INSTALL/lib/cmake/vsgImGui" \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
  -DBUILD_SHARED_LIBS=ON \
  -DCH_ENABLE_MODULE_PYTHON=ON \
  -DCH_ENABLE_MODULE_PARSERS=ON \
  -DCH_ENABLE_MODULE_IRRLICHT=ON \
  -DCH_ENABLE_MODULE_VEHICLE=ON \
  -DCH_ENABLE_MODULE_VEHICLE_MODELS=ON \
  -DCH_ENABLE_MODULE_VEHICLE_COSIM=ON \
  -DCH_ENABLE_MODULE_VSG=ON \
  -DCH_ENABLE_MODULE_FSI=ON \
  -DCH_ENABLE_MODULE_FSI_SPH=ON

cmake --build "$CHRONO_BUILD" --config Release -j"$BUILD_JOBS"
cmake --install "$CHRONO_BUILD" --config Release

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/chrono_source.sh" <<EOF
export _CHRONO_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export _CHRONO_OLD_PYTHONPATH="\${PYTHONPATH:-}"
export _CHRONO_OLD_VSG_FILE_PATH="\${VSG_FILE_PATH:-}"
export LD_LIBRARY_PATH="$CHRONO_INSTALL/lib:$VSG_INSTALL/lib:/usr/lib/wsl/lib:$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CHRONO_INSTALL/share/chrono/python:\${PYTHONPATH:-}"
export VSG_FILE_PATH="$VSG_INSTALL/share/vsgExamples"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/chrono_source.sh" <<'EOF'
export LD_LIBRARY_PATH="${_CHRONO_OLD_LD_LIBRARY_PATH:-}"
export PYTHONPATH="${_CHRONO_OLD_PYTHONPATH:-}"
export VSG_FILE_PATH="${_CHRONO_OLD_VSG_FILE_PATH:-}"
unset _CHRONO_OLD_LD_LIBRARY_PATH
unset _CHRONO_OLD_PYTHONPATH
unset _CHRONO_OLD_VSG_FILE_PATH
EOF

export LD_LIBRARY_PATH="$CHRONO_INSTALL/lib:$VSG_INSTALL/lib:/usr/lib/wsl/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CHRONO_INSTALL/share/chrono/python:${PYTHONPATH:-}"
export VSG_FILE_PATH="$VSG_INSTALL/share/vsgExamples"

python - <<'PY'
import pychrono
import pychrono.parsers
import pychrono.irrlicht
import pychrono.vsg3d
import pychrono.vehicle as veh
import pychrono.fsi as fsi

print("pychrono source build ok")
print("CRMTerrain:", hasattr(veh, "CRMTerrain"))
print("SPH VSG:", hasattr(fsi, "ChSphVisualizationVSG"))
PY

echo
echo "Done. Restart the shell or run: conda deactivate && conda activate $CONDA_DEFAULT_ENV"
