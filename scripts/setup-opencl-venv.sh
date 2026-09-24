#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null; then
  echo "Python 3 is required." >&2
  exit 1
fi

if command -v apt-get >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y python3-venv clinfo ocl-icd-libopencl1 pocl-opencl-icd
else
  echo "Install an OpenCL ICD driver and clinfo for your OS before running the app."
fi

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-opencl.txt

echo
echo "OpenCL platforms reported by clinfo:"
if command -v clinfo >/dev/null; then clinfo -l || true; fi

python - <<'PY'
import pyopencl as cl

try:
    platforms = cl.get_platforms()
except cl.Error as exc:
    print(f"No OpenCL platform available: {exc}")
    print("The app will use its NumPy CPU fallback.")
else:
    usable = []
    for platform in platforms:
        try:
            devices = platform.get_devices()
        except cl.Error:
            continue
        for device in devices:
            fp64 = bool(device.double_fp_config) or "cl_khr_fp64" in device.extensions.split()
            print(f"{platform.name}: {device.name} | double precision: {fp64}")
            if fp64:
                usable.append(device.name)
    if usable:
        print(f"App-compatible OpenCL device(s): {', '.join(usable)}")
    else:
        print("No double-precision OpenCL device found; the app will use NumPy.")
PY

echo
echo "Ready. Run: source .venv/bin/activate && python app.py"
