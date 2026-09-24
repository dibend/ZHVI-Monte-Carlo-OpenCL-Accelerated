# Zillow ZHVI Monte Carlo Simulator

## https://MicheleDiBenedetto.net

Monte Carlo simulations for projecting the **Zillow Home Value Index (ZHVI)** for any U.S. ZIP code. The app uses an embedded OpenCL kernel through PyOpenCL when a compatible device is available, and NumPy on the CPU otherwise. Results are displayed in a Gradio and Plotly interface.

## Features
- **Optional OpenCL Acceleration** – Prefer a double-precision GPU or OpenCL CPU device when PyOpenCL and a driver are installed. Automatically use NumPy on the CPU if OpenCL is unavailable or fails.
- **Interactive Interface** – Select a ZIP code, history window, and simulation parameters in a Gradio UI.
- **Real Zillow Data** – Automatically fetches the latest ZHVI dataset.
- **Plotly Visuals** – View historical trends, simulated paths, and distribution histograms.

## Installation
1. Install Python 3.8+.
2. Clone the repository:
   ```bash
   git clone https://github.com/dibend/ZHVI-Monte-Carlo-OpenCL-Accelerated.git
   cd ZHVI-Monte-Carlo-OpenCL-Accelerated
   ```
3. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. To enable OpenCL on a computer with a compatible driver and a double-precision device, install PyOpenCL:
   ```bash
   pip install -r requirements-opencl.txt
   ```
   The OpenCL kernel is embedded in `app.py`; no separate kernel file is needed. If the optional package or device is unavailable, the regular install still runs on the CPU.

## Debian / ChromeOS Linux virtual environment

From the repository root:

```bash
bash scripts/setup-opencl-venv.sh
source .venv/bin/activate
python app.py
```

The setup installs a CPU OpenCL runtime (PoCL) and checks every discovered OpenCL device for double precision, which this app's embedded kernel requires. A GPU appears only if ChromeOS passes it through to the Linux VM and a compatible OpenCL driver is installed there. When no suitable OpenCL device is available, the app uses its NumPy CPU path. The system ICD driver is installed through apt; Python dependencies stay in `.venv`. To check devices later, run `clinfo -l` and `python -c 'import pyopencl as cl; print([(p.name, [d.name for d in p.get_devices()]) for p in cl.get_platforms()])'`.

## Usage
Run the application with:
```bash
python app.py
```
This launches a local Gradio interface where you can choose a ZIP code and simulation settings. The summary reports which backend ran. To share the interface publicly, set `share=True` in `demo.launch` inside `app.py`.

## Example
Try the default ZIP code **07974** (New Providence, NJ) for a quick demo, or enter any five-digit U.S. ZIP code. Adjust the number of paths to trade off accuracy vs. runtime.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
