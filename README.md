# Zillow ZHVI Monte Carlo Simulator

## https://MicheleDiBenedetto.net

Monte Carlo simulations for projecting the **Zillow Home Value Index (ZHVI)** for any U.S. ZIP code. The app uses an embedded OpenCL kernel through PyOpenCL when a compatible device is available, and NumPy on the CPU otherwise. Results are displayed in a Gradio and Plotly interface.

## Features
- **Optional OpenCL Acceleration** – Choose an OpenCL platform and an individual GPU, CPU, or accelerator. Automatic selection tries GPUs first, then OpenCL CPUs and other devices. If OpenCL is unavailable or fails, the app uses NumPy on the CPU.
- **Device Controls** – Refresh detected devices, override automatic selection, or select **CPU (NumPy)** at any time. The summary reports the device and precision actually used, or the reason for CPU fallback.
- **Memory-aware GPU Batches** – OpenCL work is split into batches based on the selected device's memory limits. Devices without double precision use float32; capable devices retain float64.
- **Interactive Interface** – Select a ZIP code, history window, and simulation parameters in a Gradio UI.
- **Real Zillow Data** – Automatically fetches the latest ZHVI dataset.
- **Plotly Visuals** – View historical trends, simulated paths, and distribution histograms.

## Installation
1. Install Python 3.10+ (required by Gradio 6).
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
5. To enable OpenCL on a computer with a compatible OpenCL driver, install PyOpenCL:
   ```bash
   pip install -r requirements-opencl.txt
   ```
   The OpenCL kernel is embedded in `app.py`; no separate kernel file is needed. If the optional package or device is unavailable, the regular install still runs on the CPU. PyOpenCL does not install your GPU vendor's OpenCL driver. See the [PyOpenCL installation guide](https://documen.tician.de/pyopencl/misc.html) for driver options. On supported systems, `python -m pip install 'pyopencl[pocl]'` also provides a CPU OpenCL runtime; ordinary NumPy CPU mode needs neither package nor driver.

## Usage
Run the application with:
```bash
python app.py
```
This launches a Gradio interface where you can choose a ZIP code and simulation settings. The existing `share=True` setting also creates a public Gradio link; set `share=False` in `demo.launch` inside `app.py` for local-only use.

### Password protection

Set a username and enter your password at the hidden Bash prompt before launching:
```bash
export GRADIO_USERNAME=david
read -rsp 'Gradio password: ' GRADIO_PASSWORD
echo
export GRADIO_PASSWORD
python app.py
```
The login applies to the local app and its public share link. The password stays out of the source code and shell history. Both variables must be non-empty; partial or empty settings stop startup with an explanation. When neither variable is set, the app keeps its existing behavior without a login. To disable authentication again, run `unset GRADIO_USERNAME GRADIO_PASSWORD` and restart. Use `python app.py` so that the authentication configured by the launch function is applied.

### Running simulations

1. Leave **Compute backend** on **OpenCL if available (CPU fallback)** for automatic acceleration, or select **CPU (NumPy)** to force CPU execution without OpenCL.
2. Choose an **OpenCL platform** to filter the **OpenCL device** dropdown, then choose a specific device. Leave both on automatic to try all detected devices, preferring GPUs. Each simulation uses one device.
3. Click **Refresh devices** after connecting a device. If a selection disappears, the dropdown resets to automatic and explains the change. Restart the app after installing PyOpenCL or new drivers.
4. Click **Run Simulation**. **Summary & Status** reports the actual backend, device, and precision. A selected device that fails falls back to NumPy; it does not silently switch to another OpenCL device. Automatic selection can try the other available devices before falling back.

Devices belong to the machine running Python, not the browser. A Chromebook Linux VM can only use devices exposed to that VM. If none are available, CPU mode still works. Float32 devices can give slightly different numerical results. OpenCL accelerates path evolution; random-number generation, statistics, and plotting still run on the CPU, so smaller runs may be faster in NumPy. Full results still need host RAM even though GPU buffers are batched.

### Shutdown semaphore warning

The example presets use a selectable `gr.Dataset` to fill the same four inputs. This avoids the unused CSV cache logger and multiprocessing semaphore created by `gr.Examples`, which can contribute to a `resource_tracker: ... leaked semaphore` warning at shutdown. Python 3.14's default `forkserver` start method on Linux makes such locks named resources tracked by `resource_tracker`.

Other dependencies or an abrupt process exit can still produce this warning. If the app exits unexpectedly, inspect the traceback or terminal output before the warning for the underlying failure.

## Tests
Run `python -m unittest discover -s tests -v` after installing the app dependencies. The suite covers device discovery/selection, driver failures, CPU fallback, the Gradio callbacks and presets, login configuration, and named-semaphore creation under `forkserver`. With a usable OpenCL runtime installed, it also executes both float32 and float64 kernels and compares seeded, batched results with NumPy. OpenCL-only checks skip when no compatible device is available.

## Example
Try the default ZIP code **07974** (New Providence, NJ) for a quick demo, or enter any five-digit U.S. ZIP code. Adjust the number of paths to trade off accuracy vs. runtime.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
