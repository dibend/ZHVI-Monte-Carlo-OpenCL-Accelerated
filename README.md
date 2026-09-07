# Zillow ZHVI Monte Carlo Simulator

## https://MicheleDiBenedetto.net

CPU-only Monte Carlo simulations for projecting the **Zillow Home Value Index (ZHVI)** for any U.S. ZIP code. The app downloads ZHVI data, computes historical drift and volatility, and uses NumPy to generate thousands of future price paths. Results are displayed in an interactive browser UI powered by Gradio and Plotly.

## Features
- **CPU Only** – No GPU, OpenCL runtime, OpenCL drivers, or PyOpenCL installation required.
- **Same Monte Carlo Model** – Uses the same monthly Geometric Brownian Motion formula, `float64` calculations, simulation-path structure, and summary statistics as the original OpenCL version.
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

## Usage
Run the application with:
```bash
python app.py
```
This launches the Gradio interface where you can choose a ZIP code and simulation settings. The app plots historical ZHVI data and generates Monte Carlo projections entirely on the CPU.

The app keeps the original path-count range up to 10,000,000 for functional compatibility. Very large CPU simulations can require substantial RAM and take much longer than the old GPU/OpenCL backend; the app reports a clear memory error when the requested result array cannot fit.

## Example
Try the default ZIP code **07974** (New Providence, NJ) for a quick demo, or enter any five-digit U.S. ZIP code. Adjust the number of paths to trade off accuracy, RAM usage, and runtime.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
