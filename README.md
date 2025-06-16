# Zillow ZHVI Monte Carlo Simulator

GPU accelerated Monte Carlo simulations for projecting the **Zillow Home Value Index (ZHVI)** for any U.S. ZIP code. The app downloads ZHVI data, computes historical drift and volatility, and then uses an OpenCL kernel to generate thousands of future price paths. Results are displayed in an interactive browser UI powered by Gradio and Plotly.

## Features
- **OpenCL Acceleration** – Run Monte Carlo simulations on your GPU/CPU for huge speedups.
- **Interactive Interface** – Select a ZIP code, history window, and simulation parameters in a Gradio UI.
- **Real Zillow Data** – Automatically fetches the latest ZHVI dataset.
- **Plotly Visuals** – View historical trends, simulated paths, and distribution histograms.

## Installation
1. Install Python 3.8+ and ensure your system has OpenCL drivers (GPU or CPU).
2. Clone the repository:
   ```bash
   git clone https://github.com/yourname/ZHVI-Monte-Carlo-OpenCL-Accelerated.git
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
This launches a local Gradio interface where you can choose a ZIP code and simulation settings. The app will plot historical ZHVI data and generate Monte Carlo projections. To share the interface publicly, set `share=True` in `demo.launch` inside `app.py`.

## Example
Try the default ZIP code **07974** (New Providence, NJ) for a quick demo, or enter any five-digit U.S. ZIP code. Adjust the number of paths to trade off accuracy vs. runtime.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
