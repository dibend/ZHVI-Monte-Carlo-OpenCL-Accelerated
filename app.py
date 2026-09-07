import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time

# --- Constants ---
# URL for Zillow Home Value Index (ZHVI) Single-Family+Condo monthly data by Zip Code
# Check Zillow Research Data page for latest URLs if this breaks: https://www.zillow.com/research/data/
ZILLOW_DATA_URL = 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
MIN_YEAR = 2000  # Earliest year for Zillow data

# --- Default UI Values ---
DEFAULT_ZIP_CODE = "07974"  # New Providence, NJ
DEFAULT_HIST_PERIOD = "10y"  # Period for calculating historical mu/sigma
DEFAULT_SIM_MONTHS = 120  # Simulate 10 years ahead (12 * 10)
DEFAULT_NUM_PATHS = 100000  # Default simulation paths

# Keep temporary random-number arrays bounded so CPU-only runs do not need an
# additional giant allocation on top of the final simulation result array.
CPU_RANDOM_TARGET_BYTES = 256 * 1024 * 1024


# --- Helper Functions ---
def run_monte_carlo_simulation_cpu(s0, mu, sigma, sim_steps, num_paths):
    """
    Runs the same Geometric Brownian Motion Monte Carlo model as the original
    OpenCL kernel, using NumPy on the CPU only.

    Args:
        s0 (float): Initial asset value.
        mu (float): Drift per time step (monthly).
        sigma (float): Volatility per time step (monthly).
        sim_steps (int): Number of months to simulate.
        num_paths (int): Number of simulation paths.

    Returns:
        numpy.ndarray: Shape (num_paths, sim_steps + 1), including s0 at column 0.
    """
    print(f"Preparing CPU simulation: {num_paths} paths, {sim_steps} steps...")
    start_time = time.time()

    np_dtype = np.float64
    s0 = np_dtype(s0)
    mu = np_dtype(mu)
    sigma = np_dtype(sigma)

    # Same formula as the OpenCL kernel. dt = 1 because mu/sigma are monthly.
    drift_term = (mu - np_dtype(0.5) * sigma * sigma)
    vol_term = sigma

    try:
        sim_paths = np.empty((num_paths, sim_steps + 1), dtype=np_dtype)
    except (MemoryError, ValueError) as e:
        required_gib = (num_paths * (sim_steps + 1) * np.dtype(np_dtype).itemsize) / (1024 ** 3)
        raise RuntimeError(
            f"Not enough RAM for the requested simulation result array "
            f"(approximately {required_gib:.2f} GiB required before plotting overhead)."
        ) from e

    sim_paths[:, 0] = s0

    # Preserve path-major random-number ordering used by the original flattened
    # OpenCL input while limiting the temporary random array size.
    bytes_per_path = max(sim_steps, 1) * np.dtype(np_dtype).itemsize
    chunk_paths = max(1, min(num_paths, CPU_RANDOM_TARGET_BYTES // bytes_per_path))

    for start in range(0, num_paths, chunk_paths):
        stop = min(start + chunk_paths, num_paths)
        rows = stop - start

        try:
            rand_normals = np.random.randn(rows, sim_steps).astype(np_dtype, copy=False)
            growth_factors = np.exp(drift_term + vol_term * rand_normals)
            np.cumprod(growth_factors, axis=1, out=growth_factors)
            growth_factors *= s0
            np.maximum(growth_factors, np_dtype(0.01), out=growth_factors)
            sim_paths[start:stop, 1:] = growth_factors
        except MemoryError as e:
            raise RuntimeError(
                "Not enough RAM for the requested CPU simulation. "
                "Try fewer paths or fewer simulation months."
            ) from e

    elapsed = time.time() - start_time
    print(f"CPU simulation finished in {elapsed:.3f} seconds.")
    return sim_paths


# --- Zillow Data Functions ---
# Global variable to cache the loaded Zillow DataFrame
zillow_df_cache = None
cache_load_time = None


def load_zillow_data_cached(max_age_hours=24):
    """
    Loads the Zillow ZHVI data from the URL, caching it globally to avoid
    repeated downloads within a session or defined period.
    """
    global zillow_df_cache, cache_load_time
    now = time.time()

    if zillow_df_cache is not None and cache_load_time is not None:
        age_seconds = now - cache_load_time
        if age_seconds < max_age_hours * 3600:
            print("Using cached Zillow data.")
            return zillow_df_cache.copy()

    print(f"Loading Zillow data from {ZILLOW_DATA_URL}...")
    try:
        df = pd.read_csv(ZILLOW_DATA_URL)
        df['RegionName'] = df['RegionName'].astype(str).str.zfill(5)
        zillow_df_cache = df
        cache_load_time = now
        print("Zillow data loaded and cached.")
        return df.copy()
    except Exception as e:
        print(f"Error loading Zillow data: {e}")
        raise gr.Error(f"Failed to load data from Zillow URL. Error: {e}")


def fetch_and_prepare_zillow_data(zip_code, hist_period_str):
    """
    Filters Zillow data for a single ZIP code, selects the requested historical
    period, calculates monthly log returns, and derives monthly drift/volatility.
    """
    print(f"Fetching & preparing Zillow data for ZIP: {zip_code}, History: {hist_period_str}")
    df_zillow = load_zillow_data_cached()
    zip_code_str = str(zip_code).strip().zfill(5)

    df_zip = df_zillow[df_zillow['RegionName'] == zip_code_str]
    if df_zip.empty:
        raise ValueError(f"Data not found for ZIP code '{zip_code_str}'.")

    first_date_col_index = -1
    for i, col_name in enumerate(df_zillow.columns):
        if isinstance(col_name, str) and (col_name.count('-') == 1 or col_name.count('-') == 2):
            try:
                pd.to_datetime(col_name, errors='raise')
                first_date_col_index = i
                break
            except (ValueError, TypeError):
                continue

    if first_date_col_index == -1:
        raise ValueError("Could not identify date columns in Zillow CSV.")

    date_cols = df_zillow.columns[first_date_col_index:]
    series = df_zip[date_cols].iloc[0].copy()
    series.index = pd.to_datetime(series.index)
    series = series.dropna()

    if series.empty:
        raise ValueError(f"No valid price data points found for ZIP code '{zip_code_str}'.")

    last_date = series.index[-1]
    hist_start_date = series.index[0]

    if hist_period_str != 'max':
        try:
            offset = pd.tseries.frequencies.to_offset(
                hist_period_str.replace('y', 'Y').replace('m', 'M')
            )
            calculated_start = last_date - offset
            min_data_date = series.index[0]
            min_allowed_date = pd.Timestamp(year=MIN_YEAR, month=1, day=1)
            hist_start_date = max(calculated_start, min_data_date, min_allowed_date)
            series = series[series.index >= hist_start_date]
        except Exception as e:
            print(
                f"Warning: Could not parse period '{hist_period_str}', "
                f"using max history. Error: {e}"
            )
            hist_start_date = series.index[0]

    print(
        f"Using historical data from {series.index[0].strftime('%Y-%m-%d')} "
        f"to {last_date.strftime('%Y-%m-%d')} for calculations ({len(series)} points)."
    )

    if len(series) < 12:
        raise ValueError(
            f"Insufficient historical data ({len(series)} months) for ZIP "
            f"{zip_code_str} in period '{hist_period_str}'. Need at least 12."
        )

    log_returns = np.log(series / series.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Could not calculate log returns (maybe only 1 data point?).")

    mu_monthly = log_returns.mean()
    sigma_monthly = log_returns.std()
    s0 = series.iloc[-1]

    if sigma_monthly <= 0 or pd.isna(sigma_monthly) or pd.isna(mu_monthly) or pd.isna(s0):
        raise ValueError(
            f"Calculated parameters invalid: mu={mu_monthly}, "
            f"sigma={sigma_monthly}, s0={s0}. Check historical data."
        )

    print(
        f"Params calculated: s0=${s0:,.0f}, mu_monthly={mu_monthly:.6f}, "
        f"sigma_monthly={sigma_monthly:.6f}"
    )
    return series, s0, mu_monthly, sigma_monthly


def create_zillow_plots(hist_series, sim_paths, zip_code, sim_months):
    """Creates the same three Plotly figures as the original application."""
    print("Generating plots...")

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Scatter(
            x=hist_series.index,
            y=hist_series,
            mode='lines',
            name=f'{zip_code} Historical ZHVI',
        )
    )
    fig_hist.update_layout(
        title=f"Zillow Home Value Index (ZHVI) - ZIP: {zip_code}",
        xaxis_title="Date",
        yaxis_title="ZHVI ($)",
        template="plotly_dark",
    )

    fig_sim = go.Figure()
    num_paths_to_plot = min(sim_paths.shape[0], 1000)
    last_hist_date = hist_series.index[-1]
    start_sim_date = last_hist_date + pd.DateOffset(months=1)
    sim_dates_full_path = pd.date_range(
        start=start_sim_date, periods=sim_months, freq='ME'
    )
    sim_dates_plotting = pd.Index([last_hist_date]).union(sim_dates_full_path)

    for i in range(num_paths_to_plot):
        fig_sim.add_trace(
            go.Scatter(
                x=sim_dates_plotting,
                y=sim_paths[i, :],
                mode='lines',
                line=dict(width=0.5),
                showlegend=False,
                opacity=0.1,
            )
        )

    mean_path = sim_paths.mean(axis=0)
    fig_sim.add_trace(
        go.Scatter(
            x=sim_dates_plotting,
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(color='red', width=2),
        )
    )
    fig_sim.update_layout(
        title=f"{zip_code} ZHVI Monte Carlo Simulations ({sim_paths.shape[0]:,} Paths)",
        xaxis_title="Date",
        yaxis_title="Simulated ZHVI ($)",
        template="plotly_dark",
        showlegend=True,
    )
    fig_sim.update_xaxes(range=[sim_dates_plotting[0], sim_dates_plotting[-1]])

    final_prices = sim_paths[:, -1]
    fig_hist_final = go.Figure(
        data=[
            go.Histogram(
                x=final_prices,
                nbinsx=100,
                name='Final Value Distribution',
            )
        ]
    )

    p5 = np.percentile(final_prices, 5)
    p50 = np.percentile(final_prices, 50)
    p95 = np.percentile(final_prices, 95)
    mean_final = final_prices.mean()

    fig_hist_final.add_vline(
        x=p5,
        line_dash="dash",
        line_color="yellow",
        annotation=dict(text=f" 5th Perc: ${p5:,.0f}", textangle=-45, yshift=-10),
    )
    fig_hist_final.add_vline(
        x=p50,
        line_dash="dash",
        line_color="red",
        annotation=dict(text=f"Median: ${p50:,.0f}", textangle=-45, yshift=10),
    )
    fig_hist_final.add_vline(
        x=p95,
        line_dash="dash",
        line_color="yellow",
        annotation=dict(text=f"95th Perc: ${p95:,.0f}", textangle=-45, yshift=-20),
    )
    fig_hist_final.add_vline(
        x=mean_final,
        line_dash="dot",
        line_color="cyan",
        annotation=dict(text=f" Mean: ${mean_final:,.0f}", textangle=-45, yshift=20),
    )
    fig_hist_final.update_layout(
        title=f"{zip_code} Distribution of Final Simulated ZHVI after {sim_months} Months",
        xaxis_title="Final Simulated ZHVI ($)",
        yaxis_title="Frequency",
        template="plotly_dark",
    )

    print("Plots generated.")
    return fig_hist, fig_sim, fig_hist_final, final_prices


# --- Main Gradio Function ---
def analyze_zillow_simulation(zip_code, hist_period, sim_months, num_paths):
    """
    Orchestrates Zillow data loading, parameter calculation, CPU simulation,
    plotting, and statistics generation for the Gradio interface.
    """
    status = "Processing started..."
    try:
        status += "\nCPU simulation backend ready."

        hist_series, s0, mu_monthly, sigma_monthly = fetch_and_prepare_zillow_data(
            zip_code, hist_period
        )
        status += (
            f"\nData prepared for ZIP {zip_code}. s0=${s0:,.0f}, "
            f"mu_monthly={mu_monthly:.6f}, sigma_monthly={sigma_monthly:.6f}."
        )

        sim_months = int(sim_months)
        num_paths = int(num_paths)
        if sim_months <= 0 or num_paths <= 0:
            raise ValueError(
                "Simulation months and number of paths must be positive integers."
            )

        sim_paths = run_monte_carlo_simulation_cpu(
            s0, mu_monthly, sigma_monthly, sim_months, num_paths
        )
        status += (
            f"\nMonte Carlo simulation completed "
            f"({num_paths:,} paths, {sim_months} months)."
        )

        fig_hist, fig_sim, fig_hist_final, final_prices = create_zillow_plots(
            hist_series, sim_paths, zip_code, sim_months
        )
        status += "\nPlots generated."

        mean_final = final_prices.mean()
        median_final = np.median(final_prices)
        std_final = final_prices.std()
        p5 = np.percentile(final_prices, 5)
        p95 = np.percentile(final_prices, 95)

        summary_text = (
            f"--- Simulation Summary (ZIP: {zip_code}) ---\n"
            f"Based on Historical Period: {hist_period} "
            f"(Actual start: {hist_series.index[0].strftime('%Y-%m-%d')})\n"
            f"Last Historical Value (s0): ${s0:,.0f} "
            f"(as of {hist_series.index[-1].strftime('%Y-%m-%d')})\n"
            f"Monthly Drift (mu): {mu_monthly:.6f}\n"
            f"Monthly Volatility (sigma): {sigma_monthly:.6f}\n"
            f"Simulation Length: {sim_months} months\n"
            f"Number of Paths (Traces): {num_paths:,}\n\n"
            f"--- Final Simulated Value Statistics ---\n"
            f"Mean: ${mean_final:,.0f}\n"
            f"Median: ${median_final:,.0f}\n"
            f"Standard Deviation: ${std_final:,.0f}\n"
            f"5th Percentile: ${p5:,.0f}\n"
            f"95th Percentile: ${p95:,.0f}\n\n"
            f"Status: {status}\nProcessing finished."
        )
        return fig_hist, fig_sim, fig_hist_final, summary_text

    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(f"ERROR in analyze_zillow_simulation: {error_message}")
        empty_fig = go.Figure().update_layout(
            template="plotly_dark", title=f"Error: {e}"
        )
        return (
            empty_fig,
            empty_fig,
            empty_fig,
            f"{status}\nError:\n{error_message}",
        )


# --- Gradio Interface Definition ---
with gr.Blocks(
    theme=gr.themes.Default(primary_hue="green", secondary_hue="lime"),
    title="Zillow ZHVI MC Simulator",
) as demo:
    gr.Markdown("# CPU-Only Zillow ZHVI Simulation (Monte Carlo + NumPy)")
    gr.Markdown(
        "Select a US ZIP code and historical period to calculate parameters. Then, "
        "simulate potential future monthly Zillow Home Value Index (ZHVI) paths "
        "using NumPy on your CPU."
        "\n*Data Source: Zillow Research - [ZHVI Data](https://www.zillow.com/research/data/)*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            zip_input = gr.Textbox(label="Target ZIP Code", value=DEFAULT_ZIP_CODE)
            hist_period_input = gr.Dropdown(
                label="Historical Period for Params",
                choices=["3y", "5y", "10y", "15y", "max"],
                value=DEFAULT_HIST_PERIOD,
            )
            sim_months_input = gr.Slider(
                label="Simulation Months Ahead",
                minimum=12,
                maximum=240,
                value=DEFAULT_SIM_MONTHS,
                step=12,
            )
            num_paths_input = gr.Slider(
                label="Number of Simulation Paths (Traces)",
                minimum=10000,
                maximum=10000000,
                value=DEFAULT_NUM_PATHS,
                step=10000,
            )
            run_button = gr.Button("Run Simulation", variant="primary")

        with gr.Column(scale=3):
            summary_output = gr.Textbox(
                label="Summary & Status", lines=18, interactive=False
            )

    with gr.Tabs():
        with gr.TabItem("Historical ZHVI"):
            plot_output_hist = gr.Plot()
        with gr.TabItem("Monte Carlo Simulations"):
            plot_output_sim = gr.Plot()
        with gr.TabItem("Final Value Distribution"):
            plot_output_dist = gr.Plot()

    run_button.click(
        analyze_zillow_simulation,
        inputs=[zip_input, hist_period_input, sim_months_input, num_paths_input],
        outputs=[
            plot_output_hist,
            plot_output_sim,
            plot_output_dist,
            summary_output,
        ],
    )

    gr.Examples(
        examples=[
            ["80132", "max", 120, 250000],
            ["07074", "10y", 60, 150000],
            ["90210", "10y", 60, 100000],
            ["07974", "15y", 120, 200000],
            ["80132", "max", 180, 500000],
            ["33139", "5y", 36, 100000],
        ],
        inputs=[zip_input, hist_period_input, sim_months_input, num_paths_input],
    )


# --- Launch App ---
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
