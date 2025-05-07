import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
import os
import pyopencl as cl

# --- Constants ---
# URL for Zillow Home Value Index (ZHVI) Single-Family+Condo monthly data by Zip Code
# Check Zillow Research Data page for latest URLs if this breaks: https://www.zillow.com/research/data/
ZILLOW_DATA_URL = 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
MIN_YEAR = 2000 # Earliest year for Zillow data

# --- Default UI Values ---
DEFAULT_ZIP_CODE = "07974" # New Providence, NJ
DEFAULT_HIST_PERIOD = "10y" # Period for calculating historical mu/sigma
DEFAULT_SIM_MONTHS = 120  # Simulate 10 years ahead (12 * 10)
DEFAULT_NUM_PATHS = 100000 # Default simulation paths

# --- OpenCL Kernel (Geometric Brownian Motion) ---
monte_carlo_kernel_code = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable // Enable double precision if supported

__kernel void monte_carlo_gbm(
    __global const double* rand_normals, // Input: Pre-generated standard normal random numbers
    __global double* results,          // Output: Simulated price paths (flattened array)
    const double s0,                   // Input: Initial stock price
    const double mu,                   // Input: Drift per step (monthly)
    const double sigma,                // Input: Volatility per step (monthly)
    const unsigned int num_steps,      // Input: Number of simulation steps (months)
    const unsigned int num_paths       // Input: Total number of simulation paths (global size)
) {
    // Get the unique ID for this path (work-item)
    unsigned int path_id = get_global_id(0);

    // Boundary check
    if (path_id >= num_paths) {
        return;
    }

    double current_price = s0;
    // dt is 1 since mu and sigma are already per-step (monthly)
    double dt = 1.0;
    double drift_term = (mu - 0.5 * sigma * sigma) * dt;
    double vol_term = sigma * sqrt(dt);

    // Calculate start indices for this path in the flat arrays
    unsigned int random_offset = path_id * num_steps;
    unsigned int result_offset = path_id * (num_steps + 1); // +1 for s0

    results[result_offset] = s0; // Store initial price

    // Simulate path
    for (unsigned int step = 0; step < num_steps; ++step) {
        double Z = rand_normals[random_offset + step]; // Random shock for this step
        current_price = current_price * exp(drift_term + vol_term * Z);
        // Add a small floor to prevent non-positive prices in simulation
        results[result_offset + step + 1] = max(current_price, 0.01);
    }
}
"""

# --- Helper Functions ---

def get_opencl_context_queue():
    """
    Initializes and returns a PyOpenCL context and command queue.
    Attempts to use GPU first, falls back to CPU if no GPU is found.
    Caches the context and queue for efficiency within the session.

    Returns:
        tuple: (pyopencl.Context, pyopencl.CommandQueue)

    Raises:
        RuntimeError: If no suitable OpenCL devices are found or initialization fails.
    """
    # Use a function attribute as a simple cache
    if hasattr(get_opencl_context_queue, "cache"):
        # print("Returning cached OpenCL context/queue.") # Optional debug print
        return get_opencl_context_queue.cache

    print("Initializing OpenCL context...")
    try:
        platform = cl.get_platforms()[0] # Assuming the first platform is desired
        devices = []
        try: # Prioritize GPU
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            print(f"Found GPU: {devices[0].name}")
        except cl.RuntimeError: # Error usually means no devices of this type found
             print("No GPU found.")

        if not devices: # Fallback to CPU
            try:
                devices = platform.get_devices(device_type=cl.device_type.CPU)
                print(f"Using CPU: {devices[0].name}")
            except cl.RuntimeError:
                 print("No CPU found either.")

        if not devices:
            raise RuntimeError("No OpenCL devices found (GPU or CPU). Check drivers/ICD installation.")

        # Create context for the first device found (GPU or CPU)
        # Note: For specific device selection, set os.environ['PYOPENCL_CTX'] before first call
        context = cl.Context(devices=[devices[0]])
        queue = cl.CommandQueue(context)
        print(f"OpenCL context and queue created successfully for {devices[0].name}.")

        # Cache the result
        get_opencl_context_queue.cache = (context, queue)
        return context, queue
    except cl.Error as e:
        print(f"OpenCL Specific Error during setup: {e}")
        raise RuntimeError(f"Failed to initialize OpenCL context/queue. CL Error Code: {e.code}")
    except Exception as e:
        print(f"Generic Error during OpenCL setup: {e}")
        raise RuntimeError(f"Failed to initialize OpenCL: {e}")

def run_monte_carlo_simulation_opencl(context, queue, s0, mu, sigma, sim_steps, num_paths):
    """
    Runs the Monte Carlo simulation using the pre-compiled OpenCL kernel.

    Args:
        context (pyopencl.Context): The OpenCL context.
        queue (pyopencl.CommandQueue): The OpenCL command queue.
        s0 (float): Initial asset value.
        mu (float): Drift per time step (e.g., monthly).
        sigma (float): Volatility per time step (e.g., monthly).
        sim_steps (int): Number of steps (e.g., months) to simulate.
        num_paths (int): Number of simulation paths to run in parallel.

    Returns:
        numpy.ndarray: A 2D array of shape (num_paths, sim_steps + 1) containing
                       the simulated value paths, including the initial value s0.

    Raises:
        RuntimeError: If OpenCL buffer creation, kernel compilation/execution,
                      or result copying fails.
    """
    print(f"Preparing OpenCL simulation: {num_paths} paths, {sim_steps} steps...")
    start_time = time.time()

    np_dtype = np.float64 # Use double precision

    # --- Prepare Host Data ---
    total_random_numbers = num_paths * sim_steps
    # Generate random numbers on CPU; for very large sims, consider GPU random number generation (e.g., pyopencl.clrandom)
    rand_normals_np = np.random.randn(total_random_numbers).astype(np_dtype)
    results_np = np.empty(num_paths * (sim_steps + 1), dtype=np_dtype) # For s0 + steps

    # --- Create OpenCL Buffers ---
    mf = cl.mem_flags
    try:
        # Input buffer for random numbers, copy from host
        rand_normals_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rand_normals_np)
        # Output buffer for results, allocated on device
        results_buf = cl.Buffer(context, mf.WRITE_ONLY, results_np.nbytes)
    except cl.Error as e:
        raise RuntimeError(f"Failed to create OpenCL buffers. CL Error Code: {e.code}")

    # --- Compile Kernel (with caching) ---
    try:
        # Use function attribute as a simple program cache to avoid recompiling
        cache_key = "gbm_kernel"
        if not hasattr(run_monte_carlo_simulation_opencl, "program_cache"):
            run_monte_carlo_simulation_opencl.program_cache = {}

        if cache_key not in run_monte_carlo_simulation_opencl.program_cache:
             print("Compiling OpenCL kernel...")
             program = cl.Program(context, monte_carlo_kernel_code).build()
             run_monte_carlo_simulation_opencl.program_cache[cache_key] = program
             print("Kernel compiled and cached.")
        else:
             # print("Using cached OpenCL kernel.") # Optional debug print
             program = run_monte_carlo_simulation_opencl.program_cache[cache_key]

        kernel = program.monte_carlo_gbm
        # Explicitly set scalar arg types for clarity and potentially avoiding issues
        kernel.set_scalar_arg_dtypes([None, None, np_dtype, np_dtype, np_dtype, np.uint32, np.uint32])

    except cl.Error as e:
        build_log = "Build log not readily available."
        # Attempt to get build log for debugging
        try:
            # Assuming context.devices[0] is the device it tried to build for
            build_log = program.get_build_info(context.devices[0], cl.program_build_info.LOG)
        except Exception as log_e:
             build_log = f"Could not retrieve build log: {log_e}"
        raise RuntimeError(f"Failed to build OpenCL kernel. CL Error Code: {e.code}\nBuild Log:\n{build_log}")

    # --- Execute Kernel ---
    global_size = (num_paths,) # Total number of work-items (one per path)
    local_size = None # Let the OpenCL implementation determine the work-group size

    print(f"Executing kernel with global_size={global_size}...")
    try:
        # Pass arguments: queue, global work size, local work size, kernel args...
        kernel_event = kernel(queue, global_size, local_size,
                              rand_normals_buf, results_buf,
                              np_dtype(s0), np_dtype(mu), np_dtype(sigma),
                              np.uint32(sim_steps), np.uint32(num_paths))
        kernel_event.wait() # Ensure kernel finishes before proceeding
    except cl.Error as e:
         raise RuntimeError(f"Failed kernel execution. CL Error Code: {e.code}")

    # --- Retrieve Results ---
    try:
        # Copy data from the device results buffer back to the host numpy array
        cl.enqueue_copy(queue, results_np, results_buf).wait()
    except cl.Error as e:
         raise RuntimeError(f"Failed to copy results from device. CL Error Code: {e.code}")

    end_time = time.time()
    print(f"OpenCL simulation finished in {end_time - start_time:.3f} seconds.")

    # Reshape the flat results array into the desired (num_paths, sim_steps + 1) shape
    sim_paths = results_np.reshape((num_paths, sim_steps + 1))
    return sim_paths

# --- Zillow Data Functions ---

# Global variable to cache the loaded Zillow DataFrame
zillow_df_cache = None
cache_load_time = None

def load_zillow_data_cached(max_age_hours=24):
    """
    Loads the Zillow ZHVI data from the URL, caching it globally
    to avoid repeated downloads within a session or defined period.

    Args:
        max_age_hours (int): Maximum age of cache in hours before reloading.

    Returns:
        pandas.DataFrame: The loaded Zillow data.

    Raises:
        gr.Error: If loading data from the URL fails.
    """
    global zillow_df_cache, cache_load_time
    now = time.time()

    # Check if valid cache exists
    if zillow_df_cache is not None and cache_load_time is not None:
        age_seconds = now - cache_load_time
        if age_seconds < max_age_hours * 3600:
            print("Using cached Zillow data.")
            return zillow_df_cache.copy() # Return a copy to prevent modification issues

    print(f"Loading Zillow data from {ZILLOW_DATA_URL}...")
    try:
        df = pd.read_csv(ZILLOW_DATA_URL)
        # Standardize ZIP code format (string, 5 digits with leading zeros)
        df['RegionName'] = df['RegionName'].astype(str).str.zfill(5)
        zillow_df_cache = df # Update cache
        cache_load_time = now
        print("Zillow data loaded and cached.")
        return df.copy() # Return a copy
    except Exception as e:
        print(f"Error loading Zillow data: {e}")
        raise gr.Error(f"Failed to load data from Zillow URL. Error: {e}")

def fetch_and_prepare_zillow_data(zip_code, hist_period_str):
    """
    Filters Zillow data for a single ZIP code, selects the relevant historical
    period, calculates monthly log returns, and derives monthly drift (mu)
    and volatility (sigma).

    Args:
        zip_code (str): The target 5-digit ZIP code.
        hist_period_str (str): The historical period string (e.g., "5y", "10y", "max")
                               used for calculating mu and sigma.

    Returns:
        tuple: (pandas.Series, float, float, float) containing:
               - Historical ZHVI Series for the ZIP and period.
               - Last historical value (s0).
               - Calculated monthly drift (mu_monthly).
               - Calculated monthly volatility (sigma_monthly).

    Raises:
        ValueError: If data for the ZIP code is not found, is insufficient,
                    or parameters cannot be calculated.
        gr.Error: If the Zillow data cannot be loaded.
    """
    print(f"Fetching & preparing Zillow data for ZIP: {zip_code}, History: {hist_period_str}")
    df_zillow = load_zillow_data_cached()
    zip_code_str = str(zip_code).strip().zfill(5)

    df_zip = df_zillow[df_zillow['RegionName'] == zip_code_str]
    if df_zip.empty:
        raise ValueError(f"Data not found for ZIP code '{zip_code_str}'.")

    # Find first date column index robustly - assumes columns after metadata are dates
    first_date_col_index = -1
    for i, col_name in enumerate(df_zillow.columns):
        # Weak check: looks for YYYY-MM or YYYY-MM-DD format
        if isinstance(col_name, str) and (col_name.count('-') == 1 or col_name.count('-') == 2):
            try:
                pd.to_datetime(col_name, errors='raise')
                first_date_col_index = i
                break
            except (ValueError, TypeError): continue # Not a date format we expect
    if first_date_col_index == -1: raise ValueError("Could not identify date columns in Zillow CSV.")

    date_cols = df_zillow.columns[first_date_col_index:]
    # Extract the time series data for the specific ZIP code
    series = df_zip[date_cols].iloc[0].copy() # Use .copy()
    series.index = pd.to_datetime(series.index)
    series = series.dropna() # Drop any individual missing months

    if series.empty:
        raise ValueError(f"No valid price data points found for ZIP code '{zip_code_str}'.")

    # Filter series based on the historical period relative to the *last available date*
    last_date = series.index[-1]
    hist_start_date = series.index[0] # Default to earliest date

    if hist_period_str != 'max':
        try:
            # Convert period string (e.g. '5y') to date offset
            offset = pd.tseries.frequencies.to_offset(hist_period_str.replace('y', 'Y').replace('m', 'M'))
            calculated_start = last_date - offset
            # Ensure start date is not before the data begins or MIN_YEAR
            min_data_date = series.index[0]
            min_allowed_date = pd.Timestamp(year=MIN_YEAR, month=1, day=1)
            hist_start_date = max(calculated_start, min_data_date, min_allowed_date)

            series = series[series.index >= hist_start_date]
        except Exception as e:
            print(f"Warning: Could not parse period '{hist_period_str}', using max history. Error: {e}")
            # If period parsing fails, use the max available data from the series
            hist_start_date = series.index[0] # Already set above

    print(f"Using historical data from {series.index[0].strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')} for calculations ({len(series)} points).")

    if len(series) < 12: # Need at least 12 months for somewhat reliable stats
        raise ValueError(f"Insufficient historical data ({len(series)} months) for ZIP {zip_code_str} in period '{hist_period_str}'. Need at least 12.")

    # Calculate *monthly* log returns
    log_returns = np.log(series / series.shift(1)).dropna()

    if log_returns.empty:
        raise ValueError("Could not calculate log returns (maybe only 1 data point?).")

    # Calculate *monthly* drift and volatility
    mu_monthly = log_returns.mean()
    sigma_monthly = log_returns.std()
    s0 = series.iloc[-1] # Starting price for simulation is the last historical price

    # Basic sanity check for calculated parameters
    if sigma_monthly <= 0 or pd.isna(sigma_monthly) or pd.isna(mu_monthly) or pd.isna(s0):
         raise ValueError(f"Calculated parameters invalid: mu={mu_monthly}, sigma={sigma_monthly}, s0={s0}. Check historical data.")

    print(f"Params calculated: s0=${s0:,.0f}, mu_monthly={mu_monthly:.6f}, sigma_monthly={sigma_monthly:.6f}")
    return series, s0, mu_monthly, sigma_monthly


def create_zillow_plots(hist_series, sim_paths, zip_code, sim_months):
    """
    Creates Plotly figures for historical ZHVI, simulated paths, and final distribution.

    Args:
        hist_series (pandas.Series): Historical ZHVI data with DateTimeIndex.
        sim_paths (numpy.ndarray): 2D array of simulated paths.
        zip_code (str): The ZIP code for titles.
        sim_months (int): The number of simulated months for titles/axis.

    Returns:
        tuple: (go.Figure, go.Figure, go.Figure, numpy.ndarray) containing:
               - Historical plot figure.
               - Simulation paths figure.
               - Final value distribution histogram figure.
               - 1D array of final simulated prices.
    """
    print("Generating plots...")

    # --- 1. Historical Data Plot ---
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=hist_series.index, y=hist_series, mode='lines', name=f'{zip_code} Historical ZHVI'))
    fig_hist.update_layout(
        title=f"Zillow Home Value Index (ZHVI) - ZIP: {zip_code}",
        xaxis_title="Date", yaxis_title="ZHVI ($)", template="plotly_dark"
    )

    # --- 2. Simulation Paths Plot ---
    fig_sim = go.Figure()
    num_paths_to_plot = min(sim_paths.shape[0], 1000) # Limit plotted paths for browser performance
    last_hist_date = hist_series.index[-1]
    # Generate future monthly dates starting from the month *after* the last historical date
    # Add 1 month to last date, then generate range. Using MonthEnd freq 'ME'.
    start_sim_date = last_hist_date + pd.DateOffset(months=1)
    sim_dates_full_path = pd.date_range(start=start_sim_date, periods=sim_months, freq='ME')
    # Prepend the last historical date to align with sim_paths which includes s0 at index 0
    sim_dates_plotting = pd.Index([last_hist_date]).union(sim_dates_full_path)


    # Plot a subset of simulation paths for clarity/performance
    for i in range(num_paths_to_plot):
        fig_sim.add_trace(go.Scatter(x=sim_dates_plotting, y=sim_paths[i, :], mode='lines',
                                     line=dict(width=0.5), showlegend=False, opacity=0.1))
    # Plot the mean path
    mean_path = sim_paths.mean(axis=0)
    fig_sim.add_trace(go.Scatter(x=sim_dates_plotting, y=mean_path, mode='lines', name='Mean Path',
                                 line=dict(color='red', width=2)))

    fig_sim.update_layout(
        title=f"{zip_code} ZHVI Monte Carlo Simulations ({sim_paths.shape[0]:,} Paths)",
        xaxis_title="Date", yaxis_title="Simulated ZHVI ($)", template="plotly_dark", showlegend=True
    )
    # Set x-axis range to show only the simulation period clearly
    fig_sim.update_xaxes(range=[sim_dates_plotting[0], sim_dates_plotting[-1]])


    # --- 3. Final Price Histogram ---
    final_prices = sim_paths[:, -1] # Last value of each path
    fig_hist_final = go.Figure(data=[go.Histogram(x=final_prices, nbinsx=100, name='Final Value Distribution')])

    # Calculate statistics for annotations
    p5 = np.percentile(final_prices, 5)
    p50 = np.percentile(final_prices, 50) # Median
    p95 = np.percentile(final_prices, 95)
    mean_final = final_prices.mean()

    # Add vertical lines with rotated annotations AND vertical shift
    fig_hist_final.add_vline(x=p5, line_dash="dash", line_color="yellow",
                             annotation=dict(
                                 text=f" 5th Perc: ${p5:,.0f}",
                                 textangle=-45,
                                 yshift=-10 # Shift down
                             ))
    fig_hist_final.add_vline(x=p50, line_dash="dash", line_color="red",
                             annotation=dict(
                                 text=f"Median: ${p50:,.0f}",
                                 textangle=-45,
                                 yshift=10 # Shift up
                             ))
    fig_hist_final.add_vline(x=p95, line_dash="dash", line_color="yellow",
                             annotation=dict(
                                 text=f"95th Perc: ${p95:,.0f}",
                                 textangle=-45,
                                 yshift=-20 # Shift further down
                             ))
    fig_hist_final.add_vline(x=mean_final, line_dash="dot", line_color="cyan",
                             annotation=dict(
                                 text=f" Mean: ${mean_final:,.0f}",
                                 textangle=-45,
                                 yshift=20 # Shift further up
                             ))

    fig_hist_final.update_layout(
        title=f"{zip_code} Distribution of Final Simulated ZHVI after {sim_months} Months",
        xaxis_title="Final Simulated ZHVI ($)", yaxis_title="Frequency", template="plotly_dark"
    )

    print("Plots generated.")
    return fig_hist, fig_sim, fig_hist_final, final_prices


# --- Main Gradio Function ---
def analyze_zillow_simulation(zip_code, hist_period, sim_months, num_paths):
    """
    Orchestrates the Zillow data fetching, parameter calculation, OpenCL simulation,
    plotting, and statistics generation for the Gradio interface.

    Args:
        zip_code (str): Target ZIP code.
        hist_period (str): Historical period for calculations (e.g., "10y").
        sim_months (int): Number of months to simulate.
        num_paths (int): Number of simulation paths.

    Returns:
        tuple: Contains Plotly figures (hist, sim, dist) and a summary text string.
               Returns empty figures and error message on failure.
    """
    status = "Processing started..."
    try:
        # Initialize OpenCL context and queue (uses cache if already done)
        context, queue = get_opencl_context_queue()
        status += "\nOpenCL context ready."

        # Fetch data and calculate monthly parameters
        hist_series, s0, mu_monthly, sigma_monthly = fetch_and_prepare_zillow_data(zip_code, hist_period)
        status += f"\nData prepared for ZIP {zip_code}. s0=${s0:,.0f}, mu_monthly={mu_monthly:.6f}, sigma_monthly={sigma_monthly:.6f}."

        # Ensure simulation parameters are valid integers
        sim_months = int(sim_months)
        num_paths = int(num_paths)
        if sim_months <= 0 or num_paths <= 0:
            raise ValueError("Simulation months and number of paths must be positive integers.")

        # Run the simulation on the GPU/CPU via OpenCL
        sim_paths = run_monte_carlo_simulation_opencl(context, queue, s0, mu_monthly, sigma_monthly, sim_months, num_paths)
        status += f"\nMonte Carlo simulation completed ({num_paths:,} paths, {sim_months} months)."

        # Create plots and get final prices
        fig_hist, fig_sim, fig_hist_final, final_prices = create_zillow_plots(hist_series, sim_paths, zip_code, sim_months)
        status += "\nPlots generated."

        # Calculate summary statistics from simulation results
        mean_final = final_prices.mean()
        median_final = np.median(final_prices)
        std_final = final_prices.std()
        p5 = np.percentile(final_prices, 5)
        p95 = np.percentile(final_prices, 95)

        # Format summary text for display
        summary_text = (
            f"--- Simulation Summary (ZIP: {zip_code}) ---\n"
            f"Based on Historical Period: {hist_period} (Actual start: {hist_series.index[0].strftime('%Y-%m-%d')})\n"
            f"Last Historical Value (s0): ${s0:,.0f} (as of {hist_series.index[-1].strftime('%Y-%m-%d')})\n"
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
        # Return plots and summary text to Gradio outputs
        return fig_hist, fig_sim, fig_hist_final, summary_text

    except (ValueError, RuntimeError, cl.Error, Exception) as e:
        # Handle errors gracefully and report them in the UI
        error_message = f"An error occurred: {e}"
        print(f"ERROR in analyze_zillow_simulation: {error_message}") # Log to console
        # Create empty plots to send back on error
        empty_fig = go.Figure().update_layout(template="plotly_dark", title=f"Error: {e}")
        return empty_fig, empty_fig, empty_fig, f"{status}\nError:\n{error_message}"


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="green", secondary_hue="lime"), title="Zillow ZHVI MC Simulator") as demo:
    gr.Markdown("# GPU-Accelerated Zillow ZHVI Simulation (Monte Carlo + PyOpenCL)")
    gr.Markdown(
        "Select a US ZIP code and historical period to calculate parameters. Then, simulate potential future "
        "monthly Zillow Home Value Index (ZHVI) paths using OpenCL on your GPU/CPU."
        "\n*Data Source: Zillow Research - [ZHVI Data](https://www.zillow.com/research/data/)*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            zip_input = gr.Textbox(label="Target ZIP Code", value=DEFAULT_ZIP_CODE)
            hist_period_input = gr.Dropdown(label="Historical Period for Params", choices=["3y", "5y", "10y", "15y", "max"], value=DEFAULT_HIST_PERIOD)
            sim_months_input = gr.Slider(label="Simulation Months Ahead", minimum=12, maximum=240, value=DEFAULT_SIM_MONTHS, step=12) # Range: 1 to 20 years
            # Adjusted slider for number of paths (traces) - no log scale label
            num_paths_input = gr.Slider(label="Number of Simulation Paths (Traces)", minimum=10000, maximum=10000000, value=DEFAULT_NUM_PATHS, step=10000)
            run_button = gr.Button("Run Simulation", variant="primary")

        with gr.Column(scale=3):
            # Textbox to display summary statistics and status/error messages
            summary_output = gr.Textbox(label="Summary & Status", lines=18, interactive=False)

    with gr.Tabs():
        with gr.TabItem("Historical ZHVI"):
             plot_output_hist = gr.Plot()
        with gr.TabItem("Monte Carlo Simulations"):
             plot_output_sim = gr.Plot()
        with gr.TabItem("Final Value Distribution"):
             plot_output_dist = gr.Plot()

    # Connect the button click to the main analysis function and specify inputs/outputs
    run_button.click(
        analyze_zillow_simulation,
        inputs=[zip_input, hist_period_input, sim_months_input, num_paths_input],
        outputs=[plot_output_hist, plot_output_sim, plot_output_dist, summary_output]
    )

    # Provide examples for users to easily try (Corrected List)
    gr.Examples(
        examples=[
            # Corrected examples for 80132 and 07074
            ["80132", "max", 120, 250000],  # Monument, CO; max history; 10yr sim; 250k paths
            ["07074", "10y", 60, 150000],   # Paramus, NJ; 10y history; 5yr sim; 150k paths
            # Original valid examples below
            ["90210", "10y", 60, 100000],   # Beverly Hills; 10y history; 5yr sim; 100k paths
            ["07974", "15y", 120, 200000],  # New Providence, NJ; 15y history; 10yr sim; 200k paths
            ["80132", "max", 180, 500000],  # Monument, CO; max history; 15yr sim; 500k paths (different params)
            ["33139", "5y", 36, 100000],   # Miami Beach; 5y history; 3yr sim; 100k paths
        ],
        # Ensure inputs match the function signature for examples
        inputs=[zip_input, hist_period_input, sim_months_input, num_paths_input]
    )


# --- Launch App ---
if __name__ == "__main__":
    # Launch Gradio app. share=False keeps it local. debug=True shows errors in browser.
    demo.launch(share=True, debug=True)
