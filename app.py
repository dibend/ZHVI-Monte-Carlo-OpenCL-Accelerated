import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from functools import lru_cache

try:
    import pyopencl as cl
except (ImportError, OSError):
    cl = None

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
CPU_RANDOM_TARGET_BYTES = 256 * 1024 * 1024
OPENCL_BATCH_TARGET_BYTES = 256 * 1024 * 1024

# --- OpenCL Kernel (Geometric Brownian Motion) ---
monte_carlo_kernel_code = """
#ifdef USE_DOUBLE
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
typedef double real_t;
#else
typedef float real_t;
#endif

__kernel void monte_carlo_gbm(
    __global const real_t* rand_normals, // Input: Pre-generated standard normal random numbers
    __global real_t* results,          // Output: Simulated price paths (flattened array)
    const real_t s0,                   // Input: Initial stock price
    const real_t mu,                   // Input: Drift per step (monthly)
    const real_t sigma,                // Input: Volatility per step (monthly)
    const unsigned int num_steps,      // Input: Number of simulation steps (months)
    const unsigned int num_paths       // Input: Total number of simulation paths (global size)
) {
    // Get the unique ID for this path (work-item)
    size_t path_id = get_global_id(0);

    // Boundary check
    if (path_id >= num_paths) {
        return;
    }

    real_t current_price = s0;
    // dt is 1 since mu and sigma are already per-step (monthly)
    real_t drift_term = mu - (real_t)0.5 * sigma * sigma;
    real_t vol_term = sigma;

    // Calculate start indices for this path in the flat arrays
    size_t random_offset = path_id * num_steps;
    size_t result_offset = path_id * (num_steps + 1); // +1 for s0

    results[result_offset] = s0; // Store initial price

    // Simulate path
    for (unsigned int step = 0; step < num_steps; ++step) {
        real_t Z = rand_normals[random_offset + step]; // Random shock for this step
        current_price = current_price * exp(drift_term + vol_term * Z);
        // Add a small floor to prevent non-positive prices in simulation
        // Use fmax for floating point types
        results[result_offset + step + 1] = fmax(current_price, (real_t)0.01);
    }
}
"""

# --- Helper Functions ---

def device_supports_fp64(device):
    """Some drivers expose FP64 through a capability, others through an extension."""
    try:
        if device.double_fp_config:
            return True
    except cl.Error:
        pass
    return bool({"cl_khr_fp64", "cl_amd_fp64"} & set(device.extensions.split()))


def discover_opencl_devices():
    """Enumerate all platforms without allowing a broken driver to hide others."""
    if cl is None:
        return [], ["PyOpenCL is not installed or could not be loaded."]
    try:
        platforms = cl.get_platforms()
    except Exception as exc:
        return [], [f"OpenCL discovery failed: {exc}"]

    devices, issues = [], []
    for platform_index, platform in enumerate(platforms):
        try:
            platform_name = platform.name.strip()
            platform_key = str(platform.int_ptr)
            platform_devices = platform.get_devices()
        except Exception as exc:
            issues.append(f"Platform {platform_index + 1} could not be read: {exc}")
            continue
        for device_index, device in enumerate(platform_devices):
            try:
                name = device.name.strip()
                if not device.available or not device.compiler_available:
                    issues.append(f"{platform_name} / {name} is unavailable or has no OpenCL compiler.")
                    continue
                kind = (
                    "GPU" if device.type & cl.device_type.GPU else
                    "CPU" if device.type & cl.device_type.CPU else "Accelerator"
                )
                precision = "float64" if device_supports_fp64(device) else "float32"
                devices.append({
                    # Driver handles distinguish identically named physical devices.
                    "key": f"{platform_key}:{device.int_ptr}",
                    "platform_key": platform_key,
                    "platform_label": f"{platform_index + 1}: {platform_name}",
                    "name": name,
                    "label": f"{platform_name} / {device_index + 1}: {name} ({kind}, {precision})",
                    "kind": kind,
                    "precision": precision,
                    "device": device,
                })
            except Exception as exc:
                issues.append(f"{platform_name} device {device_index + 1} could not be read: {exc}")
    if not devices and not issues:
        issues.append("No OpenCL devices were found.")
    return devices, issues


@lru_cache(maxsize=8)
def _get_opencl_context(device):
    return cl.Context(devices=[device])


def get_opencl_context_queue(device):
    """Cache contexts per physical device, but give each simulation its own queue."""
    context = _get_opencl_context(device)
    return context, cl.CommandQueue(context, device=device)


@lru_cache(maxsize=8)
def _get_opencl_program(context, precision):
    # Programs belong to a context, not a device name. Two GPUs may share a name.
    options = ["-DUSE_DOUBLE=1"] if precision == "float64" else []
    return cl.Program(context, monte_carlo_kernel_code).build(options=options)


def clear_opencl_caches():
    _get_opencl_program.cache_clear()
    _get_opencl_context.cache_clear()


def opencl_batch_paths(device, sim_steps, num_paths, itemsize):
    """Respect both per-buffer and total device-memory limits."""
    random_bytes = sim_steps * itemsize
    result_bytes = (sim_steps + 1) * itemsize
    budget = min(OPENCL_BATCH_TARGET_BYTES, int(device.global_mem_size) // 4)
    max_allocation = int(device.max_mem_alloc_size)
    rows = min(
        num_paths,
        budget // (random_bytes + result_bytes),
        max_allocation // random_bytes,
        max_allocation // result_bytes,
    )
    if rows < 1:
        raise RuntimeError("The selected OpenCL device has too little memory for one path.")
    return rows


def run_monte_carlo_simulation_opencl(context, queue, s0, mu, sigma, sim_steps,
                                      num_paths, precision=None):
    """Run the embedded GBM kernel in bounded batches on the selected device."""
    if precision is None:
        precision = "float64" if device_supports_fp64(queue.device) else "float32"
    if precision not in ("float32", "float64"):
        raise ValueError("OpenCL precision must be float32 or float64.")
    np_dtype = np.dtype(precision).type
    print(f"Preparing OpenCL simulation: {num_paths} paths, {sim_steps} steps ({precision})...")
    start_time = time.time()

    program = _get_opencl_program(context, precision)
    # Kernel argument state must not be shared between concurrent requests.
    kernel = cl.Kernel(program, "monte_carlo_gbm")
    kernel.set_scalar_arg_dtypes([None, None, np_dtype, np_dtype, np_dtype, np.uint32, np.uint32])
    batch_paths = opencl_batch_paths(queue.device, sim_steps, num_paths, np.dtype(np_dtype).itemsize)
    results = np.empty((num_paths, sim_steps + 1), dtype=np_dtype)

    random_buffer = result_buffer = None
    try:
        random_buffer = cl.Buffer(
            context, cl.mem_flags.READ_ONLY,
            batch_paths * sim_steps * np.dtype(np_dtype).itemsize,
        )
        result_buffer = cl.Buffer(
            context, cl.mem_flags.WRITE_ONLY,
            batch_paths * (sim_steps + 1) * np.dtype(np_dtype).itemsize,
        )
        for start in range(0, num_paths, batch_paths):
            stop = min(start + batch_paths, num_paths)
            rows = stop - start
            # Path-major ordering matches the NumPy implementation.
            normals = np.random.randn(rows, sim_steps).astype(np_dtype, copy=False)
            cl.enqueue_copy(queue, random_buffer, normals, is_blocking=True)
            kernel(
                queue, (rows,), None, random_buffer, result_buffer,
                np_dtype(s0), np_dtype(mu), np_dtype(sigma),
                np.uint32(sim_steps), np.uint32(rows),
            )
            cl.enqueue_copy(queue, results[start:stop], result_buffer, is_blocking=True)
            if not np.isfinite(results[start:stop]).all():
                raise RuntimeError("OpenCL produced non-finite values; retrying on CPU.")
    finally:
        # Release device buffers even on allocation/execution/copy errors.
        for buffer in (random_buffer, result_buffer):
            if buffer is not None:
                buffer.release()

    print(f"OpenCL simulation finished in {time.time() - start_time:.3f} seconds.")
    return results


def run_simulation(s0, mu, sigma, sim_steps, num_paths, backend="auto",
                   platform_key="auto", device_key="auto"):
    """Honor explicit selections, try all automatic candidates, then fall back."""
    if sim_steps <= 0 or num_paths <= 0:
        raise ValueError("Simulation months and number of paths must be positive integers.")
    if backend == "cpu":
        return (
            run_monte_carlo_simulation_cpu(s0, mu, sigma, sim_steps, num_paths),
            "Backend: NumPy CPU (selected manually).",
        )
    if backend != "auto":
        raise ValueError("Unknown compute backend.")

    devices, issues = discover_opencl_devices()
    candidates = [
        item for item in devices
        if platform_key in (None, "auto", item["platform_key"])
        and device_key in (None, "auto", item["key"])
    ]
    if not candidates and (platform_key not in (None, "auto") or device_key not in (None, "auto")):
        issues.insert(0, "The selected OpenCL platform/device is no longer available. Refresh devices.")
    candidates.sort(key=lambda item: {"GPU": 0, "CPU": 1}.get(item["kind"], 2))
    for item in candidates:
        try:
            context, queue = get_opencl_context_queue(item["device"])
            paths = run_monte_carlo_simulation_opencl(
                context, queue, s0, mu, sigma, sim_steps, num_paths, item["precision"],
            )
            return paths, f"Backend: PyOpenCL — {item['label']}."
        except Exception as exc:
            print(f"OpenCL failed on {item['label']}: {exc}")
            issues.append(f"{item['name']}: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}")
            clear_opencl_caches()

    reason = "; ".join(issues) or "No usable OpenCL device."
    # Run the CPU fallback outside the OpenCL exception handler so failed buffers
    # and host arrays are not kept alive by its traceback.
    return (
        run_monte_carlo_simulation_cpu(s0, mu, sigma, sim_steps, num_paths),
        f"Backend: NumPy CPU (OpenCL fallback: {reason}).",
    )


def update_compute_controls(backend="auto", platform_key="auto", device_key="auto"):
    """Refresh dropdown choices, retaining selections only while they are valid."""
    devices, issues = discover_opencl_devices()
    platforms = list(dict.fromkeys(
        (item["platform_label"], item["platform_key"]) for item in devices
    ))
    platform_choices = [("All OpenCL platforms", "auto")] + platforms
    messages = []
    if platform_key not in {value for _, value in platform_choices}:
        platform_key, device_key = "auto", "auto"
        messages.append("Previous platform unavailable; selection reset to automatic.")
    filtered = [item for item in devices if platform_key in ("auto", item["platform_key"])]
    device_choices = [("Automatic (GPU first)", "auto")] + [
        (item["label"], item["key"]) for item in filtered
    ]
    if device_key not in {value for _, value in device_choices}:
        device_key = "auto"
        messages.append("Device selection reset to automatic for this platform.")
    enabled = backend != "cpu" and bool(devices)
    if backend == "cpu":
        messages.append("NumPy CPU selected. OpenCL will not be used for simulations.")
    elif devices:
        messages.append(f"Found {len(devices)} OpenCL device(s). NumPy CPU fallback is always available.")
    else:
        messages.append("No usable OpenCL device. Simulations will run on NumPy CPU.")
    messages.extend(issues)
    return (
        gr.Dropdown(choices=platform_choices, value=platform_key, interactive=enabled),
        gr.Dropdown(choices=device_choices, value=device_key, interactive=enabled and bool(filtered)),
        "\n".join(messages),
    )

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
    # Accumulate in float64 even when a device returns float32 paths.
    mean_path = sim_paths.mean(axis=0, dtype=np.float64)
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
    mean_final = final_prices.mean(dtype=np.float64)

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
def analyze_zillow_simulation(zip_code, hist_period, sim_months, num_paths,
                              backend="auto", platform_key="auto", device_key="auto"):
    """
    Orchestrates the Zillow data fetching, parameter calculation, OpenCL simulation,
    plotting, and statistics generation for the Gradio interface.

    Args:
        zip_code (str): Target ZIP code.
        hist_period (str): Historical period for calculations (e.g., "10y").
        sim_months (int): Number of months to simulate.
        num_paths (int): Number of simulation paths.
        backend (str): Use OpenCL when available (auto), or force NumPy (cpu).
        platform_key (str): Selected OpenCL platform, or auto for all platforms.
        device_key (str): Selected device, or auto to prefer GPUs.

    Returns:
        tuple: Contains Plotly figures (hist, sim, dist) and a summary text string.
               Returns empty figures and error message on failure.
    """
    status = "Processing started..."
    try:
        # Fetch data and calculate monthly parameters
        hist_series, s0, mu_monthly, sigma_monthly = fetch_and_prepare_zillow_data(zip_code, hist_period)
        status += f"\nData prepared for ZIP {zip_code}. s0=${s0:,.0f}, mu_monthly={mu_monthly:.6f}, sigma_monthly={sigma_monthly:.6f}."

        # Ensure simulation parameters are valid integers
        sim_months = int(sim_months)
        num_paths = int(num_paths)
        if sim_months <= 0 or num_paths <= 0:
            raise ValueError("Simulation months and number of paths must be positive integers.")

        sim_paths, backend_status = run_simulation(
            s0, mu_monthly, sigma_monthly, sim_months, num_paths,
            backend, platform_key, device_key,
        )
        status += f"\n{backend_status}"
        status += f"\nMonte Carlo simulation completed ({num_paths:,} paths, {sim_months} months)."

        # Create plots and get final prices
        fig_hist, fig_sim, fig_hist_final, final_prices = create_zillow_plots(hist_series, sim_paths, zip_code, sim_months)
        status += "\nPlots generated."

        # Calculate summary statistics from simulation results
        mean_final = final_prices.mean(dtype=np.float64)
        median_final = np.median(final_prices)
        std_final = final_prices.std(dtype=np.float64)
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

    except Exception as e:
        # Handle errors gracefully and report them in the UI
        error_message = f"An error occurred: {e}"
        print(f"ERROR in analyze_zillow_simulation: {error_message}") # Log to console
        # Create empty plots to send back on error
        empty_fig = go.Figure().update_layout(template="plotly_dark", title=f"Error: {e}")
        return empty_fig, empty_fig, empty_fig, f"{status}\nError:\n{error_message}"


# --- Gradio Interface Definition ---
with gr.Blocks(title="Zillow ZHVI MC Simulator") as demo:
    gr.Markdown("# Zillow ZHVI Simulation (Monte Carlo + optional PyOpenCL)")
    gr.Markdown(
        "Select a US ZIP code and historical period to calculate parameters. Then, simulate potential future "
        "monthly Zillow Home Value Index (ZHVI) paths using OpenCL when supported, or NumPy on the CPU."
        "\n*Data Source: Zillow Research - [ZHVI Data](https://www.zillow.com/research/data/)*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            zip_input = gr.Textbox(label="Target ZIP Code", value=DEFAULT_ZIP_CODE)
            hist_period_input = gr.Dropdown(label="Historical Period for Params", choices=["3y", "5y", "10y", "15y", "max"], value=DEFAULT_HIST_PERIOD)
            sim_months_input = gr.Slider(label="Simulation Months Ahead", minimum=12, maximum=240, value=DEFAULT_SIM_MONTHS, step=12) # Range: 1 to 20 years
            # Adjusted slider for number of paths (traces) - no log scale label
            num_paths_input = gr.Slider(label="Number of Simulation Paths (Traces)", minimum=10000, maximum=10000000, value=DEFAULT_NUM_PATHS, step=10000)
            backend_input = gr.Dropdown(
                label="Compute backend",
                choices=[("OpenCL if available (CPU fallback)", "auto"), ("CPU (NumPy)", "cpu")],
                value="auto", interactive=True,
            )
            platform_input = gr.Dropdown(
                label="OpenCL platform", choices=[("All OpenCL platforms", "auto")],
                value="auto", interactive=False,
            )
            device_input = gr.Dropdown(
                label="OpenCL device", choices=[("Automatic (GPU first)", "auto")],
                value="auto", interactive=False,
            )
            refresh_devices_button = gr.Button("Refresh devices")
            device_status = gr.Textbox(
                label="Device availability", value="Checking OpenCL devices...",
                lines=3, interactive=False,
            )
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
        inputs=[zip_input, hist_period_input, sim_months_input, num_paths_input,
                backend_input, platform_input, device_input],
        outputs=[plot_output_hist, plot_output_sim, plot_output_dist, summary_output]
    )

    compute_inputs = [backend_input, platform_input, device_input]
    compute_outputs = [platform_input, device_input, device_status]
    demo.load(update_compute_controls, inputs=compute_inputs, outputs=compute_outputs)
    # input fires only for user edits, avoiding recursive events from dropdown updates.
    backend_input.input(update_compute_controls, inputs=compute_inputs, outputs=compute_outputs)
    platform_input.input(update_compute_controls, inputs=compute_inputs, outputs=compute_outputs)
    refresh_devices_button.click(update_compute_controls, inputs=compute_inputs, outputs=compute_outputs)

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
    demo.launch(
        share=True, debug=True,
        theme=gr.themes.Default(primary_hue="green", secondary_hue="lime"),
    )
