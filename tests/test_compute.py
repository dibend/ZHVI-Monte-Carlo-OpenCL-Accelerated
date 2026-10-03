"""Behavioral checks; real OpenCL checks skip when no runtime is installed."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import numpy as np
import pandas as pd

import app


def device(pointer, name="Test GPU", kind=4, fp64=True, **overrides):
    fields = dict(
        int_ptr=pointer, name=name, type=kind, double_fp_config=int(fp64),
        extensions="cl_khr_fp64" if fp64 else "", available=True,
        compiler_available=True, global_mem_size=1024 ** 3,
        max_mem_alloc_size=256 * 1024 ** 2,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def platform(pointer, devices, name="Test platform"):
    return SimpleNamespace(int_ptr=pointer, name=name, get_devices=Mock(return_value=devices))


def fake_cl(platforms):
    return SimpleNamespace(
        get_platforms=Mock(return_value=platforms), Error=RuntimeError,
        device_type=SimpleNamespace(GPU=4, CPU=2),
    )


class DeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.gpu1 = device(11, fp64=False)
        self.gpu2 = device(12)  # Same name, separate device and context.
        self.cpu = device(21, name="OpenCL CPU", kind=2)
        self.cl = fake_cl([
            platform(1, [self.gpu1, self.gpu2]),
            platform(2, [self.cpu], name="CPU platform"),
        ])
        self.params = (100.0, 0.002, 0.015, 12, 8)

    def test_discovery_keeps_all_platforms_duplicate_names_and_precisions(self):
        broken = SimpleNamespace(name="Broken", int_ptr=3, get_devices=Mock(side_effect=RuntimeError("bad ICD")))
        self.cl.get_platforms.return_value.insert(0, broken)
        with patch.object(app, "cl", self.cl):
            devices, issues = app.discover_opencl_devices()
        self.assertEqual([d["key"] for d in devices], ["1:11", "1:12", "2:21"])
        self.assertEqual([d["precision"] for d in devices], ["float32", "float64", "float64"])
        self.assertEqual([d["kind"] for d in devices], ["GPU", "GPU", "CPU"])
        self.assertIn("bad ICD", issues[0])
        self.assertNotEqual(devices[0]["label"], devices[1]["label"])

    def test_unavailable_device_does_not_hide_working_devices(self):
        self.gpu1.compiler_available = False
        with patch.object(app, "cl", self.cl):
            devices, issues = app.discover_opencl_devices()
        self.assertEqual(len(devices), 2)
        self.assertIn("compiler", issues[0])

    def test_missing_pyopencl_runs_cpu(self):
        with patch.object(app, "cl", None):
            paths, status = app.run_simulation(*self.params)
        self.assertEqual(paths.shape, (8, 13))
        np.testing.assert_array_equal(paths[:, 0], 100)
        self.assertTrue(np.isfinite(paths).all())
        self.assertIn("NumPy CPU", status)
        self.assertIn("not installed", status)

    def test_missing_driver_runs_cpu(self):
        self.cl.get_platforms.side_effect = RuntimeError("PLATFORM_NOT_FOUND_KHR")
        with patch.object(app, "cl", self.cl):
            paths, status = app.run_simulation(*self.params)
        self.assertEqual(paths.shape, (8, 13))
        self.assertIn("PLATFORM_NOT_FOUND_KHR", status)

    def test_manual_cpu_does_not_probe_opencl(self):
        with patch.object(app, "discover_opencl_devices", side_effect=AssertionError("OpenCL was probed")):
            _, status = app.run_simulation(*self.params, backend="cpu", device_key="1:11")
        self.assertIn("selected manually", status)

    def test_explicit_gpu_and_opencl_cpu_selection(self):
        for platform_key, key, expected, precision in [
            ("1", "1:12", self.gpu2, "float64"),
            ("auto", "1:11", self.gpu1, "float32"),
            ("2", "2:21", self.cpu, "float64"),
        ]:
            with self.subTest(key=key), patch.object(app, "cl", self.cl), \
                    patch.object(app, "get_opencl_context_queue", return_value=("ctx", "queue")) as setup, \
                    patch.object(app, "run_monte_carlo_simulation_opencl", return_value=np.ones((8, 13))) as run:
                _, status = app.run_simulation(*self.params, platform_key=platform_key, device_key=key)
            self.assertIs(setup.call_args.args[0], expected)
            self.assertEqual(run.call_args.args[-1], precision)
            self.assertIn("PyOpenCL", status)

    def test_automatic_tries_next_device_after_failure(self):
        # Put the CPU platform first to make sure automatic still prefers GPUs.
        self.cl.get_platforms.return_value.reverse()
        with patch.object(app, "cl", self.cl), \
                patch.object(app, "get_opencl_context_queue", return_value=("ctx", "queue")) as setup, \
                patch.object(app, "run_monte_carlo_simulation_opencl", side_effect=[RuntimeError("build failed"), np.ones((8, 13))]):
            _, status = app.run_simulation(*self.params)
        self.assertEqual([c.args[0] for c in setup.call_args_list], [self.gpu1, self.gpu2])
        self.assertIn("float64", status)

    def test_selected_device_failure_falls_back_without_trying_another(self):
        for error in ("context failed", "build failed", "out of memory", "copy failed"):
            with self.subTest(error=error), patch.object(app, "cl", self.cl), \
                    patch.object(app, "get_opencl_context_queue", side_effect=RuntimeError(error)) as setup:
                paths, status = app.run_simulation(*self.params, device_key="1:11")
            setup.assert_called_once_with(self.gpu1)
            self.assertEqual(paths.shape, (8, 13))
            self.assertIn(error, status)
            self.assertIn("NumPy CPU", status)

    def test_stale_or_mismatched_selection_falls_back(self):
        for platform_key, key in [("1", "2:21"), ("gone", "auto"), ("auto", "gone")]:
            with self.subTest(key=key), patch.object(app, "cl", self.cl), \
                    patch.object(app, "get_opencl_context_queue") as setup:
                _, status = app.run_simulation(*self.params, platform_key=platform_key, device_key=key)
            setup.assert_not_called()
            self.assertIn("no longer available", status)

    def test_dropdown_filtering_preservation_and_reset(self):
        with patch.object(app, "cl", self.cl):
            p, d, _ = app.update_compute_controls("auto", "1", "1:12")
            self.assertEqual(p.value, "1")
            self.assertEqual(d.value, "1:12")
            self.assertEqual([value for _, value in d.choices], ["auto", "1:11", "1:12"])
            p, d, status = app.update_compute_controls("auto", "2", "1:12")
            self.assertEqual(d.value, "auto")
            self.assertEqual([value for _, value in d.choices], ["auto", "2:21"])
            self.assertIn("reset", status)
            p, d, status = app.update_compute_controls("auto", "gone", "gone")
            self.assertEqual((p.value, d.value), ("auto", "auto"))
            self.assertIn("Previous platform unavailable", status)

    def test_controls_disabled_for_cpu_or_no_driver(self):
        for cl, backend in [(None, "auto"), (self.cl, "cpu")]:
            with self.subTest(backend=backend), patch.object(app, "cl", cl):
                p, d, status = app.update_compute_controls(backend)
            self.assertFalse(p.interactive)
            self.assertFalse(d.interactive)
            self.assertIn("CPU", status)

    def test_batches_respect_memory_limits(self):
        small = device(1, global_mem_size=8000, max_mem_alloc_size=512)
        rows = app.opencl_batch_paths(small, 12, 10000, 8)
        self.assertGreater(rows, 0)
        self.assertLessEqual(rows * 13 * 8, 512)
        self.assertLessEqual(rows * 25 * 8, 2000)
        with self.assertRaisesRegex(RuntimeError, "too little memory"):
            app.opencl_batch_paths(small, 1000, 10000, 8)


class GradioIntegrationTests(unittest.TestCase):
    def test_float32_mean_path_does_not_accumulate_rounding_error(self):
        history = pd.Series([800001.0], index=pd.to_datetime(["2024-01-31"]))
        paths = np.full((50000, 13), 800001.0, dtype=np.float32)
        _, simulation_plot, _, _ = app.create_zillow_plots(history, paths, "07974", 12)
        np.testing.assert_array_equal(simulation_plot.data[-1].y, 800001.0)

    def test_run_button_passes_all_device_selections(self):
        events = [fn for fn in app.demo.fns.values() if fn.fn is app.analyze_zillow_simulation]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].inputs[-3:], [app.backend_input, app.platform_input, app.device_input])

    def test_analysis_produces_three_plots_and_actual_backend(self):
        history = pd.Series(np.linspace(90, 100, 24), index=pd.date_range("2023-01-31", periods=24, freq="ME"))
        with patch.object(app, "fetch_and_prepare_zillow_data", return_value=(history, 100, 0.002, 0.015)), \
                patch.object(app, "cl", None):
            plots_and_status = app.analyze_zillow_simulation("07974", "10y", 12, 8)
        self.assertTrue(all(len(fig.data) > 0 for fig in plots_and_status[:3]))
        self.assertIn("NumPy CPU", plots_and_status[-1])
        self.assertIn("Processing finished", plots_and_status[-1])


class OpenCLExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices, _ = app.discover_opencl_devices()
        if not cls.devices:
            raise unittest.SkipTest("No OpenCL device installed")

    def test_batched_kernels_match_seeded_numpy_for_both_precisions(self):
        for precision, tolerance in [("float32", 3e-5), ("float64", 1e-12)]:
            eligible = [d for d in self.devices if precision == "float32" or d["precision"] == precision]
            if not eligible:
                continue
            context, queue = app.get_opencl_context_queue(eligible[0]["device"])
            for s0, mu, sigma in [(800000, 0.003, 0.04), (0.02, -0.1, 0.04)]:
                with self.subTest(precision=precision, s0=s0):
                    steps, paths = 240, 17
                    np.random.seed(42)
                    expected = app.run_monte_carlo_simulation_cpu(s0, mu, sigma, steps, paths)
                    np.random.seed(42)
                    # Force multiple batches and an uneven final batch.
                    budget = (2 * steps + 1) * np.dtype(precision).itemsize * 3
                    with patch.object(app, "OPENCL_BATCH_TARGET_BYTES", budget):
                        actual = app.run_monte_carlo_simulation_opencl(
                            context, queue, s0, mu, sigma, steps, paths, precision,
                        )
                    self.assertEqual(actual.shape, (paths, steps + 1))
                    self.assertEqual(actual.dtype, np.dtype(precision))
                    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=1e-8)

    def test_program_cache_is_bound_to_context_not_device_name(self):
        device = self.devices[0]["device"]
        # Two contexts for the very same device/name must have separate programs.
        for context in (app.cl.Context([device]), app.cl.Context([device])):
            program = app._get_opencl_program(context, "float32")
            self.assertEqual(program.context.int_ptr, context.int_ptr)
            queue = app.cl.CommandQueue(context)
            paths = app.run_monte_carlo_simulation_opencl(context, queue, 100, 0, 0, 12, 8, "float32")
            np.testing.assert_array_equal(paths, 100)

    def test_real_selected_opencl_device_reports_acceleration(self):
        item = self.devices[0]
        paths, status = app.run_simulation(100, 0, 0, 12, 8, device_key=item["key"])
        np.testing.assert_array_equal(paths, 100)
        self.assertIn("PyOpenCL", status)
        self.assertIn(item["name"], status)

    def test_kernel_failure_falls_back_to_cpu(self):
        with patch.object(app, "_get_opencl_program", side_effect=RuntimeError("BUILD_PROGRAM_FAILURE")):
            paths, status = app.run_simulation(100, 0, 0, 12, 8, device_key=self.devices[0]["key"])
        np.testing.assert_array_equal(paths, 100)
        self.assertIn("NumPy CPU", status)
        self.assertIn("BUILD_PROGRAM_FAILURE", status)


if __name__ == "__main__":
    unittest.main()
