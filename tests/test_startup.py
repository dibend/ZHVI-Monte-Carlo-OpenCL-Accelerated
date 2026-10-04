"""Login configuration, preset clicks, and named-resource regression checks."""
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import app


class AuthenticationTests(unittest.TestCase):
    def test_unset_credentials_keep_login_optional(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(app.get_gradio_auth())

    def test_password_is_preserved_exactly(self):
        credentials = {"GRADIO_USERNAME": "david", "GRADIO_PASSWORD": " space:$'\\test "}
        with patch.dict(os.environ, credentials, clear=True):
            self.assertEqual(app.get_gradio_auth(), ("david", credentials["GRADIO_PASSWORD"]))

    def test_incomplete_or_empty_credentials_cannot_start_a_server(self):
        for values in [
            {"GRADIO_USERNAME": "david"},
            {"GRADIO_PASSWORD": "test-only-password"},
            {"GRADIO_USERNAME": "", "GRADIO_PASSWORD": "test-only-password"},
            {"GRADIO_USERNAME": "david", "GRADIO_PASSWORD": ""},
            {"GRADIO_USERNAME": "", "GRADIO_PASSWORD": ""},
        ]:
            with self.subTest(keys=list(values)), patch.dict(os.environ, values, clear=True), \
                    patch.object(app.demo, "launch") as launch:
                with self.assertRaisesRegex(ValueError, "Set both GRADIO_USERNAME"):
                    app.launch_app()
                launch.assert_not_called()


class PresetTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_preset_fills_the_original_four_inputs(self):
        expected_rows = [
            ["80132", "max", 120, 250000],
            ["07074", "10y", 60, 150000],
            ["90210", "10y", 60, 100000],
            ["07974", "15y", 120, 200000],
            ["80132", "max", 180, 500000],
            ["33139", "5y", 36, 100000],
        ]
        event = next(fn for fn in app.demo.fns.values() if fn.inputs == [app.example_dataset])
        self.assertEqual(event.outputs, [
            app.zip_input, app.hist_period_input, app.sim_months_input, app.num_paths_input,
        ])
        for index, expected in enumerate(expected_rows):
            with self.subTest(index=index):
                result = await app.demo.process_api(event, [index])
                self.assertEqual(result["data"], expected)


class NamedResourceTests(unittest.TestCase):
    @unittest.skipUnless("forkserver" in mp.get_all_start_methods(), "POSIX forkserver required")
    def test_building_the_ui_does_not_register_a_named_semaphore(self):
        # forkserver also exercises the named-lock behavior of Python 3.14
        # on Linux when the test runner uses an older Python version.
        script = textwrap.dedent("""\
            import multiprocessing as mp
            mp.set_start_method("forkserver")
            import gradio
            import multiprocessing.resource_tracker as tracker
            registered = []
            original_register = tracker.register
            def record(name, kind):
                registered.append(kind)
                return original_register(name, kind)
            tracker.register = record
            import app
            assert "semaphore" not in registered, registered
            print("No named semaphore created")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "GRADIO_ANALYTICS_ENABLED": "False"},
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("No named semaphore created", result.stdout)
        self.assertNotIn("leaked semaphore", result.stderr)


if __name__ == "__main__":
    unittest.main()
