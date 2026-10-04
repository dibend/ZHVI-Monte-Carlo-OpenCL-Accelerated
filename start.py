"""Launch app.py in the background on Ubuntu/Debian and other POSIX systems."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    if os.name != "posix":
        print("This background launcher requires a POSIX system (such as Ubuntu).", file=sys.stderr)
        return 1

    folder = Path(__file__).resolve().parent
    app_path = folder / "app.py"
    log_path = folder / "gradio.log"
    if not app_path.is_file():
        print(f"Cannot find {app_path}", file=sys.stderr)
        return 1

    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(app_path)],
                cwd=folder,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        print(f"Could not start Gradio: {exc}", file=sys.stderr)
        return 1

    # Catch immediate errors, such as missing dependencies. Gradio may take
    # longer to finish starting; the log contains its URLs and any later errors.
    try:
        returncode = process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        print(f"Started background process (PID: {process.pid}).")
        print(f"Log: {log_path}")
        print(f"Stop gracefully: kill -INT {process.pid}")
        return 0

    print(f"App exited during startup (code {returncode}). Check {log_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
