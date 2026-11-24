"""One-command launcher: installs dependencies, builds the UI, starts the server
and opens the browser. Used by start.sh and start.bat.

Options:
  --port N        port to serve on (default 8000)
  --no-browser    do not open the browser
  --test          install, then run the test suite instead of starting the app
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# Python versions the geometry libraries ship ready-made packages for, preferred first.
SUPPORTED_MINORS = (13, 14, 12, 11, 10)


def is_supported(version: tuple[int, int]) -> bool:
    return version[0] == 3 and version[1] in SUPPORTED_MINORS


def find_supported_python() -> str | None:
    """Look for another installed Python that the libraries support."""
    for minor in SUPPORTED_MINORS:
        candidates = [["py", f"-3.{minor}"]] if os.name == "nt" else []
        candidates.append([f"python3.{minor}"])
        for cmd in candidates:
            try:
                out = subprocess.run(
                    [*cmd, "-c", "import sys; print(sys.executable)"],
                    capture_output=True, text=True, timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[-1]
    return None


def ensure_supported_python() -> None:
    """Re-run this script with a supported Python, or stop with a clear message."""
    current = sys.version_info[:2]
    if is_supported(current):
        return
    other = find_supported_python()
    if other:
        print(f"Python {current[0]}.{current[1]} is not supported by the geometry libraries yet; "
              f"using {other} instead.", flush=True)
        raise SystemExit(subprocess.run([other, str(Path(__file__).resolve()), *sys.argv[1:]]).returncode)
    hint = ("  py install 3.13\n(or download it from https://www.python.org/downloads/)"
            if os.name == "nt" else "  install python3.13 with your package manager or from python.org")
    sys.exit(
        f"\nPython {current[0]}.{current[1]} cannot be used: the geometry libraries (scikit-image, "
        "scipy) have no ready-made packages for it.\n"
        "Install Python 3.13 next to it, then run the start script again:\n" + hint + "\n"
    )


def venv_version() -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            [str(VENV_PYTHON), "-c", "import sys; print(*sys.version_info[:2])"],
            capture_output=True, text=True, timeout=30,
        )
        major, minor = out.stdout.split()
        return int(major), int(minor)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def run(cmd: list[str], cwd: Path) -> None:
    print(f"> {' '.join(str(c) for c in cmd)}", flush=True)
    if subprocess.run(cmd, cwd=cwd).returncode != 0:
        sys.exit(f"\nStep failed: {' '.join(str(c) for c in cmd)}\nSee the messages above.")


def stamp_changed(stamp: Path, *sources: Path) -> str | None:
    """Return the new digest if the sources changed since the stamp was written."""
    digest = hashlib.sha256(b"".join(s.read_bytes() for s in sources if s.exists())).hexdigest()
    if stamp.exists() and stamp.read_text() == digest:
        return None
    return digest


def install_backend() -> None:
    if VENV.exists() and (not VENV_PYTHON.exists() or not is_supported(venv_version() or (0, 0))):
        print("Removing .venv made with an unsupported Python…", flush=True)
        shutil.rmtree(VENV)
    if not VENV_PYTHON.exists():
        print("Creating Python virtual environment (.venv)…", flush=True)
        venv.create(VENV, with_pip=True)
    stamp = VENV / ".install-stamp"
    digest = stamp_changed(stamp, BACKEND / "requirements.txt")
    if digest:
        run([str(VENV_PYTHON), "-m", "pip", "install", "-q", "--only-binary", ":all:",
             "-r", "requirements.txt"], BACKEND)
        stamp.write_text(digest)


def build_frontend() -> None:
    npm = shutil.which("npm")
    if not npm:
        if (FRONTEND / "dist" / "index.html").exists():
            print("Node.js not found: using the pre-built UI in frontend/dist.", flush=True)
            return
        sys.exit("Node.js (npm) was not found and there is no pre-built UI in frontend/dist.\n"
                 "Install Node.js 20 or newer from https://nodejs.org and run the start script again.")
    stamp = FRONTEND / "node_modules" / ".install-stamp"
    digest = stamp_changed(stamp, FRONTEND / "package.json", FRONTEND / "package-lock.json")
    if digest:
        run([npm, "install", "--no-audit", "--no-fund"], FRONTEND)
        stamp.write_text(
            stamp_changed(stamp, FRONTEND / "package.json", FRONTEND / "package-lock.json") or digest
        )
    run([str(VENV_PYTHON), "-m", "voronoi_engine.export_schema"], BACKEND)
    run([npm, "run", "build"], FRONTEND)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Voronoi Body Generator")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    ensure_supported_python()
    install_backend()
    if args.test:
        run([str(VENV_PYTHON), "-m", "pytest", "-q"], BACKEND)
        return
    build_frontend()

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nVoronoi Body Generator is running at {url}  (Ctrl+C to stop)\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    try:
        subprocess.run(
            [str(VENV_PYTHON), "-m", "uvicorn", "voronoi_server.app:app",
             "--host", "127.0.0.1", "--port", str(args.port)],
            cwd=BACKEND, check=False,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
