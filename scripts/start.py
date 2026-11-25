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
import re
import shutil
import subprocess
import sys
import itertools
import json
import tempfile
import threading
import time
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
LOG = ROOT / "install.log"  # full output of the install steps, for when one fails
STEPS = 4  # check Python, Python packages, user interface, start


class Progress:
    """One line per step and a progress bar redrawn in place, so it is clear the
    script is working (and the window must stay open) during the long first install."""

    WIDTH = 30

    def __init__(self) -> None:
        self.tty = sys.stdout.isatty()
        self.fraction: float | None = None  # None: still working, amount unknown
        self.detail = ""
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.ticker: threading.Thread | None = None
        self.stop = threading.Event()
        self.printed_detail = ""

    def step(self, number: int, title: str) -> None:
        self.finish()
        print(f"[{number}/{STEPS}] {title}", flush=True)

    def start(self, detail: str, fraction: float | None = None) -> None:
        """Show a bar under the current step until finish() is called."""
        self.finish()
        self.fraction, self.detail, self.started = fraction, detail, time.monotonic()
        self.stop.clear()
        if self.tty:
            self.ticker = threading.Thread(target=self._tick, daemon=True)
            self.ticker.start()
        else:
            self._plain()

    def update(self, fraction: float | None = None, detail: str | None = None) -> None:
        with self.lock:
            if fraction is not None:
                self.fraction = max(self.fraction or 0.0, min(fraction, 1.0))
            if detail is not None:
                self.detail = detail
        if not self.tty:
            self._plain()

    def finish(self, message: str | None = None) -> None:
        if self.ticker is not None:
            self.stop.set()
            self.ticker.join()
            self.ticker = None
            if message is None:  # nothing to keep: clear the bar's line
                columns = shutil.get_terminal_size((80, 20)).columns - 1
                sys.stdout.write("\r" + " " * columns + "\r")
                sys.stdout.flush()
            else:
                self.fraction = 1.0
                self.detail = message
                self._draw(final=True)
        elif message is not None:
            print(f"      {message}", flush=True)

    def _tick(self) -> None:
        for frame in itertools.count():
            self._draw(frame=frame)
            if self.stop.wait(0.2):
                return

    def _draw(self, frame: int = 0, final: bool = False) -> None:
        with self.lock:
            fraction, detail = self.fraction, self.detail
        elapsed = int(time.monotonic() - self.started)
        clock = f"{elapsed // 60}:{elapsed % 60:02d}"
        if fraction is None:  # a block sliding back and forth
            pos = frame % (2 * (self.WIDTH - 4))
            pos = pos if pos < self.WIDTH - 4 else 2 * (self.WIDTH - 4) - pos
            bar = "-" * pos + "####" + "-" * (self.WIDTH - 4 - pos)
            percent = "    "
        else:
            filled = round(fraction * self.WIDTH)
            bar = "#" * filled + "-" * (self.WIDTH - filled)
            percent = f"{round(fraction * 100):3d}%"
        line = f"      [{bar}] {percent} {clock}  {detail}"
        columns = shutil.get_terminal_size((80, 20)).columns - 1
        line = line[:columns].ljust(columns)
        sys.stdout.write("\r" + line + ("\n" if final else ""))
        sys.stdout.flush()

    def _plain(self) -> None:
        """Output redirected to a file: no redrawing, just each new detail once."""
        if self.detail and self.detail != self.printed_detail:
            self.printed_detail = self.detail
            print(f"      {self.detail}", flush=True)


progress = Progress()


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


def run(cmd: list[str], cwd: Path, on_line=None) -> None:
    """Run a step with its output going to install.log; stop with the log's end if it fails."""
    command = " ".join(str(c) for c in cmd)
    with open(LOG, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n> {command}\n")
        log.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as error:
            progress.finish()
            sys.exit(f"\nStep failed: {command}\n{error}")
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if on_line is not None:
                on_line(line.rstrip())
        code = proc.wait()
    if code != 0:
        progress.finish()
        tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        sys.exit("\n".join(["", f"Step failed: {command}", "Last lines of its output:", *tail,
                            "", f"The full output is in {LOG}"]))


def stamp_changed(stamp: Path, *sources: Path) -> str | None:
    """Return the new digest if the sources changed since the stamp was written."""
    digest = hashlib.sha256(b"".join(s.read_bytes() for s in sources if s.exists())).hexdigest()
    if stamp.exists() and stamp.read_text() == digest:
        return None
    return digest


def packages_to_install() -> int | None:
    """How many packages pip will install (a dry run), or None if pip cannot tell."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        try:
            run([str(VENV_PYTHON), "-m", "pip", "install", "--dry-run", "--quiet",
                 "--only-binary", ":all:", "--report", str(report), "-r", "requirements.txt"],
                BACKEND)
            return len(json.loads(report.read_text(encoding="utf-8"))["install"]) or None
        except (SystemExit, OSError, ValueError, KeyError):
            return None  # the real install below reports any actual problem


def pip_line_handler(total: int | None):
    """Turns pip's output lines into progress. pip first lists every package (up to 25 %),
    then downloads those not in its cache (up to 80 %), then unpacks and installs them."""
    listed: set[str] = set()
    downloaded: set[str] = set()

    def plain_name(text: str) -> str:
        return re.split(r"[<>=!~;\[ (]", text, maxsplit=1)[0]

    def on_line(line: str) -> None:
        text = line.strip()
        if text.startswith("Collecting "):
            name = plain_name(text.split()[1])
            listed.add(name)
            label = f"{len(listed)} of {total}" if total else f"{len(listed)}"
            progress.update(0.25 * len(listed) / total if total else None,
                            f"Finding package {label}: {name}")
        elif text.startswith("Downloading ") and "(" in text:
            file = text.split()[1].rsplit("/", 1)[-1]
            name = file.split("-")[0]
            size = text[text.rfind("(") + 1:text.rfind(")")]
            downloaded.add(name)
            label = f" {len(downloaded)} of {total}" if total else ""
            progress.update(0.25 + 0.55 * len(downloaded) / total if total else None,
                            f"Downloading{label}: {name} ({size})")
        elif text.startswith("Installing collected packages"):
            count = text.count(",") + 1
            progress.update(0.8, f"Unpacking and installing {count} packages (the slowest part)")
        elif text.startswith("Successfully installed"):
            progress.update(1.0, "Installed")

    return on_line


def install_backend() -> None:
    progress.step(2, "Python packages")
    if VENV.exists() and (not VENV_PYTHON.exists() or not is_supported(venv_version() or (0, 0))):
        progress.start("Removing .venv made with an unsupported Python")
        shutil.rmtree(VENV)
    if not VENV_PYTHON.exists():
        progress.start("Creating the Python environment (.venv)")
        venv.create(VENV, with_pip=True)
    stamp = VENV / ".install-stamp"
    digest = stamp_changed(stamp, BACKEND / "requirements.txt")
    if not digest:
        progress.finish("Already installed")
        return
    print("      First start (or new requirements): this takes a few minutes.\n"
          "      Do not close this window.", flush=True)
    progress.start("Checking which packages are needed")
    total = packages_to_install()
    progress.start("Starting the download", 0.0 if total else None)
    run([str(VENV_PYTHON), "-m", "pip", "install", "--progress-bar", "off",
         "--only-binary", ":all:", "-r", "requirements.txt"], BACKEND, pip_line_handler(total))
    stamp.write_text(digest)
    progress.finish(f"Installed {total} packages" if total else "Installed")


def build_frontend() -> None:
    progress.step(3, "User interface")
    npm = shutil.which("npm")
    if not npm:
        if (FRONTEND / "dist" / "index.html").exists():
            progress.finish("Node.js not found: using the pre-built UI in frontend/dist")
            return
        sys.exit("Node.js (npm) was not found and there is no pre-built UI in frontend/dist.\n"
                 "Install Node.js 20 or newer from https://nodejs.org and run the start script again.")
    stamp = FRONTEND / "node_modules" / ".install-stamp"
    digest = stamp_changed(stamp, FRONTEND / "package.json", FRONTEND / "package-lock.json")
    if digest:
        progress.start("Installing the UI build tools (npm install); do not close this window")
        run([npm, "install", "--no-audit", "--no-fund"], FRONTEND)
        stamp.write_text(
            stamp_changed(stamp, FRONTEND / "package.json", FRONTEND / "package-lock.json") or digest
        )
    progress.start("Building the UI")
    run([str(VENV_PYTHON), "-m", "voronoi_engine.export_schema"], BACKEND)
    run([npm, "run", "build"], FRONTEND)
    progress.finish("Built")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Voronoi Body Generator")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    print("Voronoi Body Generator - keep this window open while you use the app.\n", flush=True)
    progress.step(1, "Python")
    ensure_supported_python()
    progress.finish(f"Python {sys.version_info[0]}.{sys.version_info[1]}")
    install_backend()
    if args.test:
        progress.step(STEPS, "Test suite (a long run: about 25 minutes)")
        # Shown live rather than logged: pytest prints its own progress.
        raise SystemExit(subprocess.run([str(VENV_PYTHON), "-m", "pytest", "-q"], cwd=BACKEND).returncode)
    build_frontend()

    url = f"http://127.0.0.1:{args.port}"
    progress.step(STEPS, "Start")
    print("\n" + "\n".join([
        "  " + "=" * 66,
        f"  Voronoi Body Generator is running at {url}",
        "  KEEP THIS WINDOW OPEN while you use the app: closing it stops the app.",
        "  To stop the app, press Ctrl+C here (or close this window).",
        "  " + "=" * 66,
    ]) + "\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    try:
        # Warnings and errors only, so the message above stays in view.
        subprocess.run(
            [str(VENV_PYTHON), "-m", "uvicorn", "voronoi_server.app:app",
             "--host", "127.0.0.1", "--port", str(args.port), "--log-level", "warning"],
            cwd=BACKEND, check=False,
        )
    except KeyboardInterrupt:
        pass
    print("\nThe app has stopped.", flush=True)


if __name__ == "__main__":
    main()
