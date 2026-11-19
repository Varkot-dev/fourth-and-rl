"""M7: screenshots of the Streamlit app and of the static page (Playwright, Chromium).

Starts `streamlit run app/streamlit_app.py` headless on a free port, waits for its health
check, and saves a picture of the first tab with the default situation; then opens the
static page from disk (its plotly CDN script answered from the installed plotly package,
since this machine has no route to the CDN). Needs Node with the `playwright` package and
the app extra installed.

Outputs: assets/m7_app.png, assets/m7_static_page.png

Run:  python scripts/m7_app_screenshot.py
      FOURTH_AND_RL_BUNDLE=<folder> python scripts/m7_app_screenshot.py   (another bundle)
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from fourth_and_rl import app_bundle as ab
from fourth_and_rl.paths import ASSETS, ROOT

JS = ROOT / "scripts" / "m7_screenshot.js"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def node_env() -> dict[str, str]:
    env = dict(os.environ)
    if "NODE_PATH" not in env:
        root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
        env["NODE_PATH"] = root
    return env


def shoot(url: str, out: Path, width: int, height: int, plotly_js: Path | None = None) -> None:
    cmd = ["node", str(JS), url, str(out), str(width), str(height)]
    if plotly_js is not None:
        cmd.append(str(plotly_js))
    subprocess.run(cmd, check=True, env=node_env(), timeout=300)


def app_screenshot(out: Path, width: int, height: int) -> None:
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
         "--server.headless", "true", "--server.port", str(port),
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        deadline = time.time() + 90
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2):
                    break
            except OSError:
                if time.time() > deadline or proc.poll() is not None:
                    raise RuntimeError("streamlit did not start") from None
                time.sleep(1)
        shoot(f"http://127.0.0.1:{port}/", out, width, height)
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=["app", "page"], default=None)
    args = parser.parse_args()
    cfg = ab.load_config()
    if args.only in (None, "app"):
        app_screenshot(ASSETS / "m7_app.png", 1400, 1950)
    if args.only in (None, "page"):
        import plotly

        local = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        page = ROOT / cfg["static_page"]["out"]
        shoot(page.as_uri(), ASSETS / "m7_static_page.png", 420, 1900, local)


if __name__ == "__main__":
    main()
