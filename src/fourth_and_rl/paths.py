"""Repository paths, resolved from this file so scripts work from any directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
