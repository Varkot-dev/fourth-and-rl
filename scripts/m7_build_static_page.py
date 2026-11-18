"""M7: the static fourth-down explorer page, built from the app bundle.

One self-contained HTML file (app/fourth_down_explorer.html) that needs no server, so it
can be hosted on GitHub Pages: plotly comes from its CDN, pinned to an exact version
(configs/app.yaml, `static_page.plotly_cdn`), and the data is embedded as one gzip block
that the browser unpacks with DecompressionStream.

What the page carries (from app/data/, nothing recomputed; `static_page.spreads` and
`scores` in configs/app.yaml slice it, and the page says which slice it holds): for every
grid cell the one-step go, field goal and punt win probabilities, the bootstrap counts
preferring each action and the 90% interval of the go gain, and the exact game's values. It leaves
out the per-action intervals, which the Streamlit app shows. Win probabilities are stored
as 1/10,000 units with 0 meaning "not available" (a real value of exactly 0.0000 is
stored as 0.0001), as int16 differences along the yard-line axis before gzip.

Run:  python scripts/m7_build_static_page.py
      python scripts/m7_build_static_page.py --bundle <folder> --out <file.html>
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np

from fourth_and_rl import app_bundle as ab
from fourth_and_rl import app_figures as af
from fourth_and_rl.paths import ROOT

TEMPLATE = ROOT / "app" / "static_page_template.html"


def _wp_page(a: np.ndarray) -> np.ndarray:
    """Bundle uint16 win probabilities -> page uint16 with 0 = not available."""
    return np.where(a == ab.WP_NA, 0, np.maximum(a, 1)).astype(np.int32)


def _gain_page(a: np.ndarray) -> np.ndarray:
    return np.where(a == ab.GAIN_NA, 0, a).astype(np.int32)


def context_rows(b: ab.Bundle, spreads: list[float] | None) -> list[int]:
    """One-step contexts kept on the page (all, or those with one of `spreads`)."""
    return [i for i, c in enumerate(b.contexts) if spreads is None or c.spread in spreads]


def payload(
    b: ab.Bundle, scores: list[int] | None = None, spreads: list[float] | None = None
) -> tuple[bytes, dict]:
    """The packed arrays (gzip) and their manifest."""
    keep = list(range(len(b.scores))) if scores is None else [b.scores.index(s) for s in scores]
    rows = context_rows(b, spreads)
    one = {k: v[rows][:, keep] for k, v in b.one_step.items()}
    arrays: list[tuple[str, np.ndarray, bool]] = []
    for key in ("go", "fg", "punt"):
        arrays.append((f"one_{key}", _wp_page(one[key]), True))
    for key in ("gain_lo", "gain_hi"):
        arrays.append((key, _gain_page(one[key]), True))
    for key in ("n_go", "n_fg", "n_punt"):
        arrays.append((key, one[key].astype(np.uint8), False))
    for key in ("go", "fg", "punt"):
        arrays.append((f"exact_{key}", _wp_page(b.exact[key][:, keep]), True))
    blobs, manifest, offset = [], [], 0
    for name, arr, delta in arrays:
        if delta:
            d = np.diff(arr, axis=-1, prepend=0)
            if np.abs(d).max() > np.iinfo(np.int16).max:
                raise ValueError(f"{name}: differences do not fit int16")
            raw = d.astype("<i2").tobytes()
            dtype = "int16_delta"
        else:
            raw = np.ascontiguousarray(arr).tobytes()
            dtype = "uint8"
        manifest.append({"name": name, "dtype": dtype, "shape": list(arr.shape),
                         "offset": offset, "bytes": len(raw)})  # fmt: skip
        blobs.append(raw)
        offset += len(raw)
        if offset % 2:  # keep int16 views aligned
            blobs.append(b"\0")
            offset += 1
    return gzip.compress(b"".join(blobs), compresslevel=9, mtime=0), {"arrays": manifest}


def page_meta(b: ab.Bundle, scores: list[int], spreads: list[float] | None) -> dict:
    m = b.meta
    return {
        "contexts": [m["axes"]["contexts"][i] for i in context_rows(b, spreads)],
        "all_spreads": sorted({c.spread for c in b.contexts}),
        "exact_contexts": m["axes"]["exact_contexts"],
        "clocks": m["axes"]["clocks"],
        "scores": scores,
        "all_scores": b.scores,
        "distances": b.distances,
        "yardlines": b.yardlines,
        "scale": b.wp_scale,
        "n_boot": b.n_boot,
        "thresholds": m["bootstrap"]["thresholds"],
        "dist_exact": m["exact"]["dist_exact"],
        "built_at": m.get("built_at"),
        "commit": m.get("git_commit_at_build"),
        "synthetic": bool(m.get("synthetic")),
        "wp_model": m.get("one_step", {}).get("wp_model"),
        "palette": af.palette(),
    }


def build_page(
    b: ab.Bundle, cdn: str, scores: list[int] | None = None, spreads: list[float] | None = None
) -> str:
    """The page's HTML. `scores` and `spreads` slice the bundle (None keeps everything)."""
    sc = b.scores if scores is None else list(scores)
    sp = None if spreads is None else [float(x) for x in spreads]
    data, manifest = payload(b, None if scores is None else sc, sp)
    meta = {**page_meta(b, sc, sp), **manifest}
    html = TEMPLATE.read_text()
    for key, value in (("{{PLOTLY_CDN}}", cdn),
                       ("{{META_JSON}}", json.dumps(meta, separators=(",", ":"))),
                       ("{{DATA_B64}}", base64.b64encode(data).decode("ascii"))):  # fmt: skip
        if key not in html:
            raise ValueError(f"template has no {key}")
        html = html.replace(key, value)
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle", default=None, help="bundle folder (default: app/data)")
    parser.add_argument("--out", default=None, help="output file (default: config)")
    args = parser.parse_args()
    cfg = ab.load_config()["static_page"]
    b = ab.load_bundle(args.bundle)
    html = build_page(b, cfg["plotly_cdn"], cfg.get("scores"), cfg.get("spreads"))
    out = Path(args.out) if args.out else ROOT / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f"{out}: {mb:.2f} MB")
    if mb > float(cfg["max_megabytes"]):
        raise SystemExit(f"page is {mb:.1f} MB, over the {cfg['max_megabytes']} MB budget: set "
                         "static_page.scores to a slice")  # fmt: skip


if __name__ == "__main__":
    main()
