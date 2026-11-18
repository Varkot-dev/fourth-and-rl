"""M7 interactive pieces: the app bundle, its loader and lookups, the static page builder,
the Streamlit app and the notebook.

Most tests use a tiny synthetic bundle (`app_bundle.synthetic_bundle`) and run in CI without
data. Tests of the committed bundle skip when it is absent; Streamlit, Plotly and nbclient
tests skip when those packages are not installed.
"""

from __future__ import annotations

import base64
import gzip
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from fourth_and_rl import app_bundle as ab
from fourth_and_rl import bootstrap
from fourth_and_rl.paths import ROOT

APP = ROOT / "app" / "streamlit_app.py"
NOTEBOOK = ROOT / "notebooks" / "fourth_down_demo.ipynb"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def synth() -> ab.Bundle:
    return ab.synthetic_bundle(0)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory: pytest.TempPathFactory, synth: ab.Bundle) -> Path:
    d = tmp_path_factory.mktemp("bundle")
    ab.save_bundle(synth, d)
    return d


# ------------------------------------------------------------------------------ schema


def test_contexts_follow_the_grid_rules() -> None:
    grid = {"clocks": [{"seconds": 900}, {"seconds": 240}], "endgame_seconds": 300,
            "opp_timeouts": [0, 3], "spreads": [-7, 0, 7], "spread_min_seconds": 600}  # fmt: skip
    cs = ab.contexts(grid)
    assert [(c.seconds, c.opp_timeouts, c.spread) for c in cs] == [
        (900, 3, -7.0), (900, 3, 0.0), (900, 3, 7.0), (240, 0, 0.0), (240, 3, 0.0)]
    assert [(c.seconds, c.opp_timeouts) for c in ab.exact_contexts(grid)] == [
        (900, 3), (240, 0), (240, 3)]


def test_encodings_round_trip_with_missing_values() -> None:
    x = np.array([0.0, 0.12345, 0.99999, 1.0, np.nan])
    back = ab.decode_wp(ab.encode_wp(x, 10000), 10000)
    assert np.allclose(back[:4], [0.0, 0.1234, 1.0, 1.0], atol=1e-4)
    assert np.isnan(back[4])
    g = np.array([-0.2, 0.0, 0.05, np.nan, 9.0])
    gb = ab.decode_gain(ab.encode_gain(g, 10000), 10000)
    assert np.allclose(gb[:3], g[:3])
    assert np.isnan(gb[3]) and gb[4] == pytest.approx(32767 / 10000)


@pytest.mark.parametrize("delta", [True, False])
def test_save_and_load_round_trip(tmp_path: Path, synth: ab.Bundle, delta: bool) -> None:
    sizes = ab.save_bundle(synth, tmp_path, delta=delta)
    assert set(sizes) == {ab.META_FILE, ab.ONE_STEP_FILE, ab.EXACT_FILE, ab.QUESTIONS_FILE}
    back = ab.load_bundle(tmp_path)
    for name in ("one_step", "exact"):
        a, b = getattr(synth, name), getattr(back, name)
        assert set(a) == set(b)
        for k in a:
            assert b[k].dtype == a[k].dtype and np.array_equal(a[k], b[k]), (name, k)
    assert back.meta["axes"] == synth.meta["axes"]


def test_validate_rejects_bad_bundles(synth: ab.Bundle) -> None:
    bad = ab.Bundle(synth.meta, {**synth.one_step, "go": synth.one_step["go"].astype(np.int32)},
                    synth.exact)  # fmt: skip
    with pytest.raises(ValueError, match="one_step.go"):
        ab.validate(bad)
    go = synth.one_step["go"].copy()
    go[0, 0, 2, 0] = 5000  # distance 3 at yard line 1 cannot happen
    with pytest.raises(ValueError, match="impossible cells"):
        ab.validate(ab.Bundle(synth.meta, {**synth.one_step, "go": go}, synth.exact))
    missing = {k: v for k, v in synth.one_step.items() if k != "n_fg"}
    with pytest.raises(ValueError, match="n_fg is missing"):
        ab.validate(ab.Bundle(synth.meta, missing, synth.exact))


def test_load_bundle_without_files_says_how_to_build(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="m7_build_app_bundle"):
        ab.load_bundle(tmp_path)


# ----------------------------------------------------------------------------- lookups


def test_lookup_reads_the_right_cell(synth: ab.Bundle) -> None:
    cell = synth.lookup(seconds=120, score=1, yardline_100=3, ydstogo=2, opp_timeouts=0)
    i = synth.contexts.index(ab.Context(120, 0, 0.0))
    j, d, y = synth.scores.index(1), 1, 2
    for a, k in zip(ab.ACTIONS, ("go", "fg", "punt"), strict=True):
        assert cell["one_step"]["wp"][a] == pytest.approx(synth.one_step[k][i, j, d, y] / 10000)
    e = synth.exact_contexts.index(ab.Context(120, 0, 0.0))
    assert cell["exact"]["wp"]["go"] == pytest.approx(synth.exact["go"][e, j, d, y] / 10000)
    assert cell["exact"]["distance_binned"] is False
    one = cell["one_step"]
    vals = np.array([one["wp"][a] for a in ab.ACTIONS])
    assert one["best"] == ab.ACTIONS[int(np.nanargmax(vals))]
    kick = np.nanmax(vals[1:])
    assert one["go_gain"] == pytest.approx(vals[0] - kick)


def test_category_matches_the_bootstrap_rule(synth: ab.Bundle) -> None:
    th = synth.meta["bootstrap"]["thresholds"]
    for sec, score, yl, d, to in ((900, 0, 5, 1, 3), (120, -1, 4, 3, 3), (120, 1, 2, 2, 0)):
        one = synth.lookup(sec, score, yl, d, to)["one_step"]
        expect = bootstrap.classify(np.array([one["boot_pct"]]), th["confident"], th["lean"])[0]
        assert one["category"] == expect


def test_snapping_is_reported_not_hidden(synth: ab.Bundle) -> None:
    cell = synth.lookup(seconds=130, score=0, yardline_100=5, ydstogo=1, opp_timeouts=2,
                        spread=7.0)  # fmt: skip
    assert cell["context"] == ab.Context(120, 3, 0.0)
    assert any("clock" in n for n in cell["notes"])
    assert any("spread" in n for n in cell["notes"])
    assert any("timeouts" in n for n in cell["notes"])
    assert synth.lookup(900, 0, 5, 1, 3, 7.0)["notes"] == []


def test_lookup_refuses_impossible_or_off_grid_requests(synth: ab.Bundle) -> None:
    with pytest.raises(ValueError):
        synth.lookup(900, 0, yardline_100=2, ydstogo=3)
    with pytest.raises(KeyError):
        synth.lookup(900, 5, yardline_100=4, ydstogo=1)


def test_chart_agrees_with_lookups(synth: ab.Bundle) -> None:
    ch = synth.chart(120, 1, 3, 0.0)
    n_d, n_y = len(synth.distances), len(synth.yardlines)
    assert ch["best"].shape == (n_d, n_y)
    ok = ab.valid_mask(synth.distances, synth.yardlines)
    assert np.all(ch["best"][~ok] == -1) and np.all(ch["category"][~ok] == -1)
    codes = {"confident": 0, "lean": 1, "uncertain": 2}
    for di, d in enumerate(synth.distances):
        for yi, y in enumerate(synth.yardlines):
            if d > y:
                continue
            one = synth.lookup(120, 1, y, d, 3)["one_step"]
            assert ab.ACTIONS[ch["best"][di, yi]] == one["best"]
            assert ch["category"][di, yi] == codes[one["category"]]


# ------------------------------------------------------------------------- static page


def _decode_payload(html: str) -> tuple[dict, dict[str, np.ndarray]]:
    """The page's JavaScript decoder, in Python."""
    meta = json.loads(html.split('<script id="meta" type="application/json">')[1]
                      .split("</script>")[0])  # fmt: skip
    b64 = html.split('<script id="data" type="text/plain">')[1].split("</script>")[0]
    raw = gzip.decompress(base64.b64decode(b64))
    out = {}
    for m in meta["arrays"]:
        chunk = raw[m["offset"] : m["offset"] + m["bytes"]]
        if m["dtype"] == "uint8":
            out[m["name"]] = np.frombuffer(chunk, dtype=np.uint8).reshape(m["shape"])
        else:
            d = np.frombuffer(chunk, dtype="<i2").reshape(m["shape"])
            out[m["name"]] = np.cumsum(d.astype(np.int64), axis=-1)
    return meta, out


def test_static_page_carries_the_bundle_exactly(synth: ab.Bundle) -> None:
    page = _load_script("m7_build_static_page")
    cdn = "https://cdn.plot.ly/plotly-4.1.1.min.js"
    html = page.build_page(synth, cdn)
    assert f'<script src="{cdn}"></script>' in html and "{{" not in html
    meta, arr = _decode_payload(html)
    assert meta["synthetic"] is True and meta["scores"] == synth.scores
    go = synth.one_step["go"]
    expect = np.where(go == ab.WP_NA, 0, np.maximum(go, 1))
    assert np.array_equal(arr["one_go"], expect)
    assert np.array_equal(arr["n_fg"], synth.one_step["n_fg"])
    lo = synth.one_step["gain_lo"]
    assert np.array_equal(arr["gain_lo"], np.where(lo == ab.GAIN_NA, 0, lo))
    ex = synth.exact["punt"]
    assert np.array_equal(arr["exact_punt"], np.where(ex == ab.WP_NA, 0, np.maximum(ex, 1)))


def test_static_page_slices_spreads_and_scores(synth: ab.Bundle) -> None:
    page = _load_script("m7_build_static_page")
    html = page.build_page(synth, "https://cdn.plot.ly/plotly-4.1.1.min.js", scores=[0, 1],
                           spreads=[0])  # fmt: skip
    meta, arr = _decode_payload(html)
    keep = [i for i, c in enumerate(synth.contexts) if c.spread == 0]
    assert len(meta["contexts"]) == len(keep) < len(synth.contexts)
    assert meta["all_spreads"] == [0.0, 7.0]
    assert arr["n_go"].shape[:2] == (len(keep), 2)
    assert np.array_equal(arr["n_go"], synth.one_step["n_go"][keep][:, [1, 2]])


def test_committed_static_page_is_within_budget() -> None:
    cfg = ab.load_config()["static_page"]
    path = ROOT / cfg["out"]
    if not path.exists():
        pytest.skip("static page not built (scripts/m7_build_static_page.py)")
    html = path.read_text()
    assert path.stat().st_size / 1e6 <= cfg["max_megabytes"]
    assert cfg["plotly_cdn"] in html and "{{" not in html
    meta, _ = _decode_payload(html)
    assert meta["synthetic"] is False


# ------------------------------------------------------------------------------ figures


def test_figures_build_on_the_synthetic_bundle(synth: ab.Bundle) -> None:
    pytest.importorskip("plotly")
    from fourth_and_rl import app_figures as af

    p = af.palette()
    assert len(p["cells"]) == 9 and p["cells"][0] == p["actions"]["go"]
    ch = synth.chart(900, 0, 3, 0.0)
    fig = af.field_chart(ch, synth.yardlines, synth.distances, current=(4, 2))
    heat = [t for t in fig.data if t.type == "heatmap"][0]
    assert np.asarray(heat.z).shape == (len(synth.distances), len(synth.yardlines))
    af.action_bars(synth.lookup(900, 0, 4, 2))
    assert af.field_label(75) == "Own 25" and af.field_label(30) == "Opp 30"
    assert af.score_label(-3) == "Down 3"


# ------------------------------------------------------------------------------ the app


def test_app_runs_on_the_synthetic_bundle(synth_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FOURTH_AND_RL_BUNDLE", str(synth_dir))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Synthetic test bundle" in c.value for c in at.caption)
    assert [m.label for m in at.metric] == ["Go for it", "Field goal", "Punt"]
    at.select_slider(key="clock").set_value("Q4 2:00").run()
    assert not at.exception
    at.radio(key="which").set_value("Exact full game").run()
    assert not at.exception


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_app_starts_headless(synth_dir: Path) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    port = _free_port()
    env = {**os.environ, "FOURTH_AND_RL_BUNDLE": str(synth_dir)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        deadline = time.time() + 60
        ok = False
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health",
                                            timeout=2) as r:  # fmt: skip
                    ok = r.status == 200
                    break
            except OSError:
                time.sleep(0.5)
        assert ok, "streamlit did not answer its health check"
        time.sleep(2)
        assert proc.poll() is None, "streamlit exited"
    finally:
        proc.terminate()
        proc.wait(timeout=30)


# ---------------------------------------------------------------- the committed bundle


def _committed() -> ab.Bundle:
    try:
        return ab.load_bundle()
    except FileNotFoundError:
        pytest.skip("app bundle not built (scripts/m7_build_app_bundle.py)")


def test_committed_bundle_matches_saved_results() -> None:
    b = _committed()
    m = b.meta
    assert m["synthetic"] is False and m["bootstrap"]["n_boot"] == 101
    total = sum((ab.default_dir() / n).stat().st_size for n in
                (ab.META_FILE, ab.ONE_STEP_FILE, ab.EXACT_FILE, ab.QUESTIONS_FILE))  # fmt: skip
    assert total < 20e6
    assert m["checks"]["question1_exact_map"]["max_abs_difference"] <= 1e-4
    # Question 1 headline: the exact values and the one-step gain saved by the M4 script.
    q1 = json.loads((ROOT / "results/questions/q1_late_lead_dp.json").read_text())
    h = q1["headline"]["situation"]
    cell = b.lookup(h["seconds"], h["lead"], h["yardline_100"], h["ydstogo"],
                    h["defense_timeouts"], 0.0)  # fmt: skip
    assert cell["exact"]["wp"]["go"] == pytest.approx(q1["headline"]["exact"]["go"], abs=1e-4)
    assert cell["exact"]["wp"]["punt"] == pytest.approx(q1["headline"]["exact"]["punt"], abs=1e-4)
    one_step = q1["one_step_comparison"]["headline_gap_accounting"]["one_step_go_minus_punt"]
    assert cell["one_step"]["go_gain"] == pytest.approx(one_step, abs=2e-4)
    assert m["one_step"]["wp_model_stamp"]["allow_test_seasons"] is False


# ----------------------------------------------------------------------------- notebook


@pytest.mark.data
def test_notebook_runs_end_to_end() -> None:
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    _committed()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if o.output_type == "error"]  # fmt: skip
    assert not errors
    images = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if "data" in o and "image/png" in o["data"]]  # fmt: skip
    assert len(images) == 2
