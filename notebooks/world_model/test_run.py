"""Tests for run.py with stub records/model/train/simulate modules (fast, CPU, no network)."""

from __future__ import annotations

import sys
import time
import types

import numpy as np
import pandas as pd
import pytest
import run as R
import schema as S


# ---------------------------------------------------------------- stubs
class _StubModel:
    def __init__(self, cfg=None):
        self.cfg = dict(cfg or {})

    def count_params(self):
        return 1234

    def eval(self):
        return self

    def to(self, device):
        return self


def _stubs(calls: dict, fail_train_once: bool = False, build_seed: int = 0,
           interrupt_wp_call: int | None = None) -> dict[str, types.ModuleType]:
    rec = types.ModuleType("records")

    def load_pbp(seasons):
        assert max(seasons) <= 2025
        calls.setdefault("load_pbp", []).append(list(seasons))
        return pd.DataFrame({"game_id": ["x"], "play_id": [1], "home_wp": [0.5],
                             "vegas_home_wp": [0.5]})

    def load_schedules(seasons):
        return pd.DataFrame({"game_id": ["x"]})

    def build_split(pbp, schedules, seasons):
        calls.setdefault("build_split", []).append(list(seasons))
        sp = R.synthetic_split(n_games=4, seed=int(seasons[0]) + build_seed)
        sp.game_ids = [f"{seasons[0]}_{g}" for g in sp.game_ids]
        return sp, {"games": 4}

    rec.load_pbp, rec.load_schedules, rec.build_split = load_pbp, load_schedules, build_split
    rec.validate_split = lambda split: {"ok": True}
    rec.fourth_down_index = lambda split: np.flatnonzero(
        (split.states[:, S.STATE_INDEX["phase"]] == S.PHASE["PLAY"])
        & (split.states[:, S.STATE_INDEX["down"]] == 4))

    model = types.ModuleType("model")
    model.WorldModel = _StubModel

    train = types.ModuleType("train")

    def train_model(m, tr, va, cfg, ckpt_path, deadline_ts, log):
        calls["train"] = calls.get("train", 0) + 1
        import torch

        resumed = R.Path(ckpt_path).exists()
        torch.save({"epoch": 1, "best_val": 1.1}, ckpt_path)
        if fail_train_once and calls["train"] == 1:
            raise RuntimeError("disconnect")
        log("epoch 0: train 1.0 val 1.1")
        return {"resumed": resumed, "history": [{"epoch": 0, "train_loss": 1.0,
                                                 "val_total": 1.1}],
                "train_records_per_s": 1e6}

    train.train_model = train_model
    train.load_checkpoint = lambda path, cfg_override=None, device=None: (_StubModel(), {})
    train.evaluate_nll = lambda m, split, cfg=None: {
        "total": 9.0, "fields": {n: 0.9 for n, _ in S.TARGET_FIELDS}}

    sim = types.ModuleType("simulate")

    def wp_by_simulation(m, split, idx, n, cfg, deadline_ts, rng):
        calls["wp_calls"] = calls.get("wp_calls", 0) + 1
        if interrupt_wp_call is not None and calls["wp_calls"] == interrupt_wp_call:
            raise KeyboardInterrupt
        calls["wp"] = calls.get("wp", 0) + len(idx)

        g = np.searchsorted(split.offsets, idx, side="right") - 1
        return pd.DataFrame({"idx": idx, "home_wp_sim": rng.uniform(0.1, 0.9, len(idx)),
                             "home_result": S.outcome(split.final_diff[g]),
                             "done": np.ones(len(idx), dtype=bool),
                             "mean_steps": np.full(len(idx), 10.0)})

    def plan_fourth_downs(m, split, idx, n, cfg, deadline_ts, rng):
        calls["plan"] = calls.get("plan", 0) + len(idx)
        k = len(idx)
        return pd.DataFrame({"idx": idx, "wp_go": np.full(k, 0.6), "wp_fg": np.full(k, 0.5),
                             "wp_punt": np.full(k, 0.55), "se_go": np.full(k, 0.01),
                             "se_fg": np.full(k, 0.01), "se_punt": np.full(k, 0.01),
                             "best": ["GO"] * k, "margin": np.full(k, 0.05),
                             "too_close": np.zeros(k, dtype=bool),
                             "real_action": ["PUNT"] * k, "real_choice": ["PUNT"] * k,
                             "done": np.ones(k, dtype=bool)})

    sim.wp_by_simulation, sim.plan_fourth_downs = wp_by_simulation, plan_fourth_downs
    sim.simulate_games = lambda m, split, n, cfg, d, rng: {"n_games": n, "sim": {"x": 1.0},
                                                             "real": {"x": 1.1}}
    return {"records": rec, "model": model, "train": train, "simulate": sim}


@pytest.fixture
def stubbed(monkeypatch):
    calls: dict = {}
    for name, mod in _stubs(calls).items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _tiny_preset(**kw):
    p = R.get_preset("smoke")
    p["seasons"] = {"train": [2019], "val": [2020], "test": [2021]}
    p.update(kw)
    return p


# ---------------------------------------------------------------- tests
def test_pick_preset():
    assert R.pick_preset("Tesla T4", 15.0) == "t4"
    assert R.pick_preset("NVIDIA L4", 22.5) == "l4"
    assert R.pick_preset("NVIDIA A100-SXM4-40GB", 40.0) == "a100"
    assert R.pick_preset("NVIDIA H100 80GB HBM3", 80.0) == "a100"
    assert R.pick_preset("Some New GPU", 48.0) == "a100"
    assert R.pick_preset("Some Old GPU", 8.0) == "t4"


def test_pick_preset_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert R.pick_preset() == "smoke"


def test_presets_use_full_seasons_and_fit_budget():
    for name in ("t4", "l4", "a100"):
        p = R.PRESETS[name]
        assert p["seasons"]["train"] == list(range(2010, 2022))
        assert p["seasons"]["val"] == [2022, 2023]
        assert p["seasons"]["test"] == [2024, 2025]
        assert p["budget_min"] <= 55
        assert p["reserve_package_s"] > 0
    for p in R.PRESETS.values():
        assert set(p["stage_min"]) == set(R.STAGES)
        for s in p["seasons"].values():
            assert max(s) <= R.MAX_SEASON


def test_2026_guard(tmp_path, stubbed):
    with pytest.raises(AssertionError):
        R.check_seasons([2025, 2026])
    p = _tiny_preset()
    p["seasons"]["test"] = [2026]
    run = R.Run(p, tmp_path, quiet=True)
    out = R.run_stage(run, "data")
    assert out["status"] == "error" and "2025" in out["error"]
    assert all(max(s) <= 2025 for s in stubbed.get("load_pbp", []))


def test_split_roundtrip(tmp_path):
    sp = R.synthetic_split(3, seed=2)
    R.save_split(tmp_path / "a.npz", sp, w=np.arange(R.n_records(sp), dtype=float))
    back, extra = R.load_split(tmp_path / "a.npz")
    assert back.game_ids == sp.game_ids
    assert np.array_equal(back.states, sp.states)
    assert len(extra["w"]) == R.n_records(sp)
    both = R.concat_splits([sp, back])
    assert both.offsets[-1] == 2 * R.n_records(sp)
    sub = R.subset_split(both, [4])
    assert np.array_equal(sub.states, sp.states[sp.offsets[1]:sp.offsets[2]])


def test_frequency_baseline_beats_uniform():
    sp = R.synthetic_split(6, seed=3)
    tables = R.fit_frequency_baseline(sp)
    nll = R.baseline_nll(tables, sp)
    uniform = sum(np.log(size) for _, size in S.TARGET_FIELDS)
    assert np.isfinite(nll["total_per_transition"])
    assert nll["total_per_transition"] < uniform


def test_hypothetical_edits_codes():
    sp = R.synthetic_split(4, seed=4)
    hyp, doc = R.build_hypothetical(sp)
    j = R.n_records(hyp) - 1
    st = hyp.states[j]
    assert st[S.STATE_INDEX["down"]] == 4 and st[S.STATE_INDEX["ydstogo"]] == 2
    assert st[S.STATE_INDEX["yardline"]] == 60
    assert hyp.raw[j, 0] == 240 and st[S.STATE_INDEX["clock"]] == S.clock_code(240)
    sign = 1 if st[S.STATE_INDEX["poss"]] == S.POSS_CODE["HOME"] else -1
    assert hyp.raw[j, 1] == 4 * sign
    assert st[S.STATE_INDEX["score"]] == S.score_code(4 * sign)
    assert doc["base_play_id"] in sp.play_ids


def test_bootstrap_interval_contains_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(size=200)
    b = R.game_bootstrap(v, np.repeat(np.arange(20), 10), 300, rng)
    assert b["lo95"] <= b["mean"] <= b["hi95"]


def test_full_run_resume_and_test_once(tmp_path, stubbed):
    run = R.Run(_tiny_preset(), tmp_path, quiet=True)
    out = R.run_all(run)
    for s in ("data", "train", "eval", "test", "plan", "package"):
        assert out[s]["status"] == "done", (s, out[s])
    res = R.read_json(tmp_path / "results.json")
    assert res["stages"]["test"]["status"] == "done"
    assert res["prereg"]["config_hash"] == run.full_config_hash()
    assert (tmp_path / "world_model_results.zip").exists()
    assert "Preset smoke" in R.summary(run)
    # Re-running skips every finished stage.
    before = dict(stubbed)
    run2 = R.Run(_tiny_preset(), tmp_path, quiet=True)
    R.run_all(run2)
    assert stubbed["train"] == before["train"] and stubbed["wp"] == before["wp"]
    assert stubbed["build_split"] == before["build_split"]
    # Changing the test plan after scoring is refused without the flag ...
    p = _tiny_preset()
    p["test"]["wp_states"] = 10
    out3 = R.run_stage(R.Run(p, tmp_path, quiet=True), "test")
    assert out3["status"] == "error" and "refusing" in out3["error"]
    # ... and allowed, and recorded, with it.
    out4 = R.run_stage(R.Run(p, tmp_path, quiet=True, allow_rescore=True), "test")
    assert out4["status"] == "done"
    prereg = R.read_json(tmp_path / "prereg.json")
    assert prereg["attempts"][-1]["rescore"] is True


def test_prereg_written_before_test_data_is_read(tmp_path, stubbed, monkeypatch):
    run = R.Run(_tiny_preset(), tmp_path, quiet=True)
    for s in ("data", "train"):
        R.run_stage(run, s)
    seen = []
    orig = R.Run.split

    def spy(self, name):
        if name == "test":
            seen.append((tmp_path / "prereg.json").exists())
        return orig(self, name)

    monkeypatch.setattr(R.Run, "split", spy)
    R.run_stage(run, "test")
    assert seen and all(seen)


def test_training_resumes_after_disconnect(tmp_path, monkeypatch):
    calls: dict = {}
    for name, mod in _stubs(calls, fail_train_once=True).items():
        monkeypatch.setitem(sys.modules, name, mod)
    run = R.Run(_tiny_preset(), tmp_path, quiet=True)
    R.run_stage(run, "data")
    assert R.run_stage(run, "train")["status"] == "error"
    assert not run.is_done("train")
    out = R.run_stage(R.Run(_tiny_preset(), tmp_path, quiet=True), "train")
    assert out["status"] == "done" and out["history"]["resumed"] is True


def test_deadline_guard_still_writes_results(tmp_path, stubbed):
    p = _tiny_preset(budget_min=0.01, reserve_package_s=1)
    run = R.Run(p, tmp_path, quiet=True)
    time.sleep(0.05)
    out = R.run_all(run)
    assert out["data"]["status"] == "skipped" and "time budget" in out["data"]["reason"]
    assert out["train"]["status"] == "skipped"
    res = R.read_json(tmp_path / "results.json")
    assert res["stages"]["eval"]["status"] == "skipped"
    assert out["package"]["status"] == "done"


def test_ledger_shrinks_budget_on_resume(tmp_path):
    p = _tiny_preset()
    R.write_json(tmp_path / "ledger.json", {"consumed_s": 600.0})
    run = R.Run(p, tmp_path, quiet=True)
    assert run.remaining() < 60 * p["budget_min"] - 590


def test_resume_survives_source_change(tmp_path, monkeypatch):
    calls: dict = {}
    for name, mod in _stubs(calls).items():
        monkeypatch.setitem(sys.modules, name, mod)
    run = R.Run(_tiny_preset(), tmp_path, quiet=True)
    R.run_stage(run, "data")
    R.run_stage(run, "train")
    ckpt = run.ckpt_path()
    # Edited model/train/records code: the train stage stays done, same checkpoint.
    orig = R._source_hash
    monkeypatch.setattr(R, "_source_hash", lambda mods: "edited" + orig(mods))
    run2 = R.Run(_tiny_preset(), tmp_path, quiet=True)
    assert run2.is_done("train") and run2.ckpt_path() == ckpt
    assert not run2.is_done("data")  # records changed: rebuild in memory and compare
    out = R.run_stage(run2, "data")
    assert set(out["seasons_rebuilt"].values()) == {"identical, kept"}
    assert run2.is_done("train") and run2.ckpt_path() == ckpt
    # Code that really changes the arrays replaces them, and training starts again.
    for name, mod in _stubs(calls, build_seed=99).items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(R, "_source_hash", lambda mods: "edited again" + orig(mods))
    run3 = R.Run(_tiny_preset(), tmp_path, quiet=True)
    out = R.run_stage(run3, "data")
    assert "replaced" in out["seasons_rebuilt"].values()
    assert not run3.is_done("train") and run3.ckpt_path() != ckpt


def test_stop_after_train(tmp_path, stubbed):
    run = R.Run(_tiny_preset(), tmp_path, quiet=True)
    out = R.run_all(run, stop_after="train")
    assert "eval" not in out and out["package"]["status"] == "done"
    res = R.read_json(tmp_path / "results.json")
    assert res["stages"]["train"]["status"] == "done"
    assert res["stages"]["eval"]["status"] == "missing"
    assert "train    done" in R.status_table(run)
    with pytest.raises(ValueError):
        R._beyond("eval", "nonsense")


def test_interrupted_eval_resumes_from_saved_chunks(tmp_path, monkeypatch, capsys):
    calls: dict = {}
    for name, mod in _stubs(calls, interrupt_wp_call=2).items():
        monkeypatch.setitem(sys.modules, name, mod)
    p = _tiny_preset()
    p["eval"].update(wp_states=6, chunk_states=2, sim_games=0)
    run = R.Run(p, tmp_path, quiet=True)
    out = R.run_all(run, stop_after="eval")
    assert out["stopped_at"] == "eval"
    assert "Stopped safely at eval" in capsys.readouterr().out
    res = R.read_json(tmp_path / "results.json")
    assert res["stages"]["eval"]["status"] == "interrupted"
    assert calls["wp"] == 2  # one chunk finished and was saved
    calls2: dict = {}
    for name, mod in _stubs(calls2).items():
        monkeypatch.setitem(sys.modules, name, mod)
    out = R.run_stage(R.Run(p, tmp_path, quiet=True), "eval")
    assert out["status"] == "done"
    assert calls2["wp"] == 4  # only the two missing chunks were simulated
    assert out["wp"]["progress"]["reused_states"] == 2 and out["wp"]["n_states"] == 6


def test_keyboard_interrupt_mid_training_resumes_real_modules(tmp_path, monkeypatch, capsys):
    """Real model.py and train.py: stop after epoch 0, then resume at epoch 1."""
    pytest.importorskip("torch")
    import model as real_model
    import train as real_train

    calls: dict = {}
    stubs = _stubs(calls)
    monkeypatch.setitem(sys.modules, "records", stubs["records"])
    monkeypatch.setitem(sys.modules, "model", real_model)
    monkeypatch.setitem(sys.modules, "train", real_train)
    p = _tiny_preset()
    p["model"] = {"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32, "dec_hidden": 16,
                  "max_len": 256, "dropout": 0.0}
    p["train"].update(epochs=2, patience=5, batch_games=4, warmup_steps=2)
    orig_log = R.Run.log

    def log_then_stop(self, *msg):
        orig_log(self, *msg)
        if any("epoch 0:" in str(m) for m in msg):
            raise KeyboardInterrupt

    monkeypatch.setattr(R.Run, "log", log_then_stop)
    run = R.Run(p, tmp_path, quiet=True)
    out = R.run_all(run, stop_after="train")
    assert out["stopped_at"] == "train"
    assert "Stopped safely at train" in capsys.readouterr().out
    assert R.read_json(tmp_path / "results.json")["stages"]["train"]["status"] == "interrupted"
    assert run.ckpt_path().exists() and not run.is_done("train")
    assert R.checkpoint_info(run)["epochs_done"] == 1
    monkeypatch.setattr(R.Run, "log", orig_log)
    run2 = R.Run(p, tmp_path, quiet=True)
    out = R.run_stage(run2, "train")
    assert out["status"] == "done"
    hist = out["history"]
    assert hist["epochs_done"] == 2
    assert [h["epoch"] for h in hist["history"]] == [0, 1]
    assert R.checkpoint_info(run2)["epochs_done"] == 2


def test_mismatched_checkpoint_starts_fresh(tmp_path):
    torch = pytest.importorskip("torch")
    import model as real_model

    m = real_model.WorldModel({"d_model": 16, "n_layers": 1, "n_heads": 2, "d_ff": 32,
                               "dec_hidden": 16})
    path = tmp_path / "ck.pt"
    torch.save({"current_model": {"not_a_param": torch.zeros(3)}}, path)
    assert "missing" in R._check_checkpoint(path, m)
    torch.save({"current_model": m.state_dict()}, path)
    assert R._check_checkpoint(path, m) is None
