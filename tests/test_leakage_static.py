"""Repo-wide leakage check: every raw read of the play table is on an allowlist.

`data.load_plays()` returns every season, test seasons included, and `pl.read_parquet` can
open any file. Fitting code must read plays through `splits.load_dev_plays()` and score
test seasons only through `splits.load_test_plays(purpose=...)`. This test scans `src/`
and `scripts/` for the two raw calls and fails on any use that is not listed below with a
reason. Descriptive statistics over all seasons are legitimate; fits are not.

To add a use: put it in ALLOWLIST with the number of calls in that file and a reason that
says why no fit reported on the test seasons depends on it. Entries marked "pending" have
no fixed count while another piece of work rewrites those files; tighten them afterwards.
"""

from __future__ import annotations

import re

import pytest

from fourth_and_rl.paths import ROOT

PATTERNS = ("load_plays(", "read_parquet(")

# (file relative to the repo root, pattern) -> (calls expected, reason). None = pending.
ALLOWLIST: dict[tuple[str, str], tuple[int | None, str]] = {
    # ---- the loaders themselves
    ("src/fourth_and_rl/data.py", "load_plays("): (
        1, "the definition of load_plays"),
    ("src/fourth_and_rl/data.py", "read_parquet("): (
        2, "builds the state table from the raw season files, and load_plays itself"),
    ("src/fourth_and_rl/splits.py", "read_parquet("): (
        3, "the guarded loaders: development, test (purpose logged), holdout (acknowledged)"),
    ("src/fourth_and_rl/run_pass.py", "read_parquet("): (
        1, "extra columns (participation) joined by game and play id; no season selection, "
           "the caller's frame decides which seasons are used"),
    # ---- data building and descriptive research over all seasons
    ("scripts/build_data.py", "load_plays("): (
        1, "writes the development and test files (splits.write_split_files) and the M0 "
           "data summary; nothing is fitted"),
    ("src/fourth_and_rl/team_matchups.py", "load_plays("): (
        1, "the Colab rebuild, as scripts/build_data.py: checks that no holdout season is "
           "present and writes the development and test files; every fit then reads "
           "splits.load_dev_plays (and the logged kickoff-table exception)"),
    ("scripts/research/endgame_clock_stats.py", "load_plays("): (
        2, "one call and one source string; the --split all descriptive research record; "
           "the model's constants come from the --dev-only file and the simulator's "
           "targets from the --split test file"),
    ("scripts/research/team_aggressiveness_stats.py", "load_plays("): (
        1, "descriptive research on team aggressiveness (docs/research); feeds no model"),
    ("scripts/research/leakage_checks.py", "load_plays("): (
        1, "the leakage audit itself: refits on development seasons and contrasts with "
           "fits that include test seasons on purpose, to show the test can see a leak"),
    ("scripts/research/leakage_checks.py", "read_parquet("): (
        1, "the audit reads the saved cross-fitted predictions to compare with a refit"),
    ("scripts/blown_leads.py", "load_plays("): (
        1, "M3b is a calibration test of win-probability paths over all seasons; it reads "
           "out-of-fold predictions and fits nothing that is scored on test seasons "
           "(audit m4 notes the reference simulator's in-sample shape)"),
    ("scripts/blown_leads.py", "read_parquet("): (
        1, "the saved out-of-fold win-probability predictions"),
    ("scripts/questions/q2_run_vs_pass.py", "load_plays("): (
        1, "nflfastR EPA comparisons over all seasons (descriptive); its one fitted piece "
           "(run_pass outcome tables) is fitted on development seasons behind "
           "assert_no_test_seasons, and test seasons are read via load_test_plays"),
    # ---- M6 offline RL
    ("scripts/offline_real_data.py", "read_parquet("): (
        1, "reads back its own cache of 2024 to 2025 decision outcomes, which the same "
           "script built from splits.load_test_plays(purpose=TEST_PURPOSE) (logged)"),
    # ---- M7 coach evaluation (reads of its own caches only)
    ("scripts/m7_coach_evaluation.py", "read_parquet("): (
        3, "the M3 per-play categories of 2024-2025 (written by bootstrap_decisions.py from "
           "load_test_plays), its own graded-decisions cache (built from the guarded loaders "
           "with fold models that never saw each season), and the cached nflverse schedule "
           "(head coaches per game; no plays)"),
    ("scripts/m7_penalty_ledger.py", "read_parquet("): (
        2, "its own caches of the nflverse penalty columns (penalty type, team and yards, "
           "joined by game and play id to plays read through the guarded loaders) and of the "
           "schedule (head coaches per game); nothing is fitted on them"),
    # ---- pending: being fixed by the win-probability rewrite
    ("src/fourth_and_rl/wp.py", "load_plays("): (
        None, "pending: being fixed by the win-probability rewrite"),
    ("src/fourth_and_rl/wp.py", "read_parquet("): (
        None, "pending: being fixed by the win-probability rewrite"),
}  # fmt: skip

# Files of the win-probability rewrite, matched by name: pending, any count. (The neural
# model's script, scripts/fit_wp_nn.py, is converted and is checked below like the rest.)
PENDING = (re.compile(r"^scripts/fit_wp\.py$"), re.compile(r"^scripts/wp_v2[^/]*\.py$"))
PENDING_REASON = "pending: being fixed by the win-probability rewrite"


def uses() -> dict[tuple[str, str], int]:
    found: dict[tuple[str, str], int] = {}
    for folder in ("src", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            text = path.read_text()
            for pattern in PATTERNS:
                n = text.count(pattern)
                if n:
                    found[(rel, pattern)] = n
    return found


def test_every_raw_read_of_the_play_table_is_allowlisted() -> None:
    problems = []
    for (rel, pattern), n in uses().items():
        if any(p.match(rel) for p in PENDING):
            continue
        entry = ALLOWLIST.get((rel, pattern))
        if entry is None:
            problems.append(f"{rel}: {n} x {pattern} not on the allowlist")
        elif entry[0] is not None and entry[0] != n:
            problems.append(f"{rel}: {n} x {pattern}, allowlist says {entry[0]}")
    assert not problems, "\n".join(problems)


def test_allowlist_has_no_stale_entries_and_every_entry_has_a_reason() -> None:
    found = uses()
    for (rel, pattern), (count, reason) in ALLOWLIST.items():
        assert len(reason) > 20, (rel, pattern)
        if count is not None:
            assert (rel, pattern) in found, f"stale allowlist entry: {rel} {pattern}"


@pytest.mark.parametrize(
    "rel",
    [
        "src/fourth_and_rl/components.py",
        "src/fourth_and_rl/mdp.py",
        "src/fourth_and_rl/mdp_drive.py",
        "src/fourth_and_rl/mdp_game.py",
        "src/fourth_and_rl/decision.py",
        "src/fourth_and_rl/bootstrap.py",
        "scripts/fit_components.py",
        "scripts/solve_mdp.py",
        "scripts/validate_simulator.py",
        "scripts/questions/q1_late_lead_dp.py",
        # M2 and M3 scripts: development plays for fits and reference distributions,
        # splits.load_test_plays(purpose=...) for scoring real 2024 to 2025 plays.
        "scripts/decision_chart.py",
        "scripts/compare_nfl4th.py",
        "scripts/bootstrap_decisions.py",
        "scripts/questions/q1_late_lead_one_step.py",
        # The neural win-probability model (version 1 inputs) and its script.
        "src/fourth_and_rl/wp_nn.py",
        "scripts/fit_wp_nn.py",
        # M8 team ratings and question 3 (development file, logged test loads only).
        "src/fourth_and_rl/team_strength.py",
        "scripts/m8_team_strength.py",
        "scripts/questions/q3_team_aware.py",
        # M10 predictability (development file, one logged test load; extra columns via
        # run_pass.load_extra).
        "src/fourth_and_rl/predictability.py",
        "scripts/m10_predictability.py",
    ],
)
def test_fitting_code_never_reads_the_raw_table(rel: str) -> None:
    """The component models, the game model and their scripts use only the guarded loaders."""
    text = (ROOT / rel).read_text()
    for pattern in PATTERNS:
        assert pattern not in text, f"{rel} uses {pattern}"
