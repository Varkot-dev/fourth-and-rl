"""A fingerprint of a play table, for cache keys.

Saved fits (bootstrap replicates, the development component bundle, the game-model tables)
used to be keyed by config and seasons only, so a rebuilt play table with the same seasons
would silently reuse fits made from the old one. `data_fingerprint` summarizes the input
data itself: rows and games per season, plus a hash of a few key columns. Two tables with
the same fingerprint hold the same plays in those columns, whatever their row order or
extra columns, so a table derived from the play table without dropping rows (for example
`bootstrap.prepare_plays`, which adds columns and sorts) has the fingerprint of its source.

The hash is computed on the key columns cast to text, so it is stable for one polars
version; an upgrade that changes how numbers print would change it and cause one refit,
which errs on the safe side.
"""

from __future__ import annotations

import hashlib
from typing import Any

import polars as pl

# The columns whose values the hash covers. A column missing from the frame is listed in
# the fingerprint and left out of the hash (toy frames in tests lack some of them).
KEY_COLUMNS: tuple[str, ...] = (
    "game_id", "play_id", "down", "ydstogo", "yardline_100", "play_type", "yards_gained",
    "posteam_won",
)  # fmt: skip

_NULL = "\x00"  # stands for a missing value in the hashed text


def data_fingerprint(df: pl.DataFrame) -> dict[str, Any]:
    """Rows and games per season, and a hash of the key columns, of a play table.

    Returns a JSON-ready dict (season keys are strings), so it can be written into a key
    file and compared after reading it back:

      by_season        {"2010": {"rows": ..., "games": ...}, ...} (needs `season`, `game_id`)
      n_rows           rows in the frame
      columns_hashed   the key columns present in the frame, in hashing order
      columns_missing  the key columns absent from it
      hash             SHA-1 (hex, first 16 characters) of the key columns, independent of
                       row order: the rows are sorted on every hashed column first
    """
    present = [c for c in KEY_COLUMNS if c in df.columns]
    out: dict[str, Any] = {"n_rows": int(df.height)}
    if {"season", "game_id"} <= set(df.columns):
        per = (
            df.group_by("season")
            .agg(pl.len().alias("rows"), pl.col("game_id").n_unique().alias("games"))
            .sort("season")
        )
        out["by_season"] = {
            str(int(s)): {"rows": int(r), "games": int(g)}
            for s, r, g in per.select("season", "rows", "games").iter_rows()
        }
    else:
        out["by_season"] = {}
    out["columns_hashed"] = present
    out["columns_missing"] = [c for c in KEY_COLUMNS if c not in df.columns]
    digest = hashlib.sha1(usedforsecurity=False)
    if present and df.height:
        text = (
            df.select([pl.col(c).cast(pl.String).fill_null(_NULL).alias(c) for c in present])
            .sort(present)
            .select(pl.concat_str(present, separator="\x1f").alias("row"))["row"]
        )
        digest.update("\x1e".join(text.to_list()).encode())
    out["hash"] = digest.hexdigest()[:16]
    return out
