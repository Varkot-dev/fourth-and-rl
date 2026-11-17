# Reference output from the nfl4th R package (Ben Baldwin, nflverse).
# These numbers are nfl4th's, not ours. They are used only as a sanity check for
# this project's decision engine (milestone M2).
#
# Usage:  Rscript scripts/r/nfl4th_reference.R [games.rds]
#
# The optional argument is a local copy of
#   https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.rds
# for machines where nflreadr cannot download the schedule itself.
#
# Outputs (under results/reference/):
#   nfl4th_real_fourth_downs.csv   every real fourth down in the chosen seasons
#   nfl4th_scenarios.csv           a grid of hypothetical late-lead situations

suppressMessages({
  library(nfl4th)
  library(dplyr)
  library(tibble)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 1) {
  games <- readRDS(args[1])
  assignInNamespace("load_schedules", function(...) tibble::as_tibble(games), ns = "nflreadr")
}

seasons <- 2024:2025
out_dir <- file.path("results", "reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Every real fourth down in the reference seasons.
real <- nfl4th::load_4th_pbp(seasons) %>%
  filter(down == 4, !is.na(go_boost)) %>%
  transmute(
    game_id, play_id, season, week, posteam, qtr, ydstogo, yardline_100,
    score_differential, game_seconds_remaining,
    nfl4th_first_down_prob = first_down_prob, nfl4th_fg_make_prob = fg_make_prob,
    nfl4th_go_wp = go_wp, nfl4th_fg_wp = fg_wp, nfl4th_punt_wp = punt_wp,
    nfl4th_go_boost = go_boost, went = go
  )
write.csv(real, file.path(out_dir, "nfl4th_real_fourth_downs.csv"), row.names = FALSE)

# 2. Hypothetical late-lead grid (burning question 1): leading in the fourth quarter,
# fourth down in your own half. Teams are a neutral matchup (pick'em spread).
grid <- expand.grid(
  ydstogo = c(1, 2, 3, 5, 10),
  yardline_100 = c(80, 70, 60, 50),
  score_differential = c(1, 3, 4, 7),
  quarter_seconds_remaining = c(120, 240, 360),
  defteam_timeouts_remaining = c(0, 3)
) %>%
  as_tibble() %>%
  mutate(
    home_team = "KC", away_team = "BUF", posteam = "KC", type = "reg", season = 2025,
    qtr = 4, home_opening_kickoff = 0, posteam_timeouts_remaining = 3
  )
scen <- nfl4th::add_4th_probs(grid) %>%
  transmute(
    ydstogo, yardline_100, score_differential, quarter_seconds_remaining,
    defteam_timeouts_remaining,
    nfl4th_first_down_prob = first_down_prob, nfl4th_go_wp = go_wp,
    nfl4th_fg_wp = fg_wp, nfl4th_punt_wp = punt_wp, nfl4th_go_boost = go_boost
  )
write.csv(scen, file.path(out_dir, "nfl4th_scenarios.csv"), row.names = FALSE)

cat("nfl4th version", as.character(packageVersion("nfl4th")), "\n")
cat("real fourth downs:", nrow(real), " scenarios:", nrow(scen), "\n")
