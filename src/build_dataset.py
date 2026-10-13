"""Clean raw data, engineer pre-game features and merge with the betting lines.

Every feature for a game is computed only from games played *before* it
(rolling windows are shifted by one game), so nothing leaks from the future.

Usage:  uv run python -m src.build_dataset
"""
import sqlite3

import numpy as np
import pandas as pd

from src.config import ODDS_DB, ODDS_TABLES, PROCESSED_DIR, RAW_DIR, SEASONS, WARMUP_SEASON

WINDOWS = [5, 10, 20]

# Per-game team stats that get rolled into features. "opp_" versions describe
# what the team allowed, i.e. its defence.
ROLL_STATS = [
    "pace", "ortg", "drtg", "game_total",
    "efg", "tov_pct", "orb_pct", "ftr", "fg3a_rate",
    "opp_efg", "opp_tov_pct", "opp_orb_pct", "opp_ftr",
]

ODDS_NAME_FIX = {"Los Angeles Clippers": "LA Clippers", "Charlotte Bobcats": "Charlotte Hornets"}


def load_team_games():
    frames = []
    for season in [WARMUP_SEASON] + SEASONS:
        df = pd.read_csv(RAW_DIR / f"games_{season}.csv", dtype={"GAME_ID": str})
        df["season"] = season
        frames.append(df)
    tg = pd.concat(frames, ignore_index=True)
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"])
    # "BOS vs. NYK" -> BOS at home, "NYK @ BOS" -> BOS at home. Neutral-site games
    # (NBA Cup knockouts, international games) can list "@" for both teams, so the
    # home team is taken from the first row of each game to keep exactly one.
    parts = tg["MATCHUP"].str.split(r" vs\. | @ ", regex=True)
    tg["home_abbr"] = np.where(tg["MATCHUP"].str.contains(" vs. "), parts.str[0], parts.str[1])
    game_home = tg.groupby("GAME_ID")["home_abbr"].transform("first")
    tg["is_home"] = tg["TEAM_ABBREVIATION"] == game_home

    # Attach the opponent's box score to every team-game row.
    keep = ["GAME_ID", "TEAM_ID", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "OREB", "DREB", "TOV"]
    opp = tg[keep].rename(columns={c: f"OPP_{c}" for c in keep if c != "GAME_ID"})
    tg = tg.merge(opp, on="GAME_ID")
    tg = tg[tg["TEAM_ID"] != tg["OPP_TEAM_ID"]]
    bad = tg.groupby("GAME_ID")["TEAM_ID"].transform("size") != 2
    if bad.any():
        print(f"dropping {bad.sum() // 2} games without exactly two teams")
        tg = tg[~bad]
    return tg.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)


def add_box_score_stats(tg):
    """Possessions, pace, ratings and Dean Oliver's Four Factors for each team-game."""
    poss = tg.FGA + 0.44 * tg.FTA - tg.OREB + tg.TOV
    opp_poss = tg.OPP_FGA + 0.44 * tg.OPP_FTA - tg.OPP_OREB + tg.OPP_TOV
    tg["poss"] = (poss + opp_poss) / 2
    # MIN is total team minutes (240 in regulation); pace is per 48 minutes.
    tg["pace"] = 48 * tg["poss"] / (tg["MIN"] / 5)
    tg["ortg"] = 100 * tg.PTS / tg["poss"]
    tg["drtg"] = 100 * tg.OPP_PTS / tg["poss"]
    tg["game_total"] = tg.PTS + tg.OPP_PTS

    tg["efg"] = (tg.FGM + 0.5 * tg.FG3M) / tg.FGA
    tg["tov_pct"] = tg.TOV / poss
    tg["orb_pct"] = tg.OREB / (tg.OREB + tg.OPP_DREB)
    tg["ftr"] = tg.FTM / tg.FGA
    tg["fg3a_rate"] = tg.FG3A / tg.FGA

    tg["opp_efg"] = (tg.OPP_FGM + 0.5 * tg.OPP_FG3M) / tg.OPP_FGA
    tg["opp_tov_pct"] = tg.OPP_TOV / opp_poss
    tg["opp_orb_pct"] = tg.OPP_OREB / (tg.OPP_OREB + tg.DREB)
    tg["opp_ftr"] = tg.OPP_FTM / tg.OPP_FGA
    return tg


def add_rolling_and_rest(tg):
    """Pre-game rolling averages and schedule/fatigue features, per team."""
    g = tg.groupby("TEAM_ID", group_keys=False)
    for w in WINDOWS:
        rolled = g[ROLL_STATS].transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
        tg = tg.join(rolled.add_suffix(f"_r{w}"))
        g = tg.groupby("TEAM_ID", group_keys=False)

    prev_date = g["GAME_DATE"].shift(1)
    tg["days_rest"] = (tg["GAME_DATE"] - prev_date).dt.days.clip(upper=7).fillna(7)
    tg["b2b"] = (tg["days_rest"] == 1).astype(int)
    # Third game in four nights: the game two back was at most 3 days ago.
    two_back = g["GAME_DATE"].shift(2)
    tg["three_in_four"] = ((tg["GAME_DATE"] - two_back).dt.days <= 3).astype(int)
    tg["games_played"] = tg.groupby(["TEAM_ID", "season"]).cumcount()
    return tg


def league_context(games):
    """Average game total across the league over the previous 21 days (scoring trend)."""
    daily = games.groupby("date")["total"].agg(["sum", "count"]).asfreq("D", fill_value=0)
    roll = daily.shift(1).rolling(21, min_periods=1).sum()
    return (roll["sum"] / roll["count"]).rename("league_total_21d")


def to_game_level(tg):
    """One row per game: home-team features, away-team features, outcome."""
    feat_cols = [c for c in tg.columns if any(c.startswith(f"{s}_r") for s in ROLL_STATS)]
    feat_cols += ["days_rest", "b2b", "three_in_four", "games_played"]
    base = ["GAME_ID", "GAME_DATE", "season", "TEAM_NAME", "PTS"]

    home = tg[tg.is_home][base + feat_cols].add_prefix("h_")
    away = tg[~tg.is_home][base + feat_cols].add_prefix("a_")
    games = home.merge(away, left_on="h_GAME_ID", right_on="a_GAME_ID")
    games = games.rename(columns={
        "h_GAME_ID": "game_id", "h_GAME_DATE": "date", "h_season": "season",
        "h_TEAM_NAME": "home", "a_TEAM_NAME": "away",
        "h_PTS": "home_pts", "a_PTS": "away_pts",
    }).drop(columns=["a_GAME_ID", "a_GAME_DATE", "a_season"])
    games["total"] = games.home_pts + games.away_pts
    games = games.join(league_context(games), on="date")
    return games


def add_matchup_features(games):
    """Domain-informed combinations: expected pace, efficiency and total for the matchup."""
    for w in WINDOWS:
        exp_pace = (games[f"h_pace_r{w}"] + games[f"a_pace_r{w}"]) / 2
        # A team's expected efficiency = average of its offence and the opponent's defence.
        h_eff = (games[f"h_ortg_r{w}"] + games[f"a_drtg_r{w}"]) / 2
        a_eff = (games[f"a_ortg_r{w}"] + games[f"h_drtg_r{w}"]) / 2
        games[f"exp_pace_r{w}"] = exp_pace
        games[f"exp_total_r{w}"] = exp_pace * (h_eff + a_eff) / 100
    games["rest_diff"] = games.h_days_rest - games.a_days_rest
    return games


def load_odds():
    con = sqlite3.connect(ODDS_DB)
    frames = []
    for season, table in ODDS_TABLES.items():
        df = pd.read_sql(f'SELECT Date, Home, Away, OU, Spread FROM "{table}"', con)
        df["odds_season"] = season
        frames.append(df)
    odds = pd.concat(frames, ignore_index=True)
    odds["date"] = pd.to_datetime(odds["Date"])
    for col in ["Home", "Away"]:
        odds[col] = odds[col].replace(ODDS_NAME_FIX)
    odds = odds.rename(columns={"Home": "home", "Away": "away", "OU": "line", "Spread": "spread"})
    odds = odds.drop_duplicates(["date", "home", "away"])
    # "PK" (pick'em) means a spread of zero. Positive spread = home team favoured.
    odds["spread"] = pd.to_numeric(odds["spread"].replace("PK", 0))
    odds["abs_spread"] = odds["spread"].abs()
    return odds[["date", "home", "away", "line", "spread", "abs_spread"]]


def main():
    tg = load_team_games()
    tg = add_box_score_stats(tg)
    tg = add_rolling_and_rest(tg)
    games = to_game_level(tg)
    games = add_matchup_features(games)
    games = games[games.season != WARMUP_SEASON]

    odds = load_odds()
    merged = games.merge(odds, on=["date", "home", "away"], how="left")
    print(f"games: {len(games)}, matched to a line: {merged.line.notna().sum()}")
    print(merged.groupby("season").line.apply(lambda s: f"{s.notna().sum()}/{len(s)}").to_string())

    merged = merged.dropna(subset=["line"])
    # Sanity filter: lines outside a plausible range are data-entry errors.
    implausible = ~merged.line.between(170, 270)
    if implausible.any():
        print(f"dropping {implausible.sum()} games with implausible lines")
        merged = merged[~implausible]
    # Rolling windows need 20 prior games; only the very first games of 2017-18 lack them
    # (2016-17 supplies history for the rest).
    before = len(merged)
    merged = merged.dropna().copy()
    print(f"dropped {before - len(merged)} games missing rolling history")

    merged["resid"] = merged.total - merged.line          # how far the total landed from the line
    merged["over"] = (merged.total > merged.line).astype(int)
    merged["push"] = (merged.total == merged.line).astype(int)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "dataset.csv"
    merged.sort_values("date").to_csv(out, index=False)
    print(f"saved {len(merged)} games x {merged.shape[1]} columns -> {out.relative_to(PROCESSED_DIR.parents[1])}")


if __name__ == "__main__":
    main()
