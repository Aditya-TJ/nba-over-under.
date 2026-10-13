"""Download raw data: betting odds (SQLite) and team box scores (NBA stats API).

Usage:  uv run python -m src.fetch_data
"""
import time

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamefinder

from src.config import ODDS_DB, ODDS_DB_URL, RAW_DIR, SEASONS, WARMUP_SEASON


def fetch_odds():
    if ODDS_DB.exists():
        print(f"odds: {ODDS_DB.name} already present")
        return
    print("odds: downloading", ODDS_DB_URL)
    r = requests.get(ODDS_DB_URL, timeout=120)
    r.raise_for_status()
    ODDS_DB.write_bytes(r.content)


def fetch_games(season, retries=3):
    out = RAW_DIR / f"games_{season}.csv"
    if out.exists():
        print(f"games {season}: already present")
        return
    for attempt in range(retries):
        try:
            df = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                season_type_nullable="Regular Season",
                timeout=60,
            ).get_data_frames()[0]
            break
        except Exception as e:  # the stats API is flaky; back off and retry
            print(f"games {season}: attempt {attempt + 1} failed ({e})")
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"could not download {season}")
    df.to_csv(out, index=False)
    print(f"games {season}: {len(df)} team-game rows")
    time.sleep(1.5)  # be polite to stats.nba.com


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch_odds()
    for season in [WARMUP_SEASON] + SEASONS:
        fetch_games(season)


if __name__ == "__main__":
    main()
