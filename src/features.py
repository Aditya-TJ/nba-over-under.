"""Feature sets and the chronological train / validation / test split."""
import pandas as pd

from src.config import PROCESSED_DIR, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS
from src.build_dataset import WINDOWS

LINE_COLS = ["line", "spread", "abs_spread"]


def load_dataset():
    df = pd.read_csv(PROCESSED_DIR / "dataset.csv", parse_dates=["date"])
    # "Modeling against the line": how far each of our own estimates sits from the book.
    new = {}
    for w in WINDOWS:
        new[f"edge_exp_total_r{w}"] = df[f"exp_total_r{w}"] - df["line"]
        avg_total = (df[f"h_game_total_r{w}"] + df[f"a_game_total_r{w}"]) / 2
        new[f"edge_avg_total_r{w}"] = avg_total - df["line"]
    new["line_vs_league"] = df["line"] - df["league_total_21d"]
    return pd.concat([df, pd.DataFrame(new)], axis=1)


def stat_features(df):
    """Everything known before tip-off that does NOT come from the sportsbook."""
    team = [c for c in df.columns if c.startswith(("h_", "a_"))]
    matchup = [c for c in df.columns if c.startswith(("exp_pace_", "exp_total_"))]
    return team + matchup + ["rest_diff", "league_total_21d"]


def line_features(df):
    """Sportsbook information plus our estimates expressed relative to the line."""
    edges = [c for c in df.columns if c.startswith("edge_")]
    return LINE_COLS + edges + ["line_vs_league"]


def split(df):
    train = df[df.season.isin(TRAIN_SEASONS)]
    val = df[df.season.isin(VAL_SEASONS)]
    test = df[df.season.isin(TEST_SEASONS)]
    return train, val, test
