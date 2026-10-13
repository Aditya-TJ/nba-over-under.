"""Shared paths and season settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

ODDS_DB = RAW_DIR / "OddsData.sqlite"
ODDS_DB_URL = (
    "https://raw.githubusercontent.com/kyleskom/NBA-Machine-Learning-Sports-Betting/"
    "master/Data/OddsData.sqlite"
)

# 2016-17 is only used to warm up the rolling features; it is never trained on.
WARMUP_SEASON = "2016-17"
SEASONS = [
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25", "2025-26",
]

# Which table in OddsData.sqlite holds each season (some seasons exist in several
# versions; these are the most complete ones).
ODDS_TABLES = {
    "2017-18": "odds_2017-18_new",
    "2018-19": "odds_2018-19_new",
    "2019-20": "odds_2019-20_new",
    "2020-21": "odds_2020-21_new",
    "2021-22": "odds_2021-22_new",
    "2022-23": "odds_2022-23_new",
    "2023-24": "2023-24",
    "2024-25": "2024-25",
    "2025-26": "odds_2025-26",
}

TRAIN_SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
VAL_SEASONS = ["2023-24"]
TEST_SEASONS = ["2024-25", "2025-26"]

# Standard -110 pricing: risk 110 to win 100, so break-even is 110/210.
ODDS_DECIMAL = 1 + 100 / 110
BREAK_EVEN = 110 / 210

SEED = 42
