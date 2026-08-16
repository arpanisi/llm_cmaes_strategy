from pathlib import Path

from .settings import BOOKS, FRED_SERIES

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MARKET_DIR = DATA_DIR / "market"       # per-ticker OHLCV: market/<book>/<TICKER>.csv
ECONOMIC_DIR = DATA_DIR / "economic"   # macro series: economic/<FRED_SERIES>.csv
OUTPUTS_DIR = PROJECT_ROOT / "outputs" # run outputs: summaries, best strategies, live reports
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_YAML = CONFIG_DIR / "models.yaml"

RUNBOOK_DIR = PROJECT_ROOT / "runbook.md"


def ensure_dirs() -> None:
    for d in (
        MARKET_DIR,
        *(MARKET_DIR / book for book in BOOKS),
        ECONOMIC_DIR,
        OUTPUTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def ticker_ohlcv_file(book: str, ticker: str) -> Path:
    return MARKET_DIR / book / f"{ticker}.csv"


def ticker_ohlcv_meta_file(book: str, ticker: str) -> Path:
    return MARKET_DIR / book / f"{ticker}.meta.json"


def macro_series_file() -> Path:
    return ECONOMIC_DIR / f"{FRED_SERIES}.csv"


def macro_series_meta_file() -> Path:
    return ECONOMIC_DIR / f"{FRED_SERIES}.meta.json"
