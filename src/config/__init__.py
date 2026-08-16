from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import paths
from .settings import *  # noqa: F401,F403
from .settings import BOOKS, DATE_END, DATE_START, EQUITY_TICKERS, CRYPTO_TICKERS


def load_models(path: Path | None = None) -> dict:
    p = path or paths.MODELS_YAML
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_dotenv_silently() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(paths.PROJECT_ROOT / ".env")
    except Exception:
        pass


@dataclass(frozen=True)
class RunConfig:
    book: str
    date_start: str = DATE_START
    date_end: str = DATE_END
    train_bars: int = TRAIN_BARS
    test_bars: int = TEST_BARS
    step_bars: int = STEP_BARS
    cma_restarts: int = CMA_RESTARTS
    cma_evals: int = CMA_EVALS_PER_RESTART
    seed: int = 42
    rounds: int = 5
    model: str | None = None
    offline: bool = False
    tickers_override: tuple[str, ...] | None = None
    extra: dict = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        if self.tickers_override is not None:
            return list(self.tickers_override)
        return EQUITY_TICKERS if self.book == "equity" else CRYPTO_TICKERS

    @property
    def bars_per_year(self) -> int:
        return BARS_PER_YEAR[self.book]

    def as_record(self) -> dict:
        return {
            "book": self.book,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "train_bars": self.train_bars,
            "test_bars": self.test_bars,
            "step_bars": self.step_bars,
            "cma_restarts": self.cma_restarts,
            "cma_evals": self.cma_evals,
            "seed": self.seed,
            "rounds": self.rounds,
            "model": self.model,
            "offline": self.offline,
        }
