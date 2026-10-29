"""Загрузка цен и расчёт доходностей.

Результат этого модуля — матрица доходностей R размером T x N
(строки — торговые дни, столбцы — активы). Всё остальное в проекте
(ковариация, PCA, риск портфеля) — операции над этой матрицей.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_CSV = DATA_DIR / "prices.csv"

# Сектора подобраны так, чтобы у PCA были шансы найти интерпретируемые факторы
UNIVERSE = {
    "Tech":        ["AAPL", "MSFT", "NVDA", "GOOGL", "ORCL"],
    "Financials":  ["JPM", "BAC", "GS", "MS", "WFC"],
    "Energy":      ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Healthcare":  ["JNJ", "PFE", "MRK", "UNH", "ABBV"],
    "Staples":     ["PG", "KO", "PEP", "WMT", "COST"],
    "Defensive":   ["TLT", "GLD"],  # облигации и золото — для контраста с акциями
}
TICKERS = [t for group in UNIVERSE.values() for t in group]
SECTOR_OF = {t: sector for sector, group in UNIVERSE.items() for t in group}

# Фиксированные даты, чтобы результаты воспроизводились
START, END = "2021-10-01", "2026-10-01"


def download_prices(tickers=TICKERS, start=START, end=END) -> pd.DataFrame:
    """Скачивает скорректированные цены закрытия (с учётом дивидендов и сплитов)."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"][tickers]
    missing = prices.columns[prices.isna().all()].tolist()
    if missing:
        raise RuntimeError(f"Не скачались тикеры: {missing}")
    return prices


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Цены из локального кэша; скачивает, если кэша нет или refresh=True."""
    if refresh or not PRICES_CSV.exists():
        DATA_DIR.mkdir(exist_ok=True)
        download_prices().to_csv(PRICES_CSV)
    return pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)


def clean_prices(prices: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Заполняет короткие пропуски последней ценой и выкидывает неполные строки.

    Последняя известная цена — это то, что реально видел бы инвестор.
    Интерполяция «из будущего» здесь недопустима (look-ahead bias).
    """
    return prices.ffill(limit=max_gap).dropna()


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = P_t / P_{t-1} - 1. Линейны по весам: доходность портфеля = w^T r."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """l_t = ln(P_t / P_{t-1}). Аддитивны во времени: сумма за n дней = ln(P_n / P_0)."""
    return np.log(prices / prices.shift(1)).dropna()
