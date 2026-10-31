"""Walk-forward тест: веса считаются только по прошлому окну и держатся следующий период."""
import numpy as np
import pandas as pd

from src.risk import TRADING_DAYS


def walk_forward(returns: pd.DataFrame, strategy, window: int = 504, step: int = 21) -> dict:
    """strategy(train_returns) -> веса (pd.Series). Возвращает ряд доходностей и журнал пересчётов.

    Допущения: веса фиксированы внутри периода (ежедневная ребалансировка к целевым),
    без транзакционных издержек.
    """
    pieces, log, prev = [], [], None
    for start in range(window, len(returns), step):
        train = returns.iloc[start - window:start]
        test = returns.iloc[start:start + step]
        w = strategy(train).reindex(returns.columns).fillna(0.0)
        cov = train.cov().to_numpy() * TRADING_DAYS
        log.append({
            "date": test.index[0],
            "predicted_vol": float(np.sqrt(w @ cov @ w)),
            "turnover": float(np.abs(w - prev).sum() / 2) if prev is not None else np.nan,
            "gross": float(w.abs().sum()),
            "max_weight": float(w.max()),
        })
        pieces.append(test @ w)
        prev = w
    return {"returns": pd.concat(pieces), "log": pd.DataFrame(log).set_index("date")}


def sharpe(r) -> float:
    """Годовой Sharpe при нулевой безрисковой ставке."""
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))


def bootstrap_diff(a: pd.Series, b: pd.Series, stat=sharpe, block: int = 21, n: int = 5000, seed: int = 0) -> dict:
    """Блочный бутстрэп разности stat(a) − stat(b): насколько она отличима от шума.

    Берём случайные блоки по 21 дню, одни и те же для обеих стратегий: блоки сохраняют
    связь соседних дней, общие блоки — связь стратегий между собой.
    """
    x, y = a.to_numpy(), b.to_numpy()
    rng = np.random.default_rng(seed)
    T, diffs = len(x), []
    for _ in range(n):
        starts = rng.integers(0, T - block + 1, T // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        diffs.append(stat(x[idx]) - stat(y[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"разность": stat(x) - stat(y), "95% от": low, "95% до": high, "P(разность > 0)": float(np.mean(np.array(diffs) > 0))}


def summarize(result: dict) -> dict:
    """Метрики теста: доходность, риск, Sharpe, обещанный риск и оборот (всё годовое)."""
    r, log = result["returns"], result["log"]
    ret = r.mean() * TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    wealth = (1 + r).cumprod()
    return {
        "доходность": ret,
        "риск": vol,
        "Sharpe": ret / vol,
        "обещанный риск": log.predicted_vol.mean(),
        "риск факт/обещ.": vol / log.predicted_vol.mean(),
        "макс. просадка": (wealth / wealth.cummax() - 1).min(),
        "оборот в год": log.turnover.mean() * TRADING_DAYS / 21,
        "плечо (Σ|w|)": log.gross.mean(),
    }
