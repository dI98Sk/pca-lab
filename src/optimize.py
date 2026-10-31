"""Mean-variance оптимизация (Марковиц).

Везде решаем вариации одной задачи: min wᵀΣw при ограничениях на веса.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def min_variance(cov: pd.DataFrame) -> pd.Series:
    """Портфель минимальной дисперсии без ограничений на знак весов.

    min wᵀΣw  при  1ᵀw = 1.
    Лагранжиан: wᵀΣw − γ(1ᵀw − 1) → 2Σw = γ1 → w ∝ Σ⁻¹1, затем нормируем на сумму.
    """
    ones = np.ones(len(cov))
    x = np.linalg.solve(cov.to_numpy(), ones)    # x = Σ⁻¹1 без явного обращения матрицы
    return pd.Series(x / x.sum(), index=cov.index)


def min_variance_long_only(cov: pd.DataFrame) -> pd.Series:
    """То же, но веса w_i ≥ 0 (без коротких продаж). Аналитического решения нет — численно.

    Передавать годовую Σ: на дневной дисперсия очень мала и SLSQP останавливается раньше времени.
    """
    S = cov.to_numpy()
    n = len(S)
    res = minimize(
        lambda w: w @ S @ w,
        x0=np.full(n, 1 / n),
        jac=lambda w: 2 * S @ w,
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(res.message)
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=cov.index)


def expected_returns(returns: pd.DataFrame) -> pd.Series:
    """Историческая оценка μ: средняя дневная доходность × 252."""
    return returns.mean() * 252


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """Портфель с максимальным Sharpe = (wᵀμ − rf) / √(wᵀΣw), без ограничений на знак.

    Sharpe не меняется при умножении w на число, поэтому достаточно найти направление:
    производная = 0 → Σw ∝ (μ − rf) → w ∝ Σ⁻¹(μ − rf), затем нормируем на сумму.
    """
    x = np.linalg.solve(cov.to_numpy(), (mu - rf).to_numpy())
    return pd.Series(x / x.sum(), index=cov.index)


def max_sharpe_fixed_risk(mu: pd.Series, cov: pd.DataFrame, target_vol: float = 0.10, rf: float = 0.0) -> pd.Series:
    """То же направление w ∝ Σ⁻¹(μ − rf), но масштаб задаёт прогнозный риск, а не сумма весов.

    Деление на 1ᵀΣ⁻¹μ раздувает позиции, когда эта сумма близка к нулю. Здесь √(wᵀΣw) = target_vol,
    сумма весов может отличаться от 1: разница — кэш или заём под нулевой процент.
    """
    S = cov.to_numpy()
    x = np.linalg.solve(S, (mu - rf).to_numpy())
    return pd.Series(x * target_vol / np.sqrt(x @ S @ x), index=cov.index)


def max_sharpe_long_only(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """Max Sharpe при w_i ≥ 0. Минимизируем −Sharpe численно."""
    S, m = cov.to_numpy(), (mu - rf).to_numpy()
    n = len(S)
    res = minimize(
        lambda w: -(w @ m) / np.sqrt(w @ S @ w),
        x0=np.full(n, 1 / n),
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(res.message)
    w = np.clip(res.x, 0, None)
    return pd.Series(w / w.sum(), index=cov.index)


def frontier(mu: pd.Series, cov: pd.DataFrame, targets) -> pd.DataFrame:
    """Граница эффективности без ограничений на знак: min wᵀΣw при 1ᵀw = 1 и μᵀw = m.

    Два ограничения → два множителя Лагранжа: 2Σw = γ1 + δμ → w = Σ⁻¹(a·1 + b·μ).
    Коэффициенты a, b находим из двух ограничений — система 2×2.
    """
    S, m = cov.to_numpy(), mu.to_numpy()
    ones = np.ones(len(m))
    inv_1, inv_mu = np.linalg.solve(S, ones), np.linalg.solve(S, m)
    A, B, C = ones @ inv_1, ones @ inv_mu, m @ inv_mu     # 1ᵀΣ⁻¹1, 1ᵀΣ⁻¹μ, μᵀΣ⁻¹μ
    rows = []
    for target in targets:
        a, b = np.linalg.solve([[A, B], [B, C]], [1, target])
        w = a * inv_1 + b * inv_mu
        rows.append({"ret": target, "vol": np.sqrt(w @ S @ w)})
    return pd.DataFrame(rows)


def frontier_long_only(mu: pd.Series, cov: pd.DataFrame, targets) -> pd.DataFrame:
    """Граница эффективности при w_i ≥ 0: для каждой целевой доходности — численная минимизация риска."""
    S, m = cov.to_numpy(), mu.to_numpy()
    n = len(m)
    rows, w0 = [], np.full(n, 1 / n)
    for target in targets:
        res = minimize(
            lambda w: w @ S @ w,
            x0=w0,
            jac=lambda w: 2 * S @ w,
            bounds=[(0, 1)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                         {"type": "eq", "fun": lambda w, t=target: w @ m - t}],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if res.success:
            w0 = res.x                                   # тёплый старт для следующей точки
            rows.append({"ret": target, "vol": np.sqrt(res.x @ S @ res.x)})
    return pd.DataFrame(rows)
