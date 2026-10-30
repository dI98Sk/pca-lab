"""Ковариационная матрица и её свойства.

Всё считается «руками» через numpy, чтобы было видно линейную алгебру:
Σ = XᵀX / (T−1), где X — доходности за вычетом средних.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252  # торговых дней в году: дневную дисперсию умножаем на 252


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Σ = XᵀX / (T−1). Элемент (i, j) — скалярное произведение столбцов i и j."""
    R = returns.to_numpy()
    X = R - R.mean(axis=0)              # центрирование: каждый столбец в сумме 0
    S = X.T @ X / (len(X) - 1)          # (N×T)·(T×N) = N×N
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)


def correlation_from_cov(cov: pd.DataFrame) -> pd.DataFrame:
    """C = D⁻¹ Σ D⁻¹, где D = diag(σ₁, …, σ_N).

    Деление строки i и столбца j на σᵢ и σⱼ: Cᵢⱼ = Σᵢⱼ / (σᵢσⱼ).
    Это та же самая ковариация, но у «выровненных» активов с дисперсией 1.
    """
    d_inv = np.diag(1 / np.sqrt(np.diag(cov)))
    return pd.DataFrame(d_inv @ cov.to_numpy() @ d_inv, index=cov.index, columns=cov.columns)


def annual_vol(cov: pd.DataFrame) -> pd.Series:
    """Годовая волатильность: √(252·σ²_daily). Дисперсия растёт ∝ времени, σ — ∝ √времени."""
    return np.sqrt(np.diag(cov) * TRADING_DAYS) * pd.Series(1.0, index=cov.index)


def spectrum(cov: pd.DataFrame) -> np.ndarray:
    """Собственные значения по убыванию. eigvalsh — для симметричных матриц."""
    return np.linalg.eigvalsh(cov.to_numpy())[::-1]


def condition_number(cov: pd.DataFrame) -> float:
    """κ = λ_max / λ_min: во сколько раз Σ⁻¹ может усилить ошибку во входных данных."""
    lam = spectrum(cov)
    return lam[0] / lam[-1]


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    """σ_p = √(wᵀΣw). Для годовой — передавать годовую Σ (дневная × 252)."""
    w = weights.reindex(cov.index).fillna(0).to_numpy()
    return float(np.sqrt(w @ cov.to_numpy() @ w))


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Вклад каждого актива в риск портфеля.

    marginal_i = ∂σ_p/∂w_i = (Σw)_i / σ_p  — на сколько вырастет риск, если чуть добавить актива i;
    contrib_i  = w_i · marginal_i          — вклад актива; сумма вкладов ровно σ_p (теорема Эйлера).
    """
    w = weights.reindex(cov.index).fillna(0)
    sigma_w = cov.to_numpy() @ w.to_numpy()
    vol = np.sqrt(w.to_numpy() @ sigma_w)
    marginal = sigma_w / vol
    contrib = w.to_numpy() * marginal
    return pd.DataFrame({"weight": w, "marginal": marginal, "contrib": contrib,
                         "share": contrib / vol}, index=cov.index)


def factor_risk(weights: pd.Series, eigenvalues: pd.Series, loadings: pd.DataFrame) -> pd.DataFrame:
    """Разложение дисперсии по главным компонентам: wᵀΣw = Σ_k λ_k (q_kᵀw)².

    exposure_k = q_kᵀw — «сколько фактора k сидит в портфеле».
    Работает с PCA ковариационной матрицы: риск считаем в деньгах, «громкость» важна.
    """
    w = weights.reindex(loadings.index).fillna(0).to_numpy()
    exposure = loadings.to_numpy().T @ w
    variance = eigenvalues.to_numpy() * exposure ** 2
    return pd.DataFrame({"exposure": exposure, "variance": variance,
                         "share": variance / variance.sum()}, index=loadings.columns)
