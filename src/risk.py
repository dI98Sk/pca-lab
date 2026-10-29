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
