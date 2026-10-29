"""PCA через спектральное разложение Σ = QΛQᵀ.

Столбец qₖ матрицы Q — k-я главная компонента: веса, с которыми активы
входят в k-й «скрытый фактор». λₖ — дисперсия этого фактора.
"""
import numpy as np
import pandas as pd


def pca(matrix: pd.DataFrame):
    """Собственные значения (по убыванию) и векторы ковариационной/корреляционной матрицы.

    Знак собственного вектора произволен (q и −q одинаково верны), поэтому
    выбираем его так, чтобы сумма весов была положительной — тогда
    «фактор вырос» читается как «активы в среднем выросли».
    """
    lam, Q = np.linalg.eigh(matrix.to_numpy())   # eigh: для симметричных, по возрастанию
    order = np.argsort(lam)[::-1]
    lam, Q = lam[order], Q[:, order]
    Q = Q * np.where(Q.sum(axis=0) < 0, -1, 1)

    names = [f"PC{k + 1}" for k in range(len(lam))]
    eigenvalues = pd.Series(lam, index=names)
    loadings = pd.DataFrame(Q, index=matrix.index, columns=names)
    return eigenvalues, loadings


def explained_variance(eigenvalues: pd.Series) -> pd.Series:
    """Доля дисперсии k-го фактора: λₖ / Σλ. Сумма λ равна следу матрицы."""
    return eigenvalues / eigenvalues.sum()


def standardize(returns: pd.DataFrame) -> pd.DataFrame:
    """(r − μ) / σ по каждому столбцу: все активы с дисперсией 1."""
    return (returns - returns.mean()) / returns.std()


def factor_returns(returns: pd.DataFrame, loadings: pd.DataFrame) -> pd.DataFrame:
    """Ряды факторов F = XQ: проекция каждого дня на оси главных компонент."""
    X = returns - returns.mean()
    return X @ loadings
