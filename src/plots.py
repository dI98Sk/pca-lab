"""Графики проекта. Единый стиль: светлая подложка, приглушённые оси."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

SURFACE = "#fcfcfb"
TEXT, TEXT_MUTED = "#0b0b0b", "#52514e"
BLUE = "#2a78d6"
# Расходящаяся шкала: синий (−) → нейтральный серый (0) → красный (+)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2a2a"]
)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def _save(fig, save_as):
    if save_as:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / save_as, dpi=150, facecolor=SURFACE)


def _sector_labels(ax, groups: dict, x: float, vlines: bool = False):
    """Белые разделители между секторами и подписи секторов справа."""
    edge = 0
    for sector, tickers in groups.items():
        if edge:
            ax.axhline(edge - 0.5, color=SURFACE, lw=2)
            if vlines:
                ax.axvline(edge - 0.5, color=SURFACE, lw=2)
        ax.text(x, edge + len(tickers) / 2 - 0.5, sector, va="center", fontsize=9, color=TEXT_MUTED)
        edge += len(tickers)


def corr_heatmap(corr: pd.DataFrame, groups: dict, title: str, save_as: str | None = None):
    """Тепловая карта корреляций; тонкие линии отделяют сектора."""
    fig, ax = plt.subplots(figsize=(9, 8))
    _style(ax)
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)

    n = len(corr)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    _sector_labels(ax, groups, x=n + 0.3, vlines=True)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.13)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=TEXT_MUTED, length=0, labelsize=8)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=12)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def scree(explained: pd.Series, k: int = 10, title: str = "", save_as: str | None = None):
    """Доля дисперсии первых k компонент; под столбцом — накопленная сумма."""
    ev = explained.iloc[:k] * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _style(ax)
    ax.bar(ev.index, ev.values, color=BLUE, width=0.6)
    for i, (v, cum) in enumerate(zip(ev.values, ev.cumsum().values)):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=9, color=TEXT)
        ax.text(i, -3.2, f"Σ {cum:.0f}%", ha="center", fontsize=8, color=TEXT_MUTED)
    ax.set_ylim(-4.5, ev.max() + 4)
    ax.set_yticks([])
    ax.axhline(0, color="#d9d8d3", lw=1)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def loadings_heatmap(loadings: pd.DataFrame, groups: dict, title: str, save_as: str | None = None):
    """Веса активов в главных компонентах: строки — активы, столбцы — PC."""
    vmax = np.abs(loadings.to_numpy()).max()
    fig, ax = plt.subplots(figsize=(5.4, 8))
    _style(ax)
    ax.imshow(loadings.to_numpy(), cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(loadings.shape[1]), loadings.columns, fontsize=9)
    ax.set_yticks(range(len(loadings)), loadings.index, fontsize=8)
    ax.xaxis.tick_top()
    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            v = loadings.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if abs(v) > 0.6 * vmax else TEXT)
    _sector_labels(ax, groups, x=loadings.shape[1] - 0.4)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=28)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def pca_2d(z: pd.DataFrame, eigenvalues: pd.Series, loadings: pd.DataFrame,
           names=("«оба вместе»", "«один против\nдругого»"), save_as: str | None = None):
    """Облако дней для двух активов и оси PC1/PC2; длина стрелки ∝ √λ."""
    a, b = z.columns
    share = eigenvalues / eigenvalues.sum() * 100
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    _style(ax)
    ax.scatter(z[a], z[b], s=9, color="#9ec5f4", edgecolors="none")
    spec = [("#e34948", (0.1, 0.45), "left"), ("#104281", (-0.1, 0.55), "right")]
    for k, (color, (dx, dy), ha) in enumerate(spec):
        v = loadings.iloc[:, k].to_numpy() * np.sqrt(eigenvalues.iloc[k]) * 2.2
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4, color=color, mutation_scale=18))
        ax.text(v[0] + dx, v[1] + dy, f"PC{k + 1} {names[k]}  {share.iloc[k]:.0f}%",
                color=color, fontsize=10.5, ha=ha, va="center", weight="bold")
    ax.axhline(0, color="#d9d8d3", lw=1, zorder=0)
    ax.axvline(0, color="#d9d8d3", lw=1, zorder=0)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{a}: насколько день отличался от обычного (в σ)", color=TEXT_MUTED)
    ax.set_ylabel(f"{b}: насколько день отличался от обычного (в σ)", color=TEXT_MUTED)
    ax.set_title("Каждая точка — один торговый день", loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def loadings_scatter(loadings: pd.DataFrame, planes=(("PC2", "PC3"), ("PC3", "PC4")), save_as: str | None = None):
    """Активы как точки в координатах своих весов. Один цвет — сектора не подсказываем."""
    fig, axes = plt.subplots(1, len(planes), figsize=(6 * len(planes), 5.6))
    for ax, (x, y) in zip(np.atleast_1d(axes), planes):
        _style(ax)
        ax.axhline(0, color="#d9d8d3", lw=1, zorder=0)
        ax.axvline(0, color="#d9d8d3", lw=1, zorder=0)
        ax.scatter(loadings[x], loadings[y], s=46, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        placed = []  # сдвигаем подпись вниз, пока она не перестанет налезать на соседние
        for t in loadings.sort_values(y, ascending=False).index:
            px, py = loadings.loc[t, x], loadings.loc[t, y]
            ly = py
            while any(abs(px - qx) < 0.07 and abs(ly - qy) < 0.022 for qx, qy in placed):
                ly -= 0.022
            placed.append((px, ly))
            leader = dict(arrowstyle="-", color="#c3c2b7", lw=0.6) if abs(ly - py) > 0.01 else None
            ax.annotate(t, (px, py), xytext=(px + 0.012, ly + 0.008), fontsize=8, color=TEXT, arrowprops=leader)
        ax.set_xlabel(f"вес в {x}", color=TEXT_MUTED)
        ax.set_ylabel(f"вес в {y}", color=TEXT_MUTED)
        ax.tick_params(labelsize=8)
    fig.suptitle("Каждая точка — актив. Координаты — его веса в главных компонентах",
                 x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


SERIES = ("#2a78d6", "#eb6834")  # категориальные слоты 1–2: синий, оранжевый


def weight_vs_risk(table: pd.DataFrame, title: str, save_as: str | None = None):
    """Парные горизонтальные столбцы: доля в весе vs доля в риске (table: колонки weight, share)."""
    t = table.sort_values("share")
    y = np.arange(len(t))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(t) + 1.4))
    _style(ax)
    for k, (col, label) in enumerate([("weight", "доля в весе"), ("share", "доля в риске")]):
        vals = t[col].to_numpy() * 100
        ax.barh(y + (h / 2 + 0.02) * (1 if k == 0 else -1), vals, height=h, color=SERIES[k], label=label)
        for yi, v in zip(y, vals):
            ax.text(v + 0.6, yi + (h / 2 + 0.02) * (1 if k == 0 else -1), f"{v:.0f}%",
                    va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y, t.index, fontsize=9)
    ax.set_xticks([])
    ax.axvline(0, color="#d9d8d3", lw=1)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


def lagrange_geometry(corr2: pd.DataFrame, cov2: pd.DataFrame, save_as: str | None = None):
    """Геометрия метода Лагранжа: касание линии уровня wᵀΣw и ограничения.

    Слева — PCA (max vᵀCv на окружности vᵀv = 1) для пары активов corr2,
    справа — min variance (min wᵀΣw на прямой w₁ + w₂ = 1) для пары cov2 (годовая Σ).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))

    # --- PCA: окружность и линии уровня vᵀCv
    C = corr2.to_numpy()
    a, b = corr2.columns
    lam, Q = np.linalg.eigh(C)
    ax = axes[0]
    _style(ax)
    g = np.linspace(-1.6, 1.6, 400)
    X, Y = np.meshgrid(g, g)
    F = C[0, 0] * X ** 2 + 2 * C[0, 1] * X * Y + C[1, 1] * Y ** 2
    levels = sorted({0.2, round(lam[0], 2), 0.8, 1.2, round(lam[1], 2), 2.0, 2.6})
    cs = ax.contour(X, Y, F, levels=levels, colors="#86b6ef", linewidths=1)
    ax.clabel(cs, fontsize=7, fmt="%.2f")
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), color=TEXT, lw=1.8)
    for k, label, color in [(1, "максимум: PC1", "#e34948"), (0, "минимум: PC2", "#104281")]:
        v = Q[:, k] * (1 if Q[:, k].sum() >= 0 else -1)
        if k == 0 and v[1] > 0:
            v = -v
        ax.scatter(*v, s=60, color=color, zorder=5)
        ax.annotate("", xy=v + 0.22 * (2 * C @ v), xytext=v,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
        ax.text(v[0] + 0.08, v[1] + (0.62 if v[1] > 0 else -0.62), f"{label}, λ = {lam[k]:.2f}",
                color=color, fontsize=9.5, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel(f"v₁ (вес {a})", color=TEXT_MUTED)
    ax.set_ylabel(f"v₂ (вес {b})", color=TEXT_MUTED)
    ax.set_title("PCA: max vᵀΣv на окружности vᵀv = 1", loc="left", color=TEXT, fontsize=11.5)

    # --- Min variance: прямая w₁ + w₂ = 1 и линии уровня σ
    S = cov2.to_numpy()
    a, b = cov2.columns
    x = np.linalg.solve(S, np.ones(2))
    w = x / x.sum()
    var_min = w @ S @ w
    ax = axes[1]
    _style(ax)
    g = np.linspace(-0.4, 1.4, 400)
    X, Y = np.meshgrid(g, g)
    F = S[0, 0] * X ** 2 + 2 * S[0, 1] * X * Y + S[1, 1] * Y ** 2
    cs = ax.contour(X, Y, np.sqrt(F), levels=np.sqrt(var_min * np.array([0.5, 1, 2, 4, 8, 14])),
                    colors="#86b6ef", linewidths=1)
    ax.clabel(cs, fontsize=7, fmt=lambda s: f"σ={s:.0%}")
    ax.plot(g, 1 - g, color=TEXT, lw=1.8)
    ax.scatter(*w, s=60, color="#e34948", zorder=5)
    ax.annotate("", xy=w + 4 * (2 * S @ w), xytext=w,
                arrowprops=dict(arrowstyle="-|>", color="#e34948", lw=1.8))
    ax.text(1.0, 0.5, f"минимум:\n{a} {w[0]:.0%}, {b} {w[1]:.0%}\nσ = {np.sqrt(var_min):.1%}",
            color="#e34948", fontsize=9.5, weight="bold", ha="left")
    ax.text(0.22, 0.86, "w₁ + w₂ = 1", color=TEXT, fontsize=9, rotation=-45, rotation_mode="anchor")
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel(f"w₁ (вес {a})", color=TEXT_MUTED)
    ax.set_ylabel(f"w₂ (вес {b})", color=TEXT_MUTED)
    ax.set_title("Min variance: min wᵀΣw на прямой w₁ + w₂ = 1", loc="left", color=TEXT, fontsize=11.5)

    fig.tight_layout()
    _save(fig, save_as)
    return fig


def efficient_frontier(assets: pd.DataFrame, curves: dict, points: dict, title: str, save_as: str | None = None):
    """Граница эффективности: x — риск, y — доходность (годовые, доли).

    assets: колонки vol, ret, индекс — тикеры (серые точки);
    curves: {подпись: DataFrame(vol, ret)} — до двух линий (синяя, оранжевая);
    points: {подпись: (vol, ret)} — отмеченные портфели.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    _style(ax)
    ax.scatter(assets.vol * 100, assets.ret * 100, s=22, color="#c3c2b7", zorder=2)
    for t in assets.index:
        ax.annotate(t, (assets.vol[t] * 100, assets.ret[t] * 100), xytext=(4, 2),
                    textcoords="offset points", fontsize=7, color=TEXT_MUTED)
    for (label, df), color in zip(curves.items(), SERIES):
        ax.plot(df.vol * 100, df.ret * 100, color=color, lw=2, label=label, zorder=3)
    for label, (v, m) in points.items():
        ax.scatter(v * 100, m * 100, s=70, color=TEXT, edgecolors=SURFACE, linewidths=1.5, zorder=5)
        ax.annotate(label, (v * 100, m * 100), xytext=(8, -4), textcoords="offset points",
                    fontsize=9, color=TEXT, weight="bold")
    ax.axhline(0, color="#d9d8d3", lw=1, zorder=0)
    ax.set_xlabel("риск, % годовых (σ)", color=TEXT_MUTED)
    ax.set_ylabel("ожидаемая доходность, % годовых (μ)", color=TEXT_MUTED)
    ax.grid(color="#efeeeb", lw=0.8, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    _save(fig, save_as)
    return fig


SERIES3 = ("#2a78d6", "#eb6834", "#1baf7a")  # категориальные слоты 1–3


def wealth_curves(returns: dict, title: str, save_as: str | None = None):
    """Рост $1 для нескольких стратегий (до трёх), с прямыми подписями у концов линий."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _style(ax)
    ends = []
    for (label, r), color in zip(returns.items(), SERIES3):
        wealth = (1 + r).cumprod()
        ax.plot(wealth.index, wealth.values, color=color, lw=2, label=label)
        ends.append((wealth.iloc[-1], label, color))
    for value, label, color in sorted(ends):
        ax.annotate(f"${value:.2f}", (wealth.index[-1], value), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT)
    ax.axhline(1, color="#d9d8d3", lw=1, zorder=0)
    ax.grid(axis="y", color="#efeeeb", lw=0.8, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT)
    ax.set_ylabel("стоимость $1", color=TEXT_MUTED)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    _save(fig, save_as)
    return fig
