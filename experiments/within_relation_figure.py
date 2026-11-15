"""관계 안 위험군 곡선 그림 두 장.   python within_relation_figure.py     (within_relation_<데이터셋>.csv 가 있어야 한다)

실선 = 예측 생존확률 평균, 점선 = 실제 생존 비율.  색 = 위험군 (같은 관계 안에서 나눈 3군).  x = 예측 기간.  y = 생존 비율(%).
모델 = LaST 와 LR (Relation+age).  같은 관계를 그린 두 패널은 y축 범위를 같게 한다 (두 모델의 군 간격을 눈으로 비교하려고).
관계 = relation_choice_<데이터셋>.csv 의 순위 순서에서 EXCLUDE (relation_lifetime.py 와 같은 목록) 를 뺀 앞의 몇 개.

  fig_within_relation_main.*   본문용.  4행 (데이터셋마다 1위 관계) x 2열 (LaST, LR).
  fig_within_relation.*        부록용.  6행 (관계 3개 x 모델 2개) x 4열 (데이터셋).
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import budget_curve as K
from budget_curve_figures import horizon_name
from make_budget_figures import DATASETS, OUT, set_style
from relation_lifetime import EXCLUDE
from risk_group_curves import NAME

MODELS = {"LaST": "LaST", "Logistic": "LR (Relation+age)"}
COLOR = ["#D62728", "#7F7F7F", "#1F77B4"]


def read(dataset):
    """시드 평균 (pred, obs) + 실제 비율의 시드 표준편차 (obs_sd, 음영).  시드가 1개인 모델은 0."""
    g = pd.read_csv(K.C.HERE / f"within_relation_{dataset}.csv").groupby(["model", "relation", "group", "horizon"])
    return g[["pred", "obs"]].mean().join(g["obs"].std(ddof=0).rename("obs_sd")).reset_index()


def relations(dataset, how_many):
    """순위 순서에서 뺄 관계를 거른 앞의 how_many 개."""
    ranked = pd.read_csv(K.C.HERE / f"relation_choice_{dataset}.csv").relation.tolist()
    return [r for r in ranked if r not in EXCLUDE.get(dataset, [])][:how_many]


def y_range(t, relation):
    """그 관계의 두 모델이 쓰는 y 범위 (군 3개, 실제와 예측 모두)."""
    part = t[(t.relation == relation) & t.model.isin(MODELS)]
    low, high = 100 * min(part.obs.min(), part.pred.min()), 100 * max(part.obs.max(), part.pred.max())
    pad = 0.06 * (high - low)
    return low - pad, high + pad


def draw(ax, t, dataset, relation, model, ylim):
    for g in range(3):
        part = t[(t.model == model) & (t.relation == relation) & (t.group == g)]
        x = list(range(len(part)))
        ax.fill_between(x, 100 * (part.obs - part.obs_sd), 100 * (part.obs + part.obs_sd), color=COLOR[g], alpha=0.15, lw=0)       # 시드 ±1 표준편차
        ax.plot(x, 100 * part.obs, color=COLOR[g], ls="--", lw=1.3, label=f"{NAME[g]} (observed)")                       # 점선 = 실제
        ax.plot(x, 100 * part.pred, color=COLOR[g], marker="o", lw=2.2, label=f"{NAME[g]} (predicted)")                  # 실선 = 예측
    ax.set_xticks(x)
    ax.set_xticklabels([horizon_name(dataset, h).split()[0] for h in part.horizon])
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.set_axisbelow(True)


def finish(fig, axes, name, label_size):
    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    fig.supxlabel("Prediction horizon (months; YAGO: years)", y=0.0, fontsize=label_size)
    fig.supylabel("Survival rate (%)", x=0.0, fontsize=label_size)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=14)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main_figure():
    """본문용: 열 = 데이터셋 (그 데이터셋의 1위 관계), 행 = 모델 (LaST, LR).  가로로 넓은 2 x 4."""
    set_style()
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 16, "axes.titlesize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13})
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(4.0 * len(DATASETS), 6.6), squeeze=False)
    for col, d in enumerate(DATASETS):
        t, rel = read(d), relations(d, 1)[0]
        ylim = y_range(t, rel)                                              # 한 열의 두 모델은 y 범위를 같게
        axes[0, col].set_title(f"{DATASETS[d]}: {rel}", pad=8)
        for row, (model, label) in enumerate(MODELS.items()):
            ax = axes[row, col]
            draw(ax, t, d, rel, model, ylim)
            ax.text(0.03, 0.05, label, transform=ax.transAxes, fontsize=12, va="bottom")
    finish(fig, axes, "fig_within_relation_main", label_size=16)


def appendix_figure():
    """부록용: 열 = 데이터셋, 행 = 관계 3개 x 모델 2개."""
    set_style()
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 18, "axes.titlesize": 20, "xtick.labelsize": 13, "ytick.labelsize": 13})
    fig, axes = plt.subplots(6, len(DATASETS), figsize=(4.6 * len(DATASETS), 2.9 * 6), squeeze=False)
    for col, d in enumerate(DATASETS):
        t = read(d)
        axes[0, col].set_title(DATASETS[d], pad=10)
        for i, rel in enumerate(relations(d, 3)):
            ylim = y_range(t, rel)
            for j, (model, label) in enumerate(MODELS.items()):
                ax = axes[2 * i + j, col]
                draw(ax, t, d, rel, model, ylim)
                ax.text(0.03, 0.05, f"{rel} | {label}", transform=ax.transAxes, fontsize=11, va="bottom")
    finish(fig, axes, "fig_within_relation", label_size=18)


if __name__ == "__main__":
    main_figure()
    appendix_figure()
