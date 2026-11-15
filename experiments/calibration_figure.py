"""calibration_plot.py 가 만든 calibration_<데이터셋>.csv 로 그림(3 기간 x 4 데이터셋)과 ECE 표를 만든다.  (python calibration_plot.py figure)"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import budget_curve as K
from budget_curve_figures import horizon_name, horizons
from make_budget_figures import COLOR, DATASETS, LABEL, MARKER, MODELS, OUT, set_style

COLOR["KM"], MARKER["KM"] = "#555555", "*"                           # Random 이 이 그림에 없어서 회색을 KM 에 쓴다
LEARNED = ["KM"] + [m for m in MODELS if m != "Random"]          # 범례 맨 앞에 KM


def read(dataset):
    return pd.read_csv(K.C.HERE / f"calibration_{dataset}.csv")


def seed_mean(t):
    """구간 번호별 시드 평균 (점 위치) + 관측 비율의 시드 표준편차 (음영).  시드가 1개인 모델은 표준편차 0."""
    g = t.groupby(["model", "horizon", "bin"])
    return g[["predicted", "observed"]].mean().join(g["observed"].std(ddof=0).rename("sd")).reset_index()


def ece_table(t):
    """모델·기간별 ECE, 예측/실제 종료 수 비율 (시드 평균).  행 수 가중."""
    g = t.assign(gap=(t.predicted - t.observed).abs() * t.rows, expected=t.predicted * t.rows, actual=t.observed * t.rows)
    s = g.groupby(["model", "seed", "horizon"]).agg(gap=("gap", "sum"), rows=("rows", "sum"), expected=("expected", "sum"), actual=("actual", "sum")).reset_index()
    s["ECE_equal_mass"], s["ratio"] = 100 * s.gap / s.rows, s.expected / s.actual
    return s.groupby(["model", "horizon"])[["ECE_equal_mass", "ratio"]].mean()


def main():
    set_style()
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 22, "axes.titlesize": 24, "xtick.labelsize": 16, "ytick.labelsize": 16})
    fig, axes = plt.subplots(3, len(DATASETS), figsize=(4.6 * len(DATASETS), 11), squeeze=False)
    summary = []
    for col, d in enumerate(DATASETS):
        t = read(d)
        points, ece = seed_mean(t), ece_table(t)
        standard = pd.read_csv(K.C.HERE / f"calibration_standard_{d}.csv").groupby(["model", "horizon"]).ECE_standard.mean()
        summary.append(ece.join(standard).assign(dataset=d))
        axes[0, col].set_title(DATASETS[d], pad=10)
        for row, h in enumerate(horizons(t.set_index(["model", "horizon", "bin"]))):
            ax = axes[row, col]
            top = 0
            for model in [m for m in LEARNED if m in set(t.model)]:
                p = points[(points.model == model) & (points.horizon == h)]
                last = model == "LaST"                                       # 선 굵기, 마커 크기, 테두리는 예산 곡선 그림(budget_curve_figures.py 의 draw)와 같다
                ax.fill_between(p.predicted * 100, (p.observed - p.sd) * 100, (p.observed + p.sd) * 100, color=COLOR[model], alpha=0.15, lw=0)    # 시드 ±1 표준편차
                ax.plot(p.predicted * 100, p.observed * 100, color=COLOR[model], marker=MARKER[model], lw=3 if last else 2, ms=8,
                        mfc=COLOR[model] if last else "white", zorder=5 if last else 3, label=LABEL.get(model, model))
                top = max(top, p.predicted.max() * 100, p.observed.max() * 100)
            ax.plot([0, top], [0, top], "--", color="black", lw=1.2)          # 대각선 (완벽히 보정된 모델): 라벨을 안 붙여서 범례에는 안 나온다
            ax.grid(True)
            ax.set_axisbelow(True)
            ax.text(0.04, 0.95, horizon_name(d, h), transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    fig.supxlabel("Predicted termination probability (%)", y=0.0, fontsize=22)
    fig.supylabel("Observed termination rate (%)", x=0.0, fontsize=22)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=22)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_calibration.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    table = pd.concat(summary).reset_index()
    table.to_csv(K.C.HERE / "calibration_ece.csv", index=False)
    for (d, h), g in table.groupby(["dataset", "horizon"]):
        for name in ("ECE_equal_mass", "ECE_standard"):
            g = g.sort_values(name)
            print(f"{d} h={h:<2} {name:<14} 낮은 순:", "  ".join(f"{m}:{e:.2f}" for m, e in zip(g.model, g[name])))
    main_figure()
    longest_figure()


MAIN_MODELS = ["KM", "Logistic", "DeepSurv", "RE-GCN + head", "LogCL + head", "LaST"]     # 본문 그림에 보일 모델 (전부는 부록 그림)


def main_figure():
    """본문용: 데이터셋마다 가장 긴 기간 하나씩 (2 x 2), 모델 6개, LaST 를 굵게.
    축 범위 = LaST 가 닿는 곳의 1.25 배 (정사각형).  그 밖으로 가는 모델은 선이 잘려 보이므로 패널 아래에 그 모델 이름을 적는다."""
    set_style()
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 17, "axes.titlesize": 18, "xtick.labelsize": 13, "ytick.labelsize": 13})
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0), squeeze=False)
    for ax, d in zip(axes.ravel(), DATASETS):
        t = read(d)
        points = seed_mean(t)
        h = points.horizon.max()                                                   # 가장 긴 기간
        last = points[(points.model == "LaST") & (points.horizon == h)]
        lim = min(100, 1.25 * 100 * max(last.predicted.max(), last.observed.max()))      # 확률은 100% 를 넘지 않는다
        outside = []
        for model in MAIN_MODELS:
            p = points[(points.model == model) & (points.horizon == h)]
            if p.predicted.max() * 100 > lim:
                outside.append(LABEL.get(model, model))
            big = model == "LaST"
            ax.fill_between(p.predicted * 100, (p.observed - p.sd) * 100, (p.observed + p.sd) * 100, color=COLOR[model], alpha=0.25 if big else 0.12, lw=0)    # 시드 ±1 표준편차
            ax.plot(p.predicted * 100, p.observed * 100, color=COLOR[model], marker=MARKER[model], lw=3.6 if big else 1.4, ms=8 if big else 5,
                    mfc=COLOR[model] if big else "white", alpha=1 if big else 0.8, zorder=6 if big else 3, label=LABEL.get(model, model))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{DATASETS[d]} ({horizon_name(d, h)})", pad=8)
        ax.grid(True)
        ax.set_axisbelow(True)
        if outside:
            ax.text(0.97, 0.03, "beyond range:\n" + ", ".join(outside), transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, color="#444444")
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    fig.supxlabel("Predicted termination probability (%)", y=0.0, fontsize=17)
    fig.supylabel("Observed termination rate (%)", x=0.0, fontsize=17)
    hnd, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=14)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_calibration_main.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def longest_figure():
    """데이터셋마다 가장 긴 기간 하나, 가로 1 x 4, 모델 전부, 예산 곡선과 같은 스타일.  축은 자동이라 어떤 선도 잘리지 않는다 (대신 LaST 가 작아 보일 수 있다)."""
    set_style()
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 20, "axes.titlesize": 22, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.6 * len(DATASETS), 4.6), squeeze=False)
    for ax, d in zip(axes[0], DATASETS):
        t = read(d)
        points = seed_mean(t)
        h = points.horizon.max()
        top = 0
        for model in [m for m in LEARNED if m in set(t.model)]:
            p = points[(points.model == model) & (points.horizon == h)]
            last = model == "LaST"
            ax.fill_between(p.predicted * 100, (p.observed - p.sd) * 100, (p.observed + p.sd) * 100, color=COLOR[model], alpha=0.15, lw=0)
            ax.plot(p.predicted * 100, p.observed * 100, color=COLOR[model], marker=MARKER[model], lw=3 if last else 2, ms=8,
                    mfc=COLOR[model] if last else "white", zorder=5 if last else 3, label=LABEL.get(model, model))
            top = max(top, p.predicted.max() * 100, p.observed.max() * 100)
        ax.plot([0, top], [0, top], "--", color="black", lw=1.2)          # 대각선 (완벽히 보정된 모델): 라벨이 없어서 범례에는 안 나온다
        ax.set_title(f"{DATASETS[d]} ({horizon_name(d, h)})", pad=8)
        ax.grid(True)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    fig.supxlabel("Predicted termination probability (%)", y=0.0, fontsize=20)
    fig.supylabel("Observed termination rate (%)", x=0.0, fontsize=20)
    hnd, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=18)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_calibration_longest.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
