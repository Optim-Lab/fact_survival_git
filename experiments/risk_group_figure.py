"""risk_group_<데이터셋>.csv 로 그림과 표를 만든다.  (python risk_group_curves.py figure)

그림: 가로 = 데이터셋, 세로 = 모델 4개.  x = 예측 기간, y = 생존 비율(%).  실선 = 예측 평균, 점선 = 실제.  색 = 위험군 (3군).
표  : 모델 전부.  Separation = (위험 낮은 군 실제 생존) - (위험 높은 군 실제 생존), 가장 긴 기간, %p (클수록 군이 잘 갈린다).
      Gap = 예측과 실제 생존의 차이 |pred - obs| 를 군·기간에 걸쳐 평균, %p (작을수록 예측이 군별로 맞는다).
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import budget_curve as K
from budget_curve_figures import horizon_name
from make_budget_figures import DATASETS, LABEL, MODELS, OUT, set_style
from risk_group_curves import NAME

SHOWN = ["LaST", "LogCL + head", "RE-GCN + head", "DeepSurv"]           # 그림에 보일 모델 (표에는 전부)
COLOR = ["#D62728", "#7F7F7F", "#1F77B4"]                              # 위험 높음 -> 낮음


def read(dataset):
    t = pd.read_csv(K.C.HERE / f"risk_group_{dataset}.csv")
    return t.groupby(["model", "group", "horizon"])[["pred", "obs", "rows"]].mean().reset_index()     # 시드 평균


def table(dataset):
    t, rows = read(dataset), []
    for model, g in t.groupby("model"):
        last = g[g.horizon == g.horizon.max()].set_index("group").obs
        rows.append(dict(dataset=dataset, model=model, separation=100 * (last[2] - last[0]), gap=100 * (g.pred - g.obs).abs().mean()))
    return pd.DataFrame(rows)


def main():
    set_style()
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 20, "axes.titlesize": 22, "xtick.labelsize": 15, "ytick.labelsize": 15})
    fig, axes = plt.subplots(len(SHOWN), len(DATASETS), figsize=(4.6 * len(DATASETS), 3.4 * len(SHOWN)), squeeze=False)
    for col, d in enumerate(DATASETS):
        t = read(d)
        axes[0, col].set_title(DATASETS[d], pad=10)
        for row, model in enumerate(SHOWN):
            ax = axes[row, col]
            for g in range(3):
                part = t[(t.model == model) & (t.group == g)]
                x = list(range(len(part)))
                ax.plot(x, 100 * part.obs, color=COLOR[g], ls="--", lw=1.5, label=f"{NAME[g]} (observed)")                 # 점선 = 실제
                ax.plot(x, 100 * part.pred, color=COLOR[g], marker="o", lw=2.5, label=f"{NAME[g]} (predicted)")            # 실선 = 예측
            ax.set_xticks(x)
            ax.set_xticklabels([horizon_name(d, h).split()[0] for h in part.horizon])
            ax.grid(True)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(LABEL.get(model, model), fontsize=15)
    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    fig.supxlabel("Prediction horizon (months; YAGO: years)", y=0.0, fontsize=20)
    fig.supylabel("Survival rate (%)", x=0.0, fontsize=20)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=17)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_risk_group.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    out = pd.concat([table(d) for d in DATASETS])
    out.to_csv(K.C.HERE / "risk_group_summary.csv", index=False)
    for d, g in out.groupby("dataset"):
        print(f"\n{d}  (Separation 큰 순, %p)")
        print(g.sort_values("separation", ascending=False)[["model", "separation", "gap"]].round(2).to_string(index=False))
