"""예산 곡선의 그림과 표 (budget_curve.py 가 만든 curve_<데이터셋>.csv 를 읽는다).

    python budget_curve_figures.py

그림: 가로 = 데이터셋, 세로 = 기간.  x = 본 사실 비율(%).  띠 = 시드 평균 ± 표준편차.  Perfect(완벽) 은 그림에 안 그리고 표에만 둔다.
지표 두 가지 (같은 줄 세우기, 나누는 수만 다르다):
  Recall    = 찾은 종료 수 / 전체 종료 수   -> fig_budget_curve.*            table_budget_curve.tex
  Precision = 찾은 종료 수 / 본 사실 수     -> fig_budget_curve_precision.*  table_budget_curve_precision.tex
표  : 같은 값을 숫자로 (가장 높은 모델 = 굵게, Random·Perfect 는 굵게 안 함).
"""
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import budget_curve as K
from make_budget_figures import COLOR, DATASETS, LABEL, MARKER, MODELS, OUT, cell, set_style

LINES = MODELS + ["Perfect"]
COLOR["Perfect"], MARKER["Perfect"] = "black", None
METRICS = {"recall": (r"Termination Recall (%) $\uparrow$", "endings", ""),             # 이름: (y축 글자, 나누는 열, 파일 이름 끝)
           "precision": (r"Termination Precision (%) $\uparrow$", "picks", "_precision")}
XLABEL = "Review budget (% of candidate facts per landmark)"                # 후보 = 그 달(landmark)에 살아 있는 사실


def summary(dataset, metric):
    """(모델, 기간, 예산) -> 지표(%) 의 시드 평균(mean)과 표준편차(sd)."""
    t = pd.read_csv(K.C.HERE / f"curve_{dataset}.csv")
    g = (100 * t.found / t[METRICS[metric][1]]).groupby([t.model, t.horizon, t.budget])
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1)}).fillna(0.0)


def horizons(s):
    return sorted(s.index.get_level_values("horizon").unique())


def horizon_name(dataset, h):
    return f"{h} year{'s' * (h > 1)}" if dataset == "yago" else f"{h} months"


def draw(ax, s, h, metric):
    start = [0] if metric == "recall" else []                      # 0% 를 보면 Recall 은 0, Precision 은 정해지지 않는다
    for model in [m for m in MODELS if m in s.index.get_level_values(0)]:      # Perfect 는 그림에 안 그린다 (표에만 있다)
        part = s.loc[(model, h)]
        x, y, sd = start + K.BUDGETS, [0] * len(start) + list(part["mean"]), [0] * len(start) + list(part["sd"])
        ax.fill_between(x, [a - b for a, b in zip(y, sd)], [a + b for a, b in zip(y, sd)], color=COLOR[model], alpha=0.15, lw=0)
        ax.plot(x, y, color=COLOR[model], marker=MARKER[model], ls="--" if model == "Random" else "-", lw=3 if model == "LaST" else 2,
                ms=8, mfc=COLOR[model] if model == "LaST" else "white", zorder=5 if model == "LaST" else 3, label=LABEL.get(model, model))
    ax.set_xticks(K.BUDGETS if metric == "precision" else [b for b in K.BUDGETS if b > 1])   # Recall 은 1% 글자가 0 과 붙어서 뺀다
    ax.grid(True)
    ax.set_axisbelow(True)


def make_figure(datasets, metric):
    ylabel, _, tag = METRICS[metric]
    set_style()
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 22, "axes.titlesize": 24, "xtick.labelsize": 18, "ytick.labelsize": 18})   # 글씨 크기
    fig, axes = plt.subplots(3, len(datasets), figsize=(4.6 * len(datasets), 11), squeeze=False, sharex=True)
    for col, d in enumerate(datasets):
        s = summary(d, metric)
        axes[0, col].set_title(DATASETS[d], pad=10)
        for row, h in enumerate(horizons(s)):
            draw(axes[row, col], s, h, metric)
            axes[row, col].text(0.96, 0.95 if metric == "precision" else 0.05, horizon_name(d, h), transform=axes[row, col].transAxes,
                                ha="right", va="top" if metric == "precision" else "bottom")      # 선이 없는 구석
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))                   # 가장자리를 비워 두고 공통 축 이름을 한 번만 놓는다
    fig.supxlabel(XLABEL, y=0.0, fontsize=22)
    fig.supylabel(ylabel, x=0.0, fontsize=22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=22)           # 글씨가 커져서 한 줄에 5개 (2줄)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_budget_curve{tag}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_longest_figure(datasets, metric):
    """데이터셋마다 가장 긴 기간 하나만, 가로 1 x 4.  선 스타일은 make_figure 와 같다 (draw 를 같이 쓴다)."""
    ylabel, _, tag = METRICS[metric]
    set_style()
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 20, "axes.titlesize": 22, "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 4.6), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        s = summary(d, metric)
        h = horizons(s)[-1]                                                          # 가장 긴 기간
        draw(ax, s, h, metric)
        ax.set_title(f"{DATASETS[d]} ({horizon_name(d, h)})", pad=8)
    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    fig.supxlabel(XLABEL, y=0.0, fontsize=20)
    fig.supylabel(ylabel, x=0.0, fontsize=20)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=18)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_budget_curve{tag}_longest.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_table(datasets, metric):
    n = len(K.BUDGETS)
    lines = ["\\begin{tabular}{l|" + "c" * n + "}", "\\toprule", "Model & " + " & ".join(f"{q}\\%" for q in K.BUDGETS) + r" \\"]
    for d in datasets:
        s = summary(d, metric)
        learned = [m for m in MODELS if m != "Random" and m in s.index.get_level_values(0)]
        for h in horizons(s):
            lines += ["\\midrule", f"\\multicolumn{{{n + 1}}}{{l}}{{\\textbf{{{DATASETS[d]}}} \\textit{{({horizon_name(d, h)})}}}} \\\\"]
            for model in [m for m in LINES if m in s.index.get_level_values(0)]:
                cells = [cell(s.loc[(model, h, q), "mean"], s.loc[(model, h, q), "sd"], 1,
                              model in learned and s.loc[(model, h, q), "mean"] == max(s.loc[(m, h, q), "mean"] for m in learned)) for q in K.BUDGETS]
                lines.append(f"\\quad {LABEL.get(model, model)} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


if __name__ == "__main__":
    datasets = list(DATASETS)
    OUT.mkdir(parents=True, exist_ok=True)
    for metric, (_, _, tag) in METRICS.items():
        (OUT / f"table_budget_curve{tag}.tex").write_text(make_table(datasets, metric) + "\n")
        make_figure(datasets, metric)
        make_longest_figure(datasets, metric)
    print("저장:", OUT)
