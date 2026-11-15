"""Table and figures of the budget experiment, from the CSV files of make_budget_selection.py (same look as make_ablation_study.py).

    python make_budget_figures.py                         all four datasets (finwiki yago finsurvkg icewswiki)
    python make_budget_figures.py finwiki finsurvkg       only these, in this order

Input   RES/budget_selection_<dataset>/budget_curve_absolute.csv    one row per (model, seed, budget k):
            found = correct picks, precision = 100 * found / picks;  k = facts checked at EVERY origin
Output  OUT/table_budget_<datasets>.tex                  LaTeX table (layout of tab:downstream_finwiki)
        OUT/fig_budget_<datasets>.png/pdf/svg            top row Precision@k, bottom row #Detected; one column per dataset
        OUT/fig_budget_<dataset>_<precision|detected>.*  one panel each

[1] Nothing is recomputed: table and figure show the same mean (and SD over seeds, ddof = 1) of the CSV.
[2] Random is an expected value and Object degree one deterministic run: no SD. Bold = best LEARNED model of the column.
[3] Shaded band = mean +- SD over seeds.
"""
#%%
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "figures"                      # 그림과 LaTeX 표가 저장되는 곳
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = {"finwiki": "FinWiki", "yago": "YAGO", "finsurvkg": "FinSurvKG", "icewswiki": "ICEWSWiki"}   # file name -> figure title
TABLE_K = [50, 100, 250]
MODELS = ["Random", "Logistic", "Object degree", "Cox", "Discrete hazard", "DeepSurv", "LogisticHazard",
          "RE-GCN + head", "LogCL + head", "LaST"]                                           # order of legend and table rows
LABEL = {"Object degree": "LR (Object degree)", "Logistic": "LR (Relation+age)", "LogCL + head": "LogCL+head", "RE-GCN + head": "RE-GCN+head"}   # figure legend (and budget_curve tables)
TABLE_LABEL = {"LogCL + head": "LogCL +Survival head", "RE-GCN + head": "RE-GCN +Survival head"}                                       # LaTeX table (small caps: "LaST": r"\textsc{LaST}")
COLOR = dict(zip(MODELS, ["#9D9D9D", "#9D755D", "#4C78A8", "#72B7B2", "#B279A2",            # palette of make_ablation_study.py, extended
                          "#F58518", "#EECA3B", "#2F4B7C", "#54A24B", "#E45756"]))
MARKER = dict(zip(MODELS, ["o", "v", "s", "<", ">", "^", "h", "X", "D", "P"]))
METRICS = [("precision", r"Precision@k (%) $\uparrow$", "precision"), ("found", r"# detected $\uparrow$", "detected")]   # (CSV column, y label: arrow = better direction, file tag)
FONT = 22
XLABEL = "k (facts per origin)"                                                           # k = facts checked at every origin


#%%
def summary(dataset):
    """(model, k) -> mean and SD over seeds of found / precision."""
    curve = pd.read_csv(RES / f"budget_selection_{dataset}" / "budget_curve_absolute.csv").replace({"model": {"Proposed": "LaST"}})
    g = curve.groupby(["model", "budget"])[["found", "precision"]]
    return g.mean().join(g.std(ddof=1).add_suffix("_sd")).fillna(0.0)


def present(s):
    return [m for m in MODELS if m in s.index.get_level_values(0)]


#%% Table
def cell(mean, sd, decimals, bold):
    number = f"\\mathbf{{{mean:.{decimals}f}}}" if bold else f"{mean:.{decimals}f}"
    return f"${number}_{{\\pm {sd:.{decimals}f}}}$" if sd > 0 else f"${number}$"


def make_table(datasets):
    n = len(TABLE_K)
    lines = ["\\begin{tabular}{l|" + "|".join(["cc"] * n) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c{'|' if i < n - 1 else ''}}}{{$k={k}$}}" for i, k in enumerate(TABLE_K)) + r" \\",
             "Model & " + " & ".join(["\\#Detected$\\uparrow$ & Precision$\\uparrow$"] * n) + r" \\", "\\midrule"]
    for dataset in datasets:
        s = summary(dataset)
        if len(datasets) > 1:
            lines.append(f"\\multicolumn{{{1 + 2 * n}}}{{l}}{{\\textit{{{dataset}}}}} \\\\")
        learned = [m for m in present(s) if m != "Random"]
        for model in present(s):
            cells = []
            for k in TABLE_K:
                for col, decimals in (("found", 0), ("precision", 2)):
                    best = model != "Random" and s.loc[(model, k), col] == max(s.loc[(m, k), col] for m in learned)
                    cells.append(cell(s.loc[(model, k), col], s.loc[(model, k), col + "_sd"], decimals, best))
            lines.append(f"\\quad {TABLE_LABEL.get(model, model)} & " + " & ".join(cells) + r" \\")
        lines.append("\\midrule" if dataset != datasets[-1] else "\\bottomrule")
    return "\n".join(lines + ["\\end{tabular}"])


#%% Figures
def set_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.facecolor": "#EDF0F5", "axes.edgecolor": "#A9B0BA", "axes.spines.top": False,
                         "axes.spines.right": False, "grid.color": "white", "grid.linewidth": 1.15})


def draw_panel(ax, s, metric, ylabel):
    """One line per model (mean over seeds) with its +- SD band [3]; k on a log axis. LaST: thick line, filled marker."""
    for model in present(s):
        part, last = s.loc[model], model == "LaST"
        ax.fill_between(part.index, part[metric] - part[metric + "_sd"], part[metric] + part[metric + "_sd"], color=COLOR[model], alpha=0.15, lw=0)
        ax.plot(part.index, part[metric], color=COLOR[model], marker=MARKER[model], lw=3 if last else 2, ms=9 if last else 7,
                mfc=COLOR[model] if last else "white", mew=2 if last else 1, zorder=5 if last else 3, label=LABEL.get(model, model))
    ax.set_xscale("log")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)


def save(fig, handles, labels, stem, legend_size, ncol=5):           # 10 models -> 2 legend rows
    """Legend in one row above the panels; PNG, PDF and SVG."""
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False, fontsize=legend_size)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figure(datasets):
    """Columns = datasets; top row Precision@k, bottom row #Detected. Title on the top row, x label on the bottom row only."""
    set_style()
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 9), squeeze=False)
    for col, dataset in enumerate(datasets):
        s = summary(dataset)
        for row, (metric, ylabel, _) in enumerate(METRICS):
            draw_panel(axes[row, col], s, metric, ylabel)
        axes[0, col].set_title(DATASETS[dataset], pad=10)
        axes[1, col].set_xlabel(XLABEL)
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    legend = {}                                                    # the datasets do not all have the same models
    for ax in axes[0]:
        legend.update(zip(*reversed(ax.get_legend_handles_labels())))
    labels = [LABEL.get(m, m) for m in MODELS if LABEL.get(m, m) in legend]
    save(fig, [legend[l] for l in labels], labels, f"fig_budget_{'_'.join(datasets)}", FONT)


def make_single(dataset, metric, ylabel, tag):
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    draw_panel(ax, summary(dataset), metric, ylabel)
    ax.set_title(DATASETS[dataset], pad=10)
    ax.set_xlabel(XLABEL)
    fig.tight_layout()
    save(fig, *ax.get_legend_handles_labels(), f"fig_budget_{dataset}_{tag}", 14, ncol=3)


#%%
if __name__ == "__main__":
    datasets = [a for a in sys.argv[1:] if a in DATASETS] or list(DATASETS)      # Jupyter's own arguments are ignored
    missing = [d for d in datasets if not (RES / f"budget_selection_{d}" / "budget_curve_absolute.csv").exists()]
    for d in missing:
        print(f"SKIP {d}: no budget_curve_absolute.csv yet (run make_budget_selection.py {d} first)")
    datasets = [d for d in datasets if d not in missing]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"table_budget_{'_'.join(datasets)}.tex").write_text(make_table(datasets) + "\n")
    make_figure(datasets)
    for dataset in datasets:
        for metric, ylabel, tag in METRICS:
            make_single(dataset, metric, ylabel, tag)
    print(f"written to {OUT}")
