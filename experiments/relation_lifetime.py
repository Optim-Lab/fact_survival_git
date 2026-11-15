"""관계별 수명 분포 (Kaplan-Meier).   python relation_lifetime.py          (login 노드에서 바로 돈다, 몇 초)

수명 = 끝난 시점 - 시작 시점.  관측이 끝날 때까지 안 끝난 사실은 "수명이 최소 이만큼"이라서 검열(censored)로 두고 KM 으로 계산한다.
쓰는 사실: 시작이 "자료의 첫 달"이 아닌 사실.  (첫 달에 시작한 사실은 그 전에 이미 시작했을 수 있어서 시작 시점이 잘려 있다 -> 뺀다.)
관계 = relation_choice_<데이터셋>.csv 의 순위 목록(끝난 건수 순)에서 같은 간격으로 TOP 개 (끝난 사실이 MIN_ENDED 미만이거나 EXCLUDE 인 관계는 먼저 뺀다) + 전체 사실.
  (관계 안 위험군 분석은 순위 1~3 을 쓰므로 이 그림의 관계와 다르다.)

결과: lifetime_quantiles.csv (관계마다 25%, 50%, 75% 가 끝난 시점, 년),  lifetime_km_<데이터셋>.csv (그림용 곡선),  fig_relation_lifetime.*
"""
#%%
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

import budget_curve as K
from make_budget_figures import DATASETS, OUT, set_style

ROOT = Path(__file__).resolve().parents[1]                    # 저장소 루트 (data/ 와 fact_survival/ 가 있다)
sys.path.insert(0, str(ROOT))

from fact_survival.data import read_facts  # noqa: E402

FILES = {"finwiki": ("data/FinWiki/train_valid_test.pkl", "month"), "icewswiki": ("data/raw/ICEWSWiki.pkl", "month"),
         "finsurvkg": ("data/FinSurvKG/train_valid_test.pkl", "month"), "yago": ("data/YAGO/train_valid_test.pkl", "year")}

TOP = 5                                # 그림에 그릴 관계 수
MIN_ENDED = 20                           # 끝난 사실이 이 수 이상인 관계만 (FinWiki 에서 후보가 5개 되는 최대값)

EXCLUDE = {"yago": ["nationality"]}      # nationality: 사실의 92.5% 가 시작 연도 1946 (기본값으로 보임) -> 수명이 종료 연도 - 1946 이 되어 의미가 없다
COLOR = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B"]     # 회색은 "All facts" 점선이 쓴다
#%%

def kaplan_meier(duration, ended):
    """(시간, 그 시간까지 살아 있을 비율).  duration = 수명, ended = 끝났으면 True (False 는 검열)."""
    duration, ended = np.asarray(duration, float), np.asarray(ended, bool)
    times = np.unique(duration[ended])                                  # 끝난 일이 있었던 시간들
    at_risk = np.array([(duration >= t).sum() for t in times])          # 그 시간 직전까지 살아 있던 수
    died = np.array([((duration == t) & ended).sum() for t in times])
    return times, np.cumprod(1 - died / at_risk)


def time_when(times, survival, share):
    """생존 비율이 1 - share 이하로 처음 내려가는 시간 (share=0.5 이면 중앙 수명).  끝까지 안 내려가면 NaN."""
    below = np.where(survival <= 1 - share)[0]
    return times[below[0]] if len(below) else np.nan


def pick(ranked, facts, n):
    """끝난 사실이 MIN_ENDED 이상인 관계만 남기고, 순위 목록(끝난 건수 순)에서 같은 간격으로 n 개 (맨 위 ~ 맨 아래를 고르게)."""
    ok = [r for r in ranked if facts[facts.relation == r].ended.sum() >= MIN_ENDED]
    return [ok[i] for i in np.linspace(0, len(ok) - 1, min(n, len(ok))).round().astype(int)]


def run():
    quantiles, curves = [], {}
    for d, (path, unit) in FILES.items():
        facts, info = read_facts(ROOT / path, unit)
        per_year = 12 if unit == "month" else 1 #년으로 바꾸기 위해 나눌수를 정함. 1년이 12칸이라서.
        facts = facts[facts.begin > facts.begin.min()].assign(years=lambda t: (t.stop - t.begin) / per_year, ended=lambda t: t.stop < info["study_end"])
    
        ranked = [r for r in pd.read_csv(K.C.HERE / f"relation_choice_{d}.csv").relation if r not in EXCLUDE.get(d, [])]
        chosen = pick(ranked, facts, TOP)                                            # 뺄 관계를 거른 순위 목록에서 같은 간격으로 TOP 개
        curves[d] = []
    
        for name, part in [("All facts", facts)] + [(r, facts[facts.relation == r]) for r in chosen]:
    
            t, s = kaplan_meier(part.years, part.ended)
    
            curves[d].append((name, t, s))
    
            quantiles.append(dict(dataset=d, relation=name, facts=len(part), ended=int(part.ended.sum()), censored_percent=100 * (1 - part.ended.mean()),
    
                                  q25_years=time_when(t, s, .25), median_years=time_when(t, s, .5), q75_years=time_when(t, s, .75))) #q25:사실의 25%가 끝나는시점, 생존율이 75%이하로 처음내려가는 때
            pd.DataFrame({"years": t, "survival": s}).assign(relation=name).to_csv(K.C.HERE / f"lifetime_km_{d}_{len(curves[d]) - 1}.csv", index=False)
    
    table = pd.DataFrame(quantiles)
    table.to_csv(K.C.HERE / "lifetime_quantiles.csv", index=False)
    print(table.round(2).to_string(index=False))
    return curves


def figure(curves):
    set_style()
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 18, "axes.titlesize": 20})
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.8 * len(DATASETS), 4.4), squeeze=False)
    for ax, d in zip(axes[0], DATASETS):
        for i, (name, t, s) in enumerate(curves[d]):
            kw = dict(color="gray", ls="--", lw=1.8) if i == 0 else dict(color=COLOR[i - 1], lw=2.6)
            ax.step(np.r_[1 / 12 if d != "yago" else 1, t], 100 * np.r_[1, s], where="post", label=name, **kw)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))      # 0.1, 1, 10, 100 으로 읽기 쉽게
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(DATASETS[d], pad=8)
        ax.set_ylim(0, 102)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout(rect=(0.02, 0.02, 1, 1))
    fig.supxlabel("Fact age (years)", y=0.0, fontsize=18)
    fig.supylabel("Facts still valid (%)", x=0.0, fontsize=18)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"fig_relation_lifetime.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure(run())
