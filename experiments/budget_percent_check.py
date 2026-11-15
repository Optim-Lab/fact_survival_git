"""점검: 예산을 "개수 k" 대신 "원점 사실의 q%"로 바꾸면 결론(모델 순위)이 달라지는가?  + 동점 처리가 결과를 흔드는가?

    python budget_percent_check.py finwiki yago finsurvkg icewswiki

모델 목록은 아래 MODELS (job 1074152 와 같음), 읽기 함수(find_file, load, signature)는 make_budget_selection.py 것을 쓴다. 원래 스크립트는 바꾸지 않는다.

예산 두 종류 (원점마다 따로 고르고, 모든 원점의 맞힘 수 / 고른 수를 더한다 = 원래 스크립트와 같음)

  k   : 원점마다 k개                         (원래 실험, 25 ... 1000)
  pct : 원점마다 ceil(q% x 그 원점의 사실 수)개  (q = 0.5, 1, 2, 5, 10)
맞힘 수 두 가지
  found_file : 동점이면 파일에 먼저 적힌 사실을 고른다 (원래 스크립트와 같음)
  found_tie  : 동점끼리는 무작위로 고른다고 보고 그 기댓값을 쓴다 (경계에 걸친 동점 묶음에서 남은 자리 x 그 묶음의 종료 비율)
              -> 둘의 차이가 크면 원래 결과가 파일 순서에 기대고 있다는 뜻
출력: results/analysis/percent_<dataset>.csv (모델 x 시드 x 예산 한 줄), 화면에 요약
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "results" / "analysis"       # 중간 결과 CSV 가 저장되는 곳
HERE.mkdir(parents=True, exist_ok=True)
import make_budget_selection as B  # noqa: E402

# 모델 10개 목록 (2026-11-12 예산 실행 job 1074152 에 쓴 것). make_budget_selection.py 는 2026-11-13 01:13 에 모델 4개 판으로 바뀌어 있어서
# 여기에 따로 적는다. 읽기 함수(find_file, load, signature)만 그 파일에서 가져온다.
R = "original_encoder_suite_20261023/"

def _std(d, proposed, degree, enc):
    return {"Proposed": ("p_neural", proposed), "DeepSurv": ("p_deepsurv", f"{d}_featladder_test_deepsurv_feat-age_seed{{s}}_20261101/test_predictions.csv.gz"),
            "LogCL + head": ("p_logcl", enc.format(m="logcl")), "Object degree": ("p_object_degree", degree),
            "LogisticHazard": ("p_logistic_hazard", f"{d}_featladder_test_logistic_hazard_feat-age_seed{{s}}_20261101/test_predictions.csv.gz"),
            "RE-GCN + head": ("p_regcn", enc.format(m="regcn")),
            "Cox": ("p_cox", f"classical_rolling_20261030/{d}/test_predictions.csv.gz"),
            "Discrete hazard": ("p_discrete_hazard", f"classical_rolling_20261030/{d}/test_predictions.csv.gz"),
            "Logistic": ("p_relation_age", degree if d != "finwiki" else proposed.replace("{s}", "42"))}

MODELS = {
    "finwiki": _std("finwiki", "finwiki_trial25_ruleB_test_s42_46_20261104/seed_{s}/attempt_*/test_predictions.csv.gz",
                    "finwiki_repeat_v2/regcn_official_encoder_seed42/test_predictions.csv.gz", "finwiki_{m}_seed{{s}}_1051295/test_predictions.csv.gz"),
    "finsurvkg": _std("finsurvkg", "finsurvkg_ruleB_test_proposed_seed{s}_20261101/test_predictions.csv.gz",
                      "finsurvkg_ruleB_test_proposed_seed42_20261101/test_predictions.csv.gz", R + "finsurvkg_{m}_seed{{s}}/test_predictions.csv.gz"),
    "icewswiki": _std("icewswiki", "icewswiki_ruleB_test_proposed_seed{s}_20261101/test_predictions.csv.gz",
                      "icewswiki_ruleB_test_proposed_seed42_20261101/test_predictions.csv.gz", R + "icewswiki_{m}_seed{{s}}_merged/test_predictions.csv.gz"),
    "yago": _std("yago", "yago_ruleB_test_proposed_seed{s}_20261101/test_predictions.csv.gz",
                 "yago_ruleB_test_proposed_seed42_20261101/test_predictions.csv.gz", R + "yago_{m}_seed{{s}}/test_predictions.csv.gz"),
}

MODELS["finwiki"]["RE-GCN + head"] = ("p_regcn", "finwiki_repeat_v2/regcn_official_encoder_seed{s}/test_predictions.csv.gz")   # FinWiki RE-GCN 은 다른 폴더

FEWER_SEEDS = {("icewswiki", "LogCL + head"): [42, 43], ("icewswiki", "RE-GCN + head"): [42, 44]}

NO_SEED = {"Object degree", "Cox", "Discrete hazard", "Logistic"}

PERCENTS = [0.5, 1, 2, 5, 10]

BUDGETS = [("k", k) for k in B.FIXED_K] + [("pct", q) for q in PERCENTS]


def take_of(kind, value, n):
    return min(n, value if kind == "k" else max(1, math.ceil(value * n / 100)))


def count(model, seed, rows, column):
    """예산마다 picks, found_file, found_tie 를 원점별로 더한다. 동점 비율도 잰다."""
    acc = {b: [0, 0.0, 0.0] for b in BUDGETS}
    tied_rows = 0
    for _, g in rows.groupby("origin", sort=False):
        ended = g["terminated"].to_numpy(dtype=float)
        n = len(ended)
        if model == "Random":
            for b in BUDGETS:
                t = take_of(*b, n)
                acc[b][0] += t
                acc[b][1] += t * ended.mean()
                acc[b][2] += t * ended.mean()
            continue

        score = g[column].to_numpy(dtype=float)
        order = np.argsort(score, kind="mergesort")                     # 생존확률 낮은 순, 동점이면 파일 순서
        s, e = score[order], ended[order]
        cum = np.concatenate([[0.0], np.cumsum(e)])                     # cum[i] = 앞 i개의 종료 수
        _, sizes = np.unique(s, return_counts=True)
        tied_rows += sizes[sizes > 1].sum()
        
        for b in BUDGETS:
            t = take_of(*b, n)
            lo, hi = np.searchsorted(s, s[t - 1], "left"), np.searchsorted(s, s[t - 1], "right")   # 경계 점수의 동점 묶음 [lo, hi)
            acc[b][0] += t
            acc[b][1] += cum[t]
            acc[b][2] += cum[lo] + (t - lo) * (cum[hi] - cum[lo]) / (hi - lo)
    
    rate = rows["terminated"].mean()
    
    return [dict(model=model, seed=seed, kind=kind, budget=value, picks=p, found_file=ff, found_tie=ft,
                 precision_file=100 * ff / p, precision_tie=100 * ft / p, lift_tie=(ft / p) / rate,
                 event_rate=100 * rate, tied_share=100 * tied_rows / len(rows))
            for (kind, value), (p, ff, ft) in acc.items()]


def run(dataset):
    table, signatures = [], {}
    for model, (column, pattern) in MODELS[dataset].items():
        seeds = [42] if model in NO_SEED else FEWER_SEEDS.get((dataset, model), B.SEEDS)
        files = {s: B.find_file(dataset, model, pattern, s) for s in seeds}
    
        if None in files.values():
            print(f"LEFT OUT: {model}")
    
            continue
    
        for seed, path in files.items():
            rows = B.load(path, column, B.HORIZON[dataset])
            signatures[(model, seed)] = B.signature(rows)
            table += count(model, seed, rows, column)
    
        print(f"  {dataset} {model} done", flush=True)
    
    assert len(set(signatures.values())) == 1, "모델마다 보는 행이 다르다"
    table += count("Random", 42, rows, column)
    table = pd.DataFrame(table).replace({"model": {"Proposed": "LaST"}})
    table.to_csv(HERE / f"percent_{dataset}.csv", index=False)
    return table


def report(dataset, t):
    m = t.groupby(["model", "kind", "budget"])[["precision_file", "precision_tie", "lift_tie", "tied_share"]].mean().reset_index()
    print(f"\n==================== {dataset}  (종료 비율 {t.event_rate.iloc[0]:.2f}%) ====================")
    
    for kind in ("k", "pct"):
        p = m[m.kind == kind].pivot(index="model", columns="budget", values="precision_tie")
        p = p.sort_values(p.columns[min(2, len(p.columns) - 1)], ascending=False)
        print(f"\n[{kind}] precision (%), 동점 = 무작위 기댓값.  순위는 {p.columns[min(2, len(p.columns) - 1)]} 기준")
        print(p.round(2).to_string())
        print(f"[{kind}] 1등 모델:", ", ".join(f"{b}->{p[b].idxmax()}" for b in p.columns))
    
    lift = m[m.kind == "pct"].pivot(index="model", columns="budget", values="lift_tie")
    
    print("\n[pct] lift (무작위 대비 몇 배)\n" + lift.round(2).to_string())
    
    gap = (m.precision_file - m.precision_tie).abs()
    worst = m.loc[gap.idxmax()]
    print(f"\n동점 점검: |파일순서 - 무작위기댓값| 최대 {gap.max():.2f}%p  ({worst.model}, {worst.kind}={worst.budget})")
    print("모델별 동점 행 비율(%):", m.groupby("model").tied_share.first().round(1).to_dict())


if __name__ == "__main__":
    for d in [a for a in sys.argv[1:] if a in MODELS] or list(MODELS):
        report(d, run(d))
