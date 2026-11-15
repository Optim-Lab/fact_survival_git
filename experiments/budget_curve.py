"""예산 곡선: 매달(원점마다) 생존확률이 낮은 순으로 q%만 봤을 때, 실제로 끝난 사실을 몇 % 찾는가?

    python budget_curve.py finwiki yago finsurvkg icewswiki

한 줄 = (모델, 시드, 기간, 예산):  found = 찾은 종료 수,  endings = 전체 종료 수,  picks = 본 사실 수
Random  = 무작위로 골랐을 때의 평균 (계산값)        Perfect = 끝날 사실만 골랐을 때의 최대값
점수가 같은 사실은 무작위로 골랐다고 보고 평균을 쓴다.
읽는 파일은 budget_percent_check.py 의 MODELS (예산 실험과 같은 파일). 결과: curve_<데이터셋>.csv
"""
import math
import sys

import numpy as np
import pandas as pd

import budget_percent_check as C

B = C.B
BUDGETS = [1, 5, 10, 15, 25]            # 본 사실 비율 (%)


def read(path, column):
    rows = pd.read_csv(path, keep_default_na=False, usecols=["origin", "horizon", "survived", column])
    return rows.assign(ended=1 - rows.survived.astype(int))


def found_low_first(score, ended, take):
    """점수가 낮은 순으로 take개를 봤을 때 찾은 종료 수. 경계의 동점 묶음은 남은 자리만큼 평균."""
    order = np.argsort(score, kind="mergesort")
    s, total = score[order], np.concatenate([[0.0], np.cumsum(ended[order])])
    lo, hi = np.searchsorted(s, s[take - 1], "left"), np.searchsorted(s, s[take - 1], "right")
    return total[lo] + (take - lo) * (total[hi] - total[lo]) / (hi - lo)


def count(rows, column):
    """기간·예산마다: 모델이 찾은 수, 완벽한 수, 무작위 평균, 전체 종료 수, 본 수 (모든 원점을 더한다)."""
    total = {}
    for (horizon, _), g in rows.groupby(["horizon", "origin"]):
        score, ended = g[column].to_numpy(float), g.ended.to_numpy(float)
        for q in BUDGETS:
            take = max(1, math.ceil(q * len(g) / 100))
            row = [found_low_first(score, ended, take), min(take, ended.sum()), take * ended.mean(), ended.sum(), take]
            total[(horizon, q)] = np.add(total.get((horizon, q), 0), row)
    return pd.DataFrame([(h, q, *v) for (h, q), v in total.items()], columns=["horizon", "budget", "found", "perfect", "random", "endings", "picks"])


def run(dataset):
    parts = []
    for model, (column, pattern) in C.MODELS[dataset].items():
        seeds = [42] if model in C.NO_SEED else C.FEWER_SEEDS.get((dataset, model), B.SEEDS)
        for seed in seeds:
            t = count(read(B.find_file(dataset, model, pattern, seed), column), column)
            parts.append(t.assign(model=model, seed=seed)[["model", "seed", "horizon", "budget", "found", "endings", "picks"]])
        print(f"  {dataset} {model} done", flush=True)
    for name in ("perfect", "random"):                       # 정답만 쓰므로 어느 모델에서 계산해도 같다
        parts.append(t.assign(model=name.capitalize(), seed=42, found=t[name])[parts[0].columns])
    table = pd.concat(parts).replace({"model": {"Proposed": "LaST"}})
    assert table.groupby(["horizon", "budget"]).endings.nunique().max() == 1, "모델마다 보는 행이 다르다"
    table.to_csv(C.HERE / f"curve_{dataset}.csv", index=False)


if __name__ == "__main__":
    for d in [a for a in sys.argv[1:] if a in C.MODELS] or list(C.MODELS):
        run(d)
