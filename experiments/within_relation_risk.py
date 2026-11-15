"""같은 관계 안에서도 예측 위험이 높은 사실이 더 많이 끝나는가?   python within_relation_risk.py finwiki yago finsurvkg icewswiki

1) 관계 고르기 (결과를 보기 전에 정한 규칙):  서로 다른 사실이 200개 이상, 가장 긴 기간에서 끝난 행이 200개 이상인 관계를 모두 모아
   "끝난 행 수" 순으로 번호(rank)를 붙인다.  본문은 rank 1~3 (상위 3개), 부록은 전부.
2) 관계마다, 매 원점에서 그 관계의 사실만 가장 긴 기간의 예측 생존확률 순위로 3군으로 나눈다 (0 = 위험 높음, 2 = 위험 낮음).
3) 군마다 기간별 "예측 생존확률 평균(pred)"과 "실제 생존 비율(obs)"을 구한다.  세 기간이 모두 관측된 행만 쓴다 (risk_group_curves.py 와 같은 방식).
동점 처리: 점수가 같은 사실 묶음이 3등분 선에 걸치면, 묶음을 걸친 비율대로 쪼개서 군에 나눠 넣는다 (가중치).
  (이전에는 묶음 전체를 한 군에 넣었다.  YAGO 처럼 나이가 해마다 정수라 동점이 많은 경우 군이 엉뚱하게 갈라졌다.)

결과: relation_choice_<데이터셋>.csv (고른 관계와 rank),  within_relation_<데이터셋>.csv (모델, 시드, 관계, 군, 기간: pred, obs, rows)
"""
import sys

import pandas as pd

import budget_curve as K

MIN_FACTS, MIN_ENDINGS, GROUPS = 200, 200, 3
USE = ["fact_id", "relation", "origin", "horizon", "survived"]


def choose_relations(dataset):
    """LaST 시드 42 파일에서 가장 긴 기간의 끝난 행 수가 많은 순서로, 조건(사실 200개, 끝난 행 200개 이상)을 넘는 관계 전부."""
    column, pattern = K.C.MODELS[dataset]["Proposed"]
    d = pd.read_csv(K.B.find_file(dataset, "Proposed", pattern, 42), keep_default_na=False, dtype={"fact_id": str}, usecols=USE)
    d = d[d.horizon == d.horizon.max()].assign(ended=lambda t: 1 - t.survived.astype(int))
    t = d.groupby("relation").agg(rows=("ended", "size"), facts=("fact_id", "nunique"), endings=("ended", "sum"))
    t["ended_percent"] = 100 * t.endings / t.rows
    t = t[(t.facts >= MIN_FACTS) & (t.endings >= MIN_ENDINGS)].sort_values("endings", ascending=False)
    t.reset_index().assign(rank=range(1, len(t) + 1), horizon=d.horizon.max()).to_csv(K.C.HERE / f"relation_choice_{dataset}.csv", index=False)
    return list(t.index)

#관계안에서3군으로나누기
def one_run(path, column, relations):
    d = pd.read_csv(path, keep_default_na=False, dtype={"fact_id": str}, usecols=USE + [column])
    d = d[d.relation.isin(relations)].set_index(["relation", "origin", "fact_id", "horizon"])
    assert d.index.is_unique, "같은 (관계, 원점, 사실, 기간) 이 두 번 있다"
    p, y = d[column].unstack("horizon").dropna(), d["survived"].astype(float).unstack("horizon").dropna()
    y = y.loc[p.index]                                                          # 세 기간 모두 관측된 행만
    key = ["relation", "origin"]
    top = p[p.columns.max()].groupby(key)
    size = top.transform("size")
    low, high = (top.rank(method=m) for m in ("min", "max"))             # 동점 묶음이 차지하는 순위 범위 (낮은 쪽, 높은 쪽)
    low, high = (low - 1) / size, high / size                            # 0~1 사이 위치로 바꾼다
    out = []

    for g in range(GROUPS):                                              # 묶음이 g 번째 군 구간과 겹치는 비율 = 그 군에 들어가는 가중치
        w = ((high.clip(upper=(g + 1) / GROUPS) - low.clip(lower=g / GROUPS)).clip(lower=0) / (high - low)).rename("w")
        for h in p.columns:
            t = pd.DataFrame({"relation": p.index.get_level_values("relation"), "w": w.to_numpy(), "pred": p[h].to_numpy(), "obs": y[h].to_numpy()})
            t = t[t.w > 0].assign(pw=lambda x: x.pred * x.w, ow=lambda x: x.obs * x.w).groupby("relation")[["w", "pw", "ow"]].sum()
            out.append(pd.DataFrame({"relation": t.index, "group": g, "pred": (t.pw / t.w).to_numpy(), "obs": (t.ow / t.w).to_numpy(), "rows": t.w.to_numpy(), "horizon": h}))
    return pd.concat(out)


def run(dataset):
    relations = choose_relations(dataset)
    print(dataset, "고른 관계:", relations, flush=True)
    parts = []
    for model, (column, pattern) in K.C.MODELS[dataset].items(): #데이터셋의 모델목록을 하나씩 꺼냄. #ites는 이름,값쌍을줌. 
        for seed in [42] if model in K.C.NO_SEED else K.C.FEWER_SEEDS.get((dataset, model), K.B.SEEDS):
            parts.append(one_run(K.B.find_file(dataset, model, pattern, seed), column, relations).assign(model=model, seed=seed))
        print(f"  {dataset} {model} done", flush=True)
    pd.concat(parts).replace({"model": {"Proposed": "LaST"}}).to_csv(K.C.HERE / f"within_relation_{dataset}.csv", index=False)


if __name__ == "__main__":
    for d in [a for a in sys.argv[1:] if a in K.C.MODELS] or list(K.C.MODELS):
        run(d)
#pred:군안사실들의 예측 생존확률의가중평균
#obs:군안사실들의 실제 생존비율의가중평균=실제생존비율
