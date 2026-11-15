"""위험군별 생존: 예측 위험이 높은 사실이 실제로 더 많이 종료되는가?  (Table 1 비교 모델 전부, KM 은 사실마다 같은 값이라 위험군을 못 나눠서 뺀다)

    python risk_group_curves.py finwiki yago finsurvkg icewswiki     # risk_group_<데이터셋>.csv
    python risk_group_curves.py figure                               # 그림 + 표

매 landmark(원점)에서 가장 긴 기간의 예측 생존확률로 사실을 3군으로 나눈다:  0 = 위험 높음(생존확률 낮은 1/3), 1 = 중간, 2 = 위험 낮음.
세 기간을 모두 관측할 수 있는 (원점, 사실) 만 쓴다 (같은 사실들의 곡선이 되도록).  군마다 기간별로 "예측 생존확률 평균"과 "실제 생존 비율"을 구한다.
동점 처리: 점수가 같은 사실 묶음이 3등분 선에 걸치면, 걸친 비율대로 쪼개서 군에 나눠 넣는다 (가중치).  그래서 군 크기가 늘 같다.
  (이전에는 묶음 전체를 한 군에 넣어서 동점이 많은 단순 모델은 군 크기가 어긋났다.)  rows = 가중치 합.  결과 한 줄 = (모델, 시드, 군, 기간): pred, obs, rows.
"""
import sys

import pandas as pd

import budget_curve as K
#K:어떤 모델의 예측파일을 어떤시드로 어디서 읽을지 정함. 
GROUPS = 3
NAME = ["High risk", "Medium", "Low risk"]


def one_run(path, column):
    d = pd.read_csv(path, keep_default_na=False, dtype={"fact_id": str}, usecols=["fact_id", "origin", "horizon", "survived", column])

    d = d.set_index(["origin", "fact_id", "horizon"]) #세열을 행이름으로 만ㄷ름. 
    
    assert d.index.is_unique, "같은 (원점, 사실, 기간) 이 두 번 있다" # 같은 사실이 두번나오지않게끔 중복처리
    p, y = d[column].unstack("horizon").dropna(), d["survived"].astype(float).unstack("horizon").dropna() #unstack으로 기간 을 열로 펼치면 어떤 사실은 긴기간이 아직 관측되지않아 그 칸이 비어있음.그러한 칸을 dropna는 행을 지움. 
    
    y = y.loc[p.index]                                                  # 세 기간 모두 관측된 행만
    top = p[p.columns.max()].groupby("origin")                          # 가장 긴 기간의 예측 생존확률, 원점 안에서 비교
    size = top.transform("size")
    low, high = (top.rank(method=m) for m in ("min", "max"))             # 동점 묶음이 차지하는 순위 범위 (낮은 쪽, 높은 쪽)
    low, high = (low - 1) / size, high / size                            # 0~1 사이 위치로 바꾼다
    out = []
    for g in range(GROUPS):                                              # 묶음이 g 번째 군 구간과 겹치는 비율 = 그 군에 들어가는 가중치
        w = ((high.clip(upper=(g + 1) / GROUPS) - low.clip(lower=g / GROUPS)).clip(lower=0) / (high - low)).to_numpy()
        for h in p.columns:
            keep = w > 0 #가중치가 0보다 큰 행만 고르는 참/거짓 배열임. 
            weight, pred, obs = w[keep], p[h].to_numpy()[keep], y[h].to_numpy()[keep] #가중치가 0보다 큰 행만 고른다. 기증<tool_call>군
            out.append(dict(group=g, pred=(pred * weight).sum() / weight.sum(), obs=(obs * weight).sum() / weight.sum(), rows=weight.sum(), horizon=h))
    return pd.DataFrame(out)


def run(dataset):
    parts = []
    for model, (column, pattern) in K.C.MODELS[dataset].items():
        for seed in [42] if model in K.C.NO_SEED else K.C.FEWER_SEEDS.get((dataset, model), K.B.SEEDS):
            parts.append(one_run(K.B.find_file(dataset, model, pattern, seed), column).assign(model=model, seed=seed))
        print(f"  {dataset} {model} done", flush=True)
    pd.concat(parts).replace({"model": {"Proposed": "LaST"}}).to_csv(K.C.HERE / f"risk_group_{dataset}.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1:] == ["figure"]:
        import risk_group_figure
        risk_group_figure.main()
    else:
        for d in [a for a in sys.argv[1:] if a in K.C.MODELS] or list(K.C.MODELS):
            run(d)
