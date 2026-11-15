"""예산 실험: "사람이 사실 k개만 확인할 수 있다면, 곧 종료될 사실을 몇 개나 찾는가?"

    python make_budget_selection.py finsurvkg          (finwiki | finsurvkg | icewswiki | yago)

방법 (롤링 origin 하나마다 따로)
  1. 후보 사실을 예측 생존확률 S(12개월)이 낮은 순서로 줄 세운다.            [1]
  2. 앞에서부터 k개를 고른다.
  3. 고른 사실이 12개월 안에 실제로 종료됐으면 "맞힘"이다 (survived == 0).
  모든 origin을 합쳐서:  found = 맞힌 수 (#Detected),  precision = found / 고른 수 x 100,  lift = precision / 종료 비율 (1 = 무작위)

입력: 모델·시드마다 test_predictions.csv.gz 한 개 (열: fact_id, origin, horizon, survived, 그리고 모델의 점수 열)
출력: RES/budget_selection_<데이터셋>/budget_curve_absolute.csv  (모델 x 시드 x k 마다 한 줄)  -> make_budget_figures.py가 읽는다
바꿀 것은 모두 SETTINGS 블록에 있다.

각주
  [1] 같은 점수가 많아도(DeepSurv, 차수 규칙) 결과가 흔들리지 않게 "안정 정렬"(mergesort)을 쓴다: 동점이면 파일 순서를 지킨다.
  [2] origin마다 따로 처리한 뒤 found와 고른 수를 더한다. origin끼리 평균을 내지 않는다 (논문 표와 같은 방식).
  [3] Random은 뽑지 않고 기댓값을 쓴다: 후보 n개 중 e개가 종료됐다면 k개를 무작위로 골랐을 때 평균 k*e/n개를 찾는다.
  [4] 모든 모델이 정확히 같은 (사실, origin, 정답) 행을 보고 있는지 확인한다. 다르면 비교가 공정하지 않으므로 멈춘다.
  [5] 파일이 없는 모델은 "LEFT OUT"으로 출력하고 뺀다. 다른 파일이나 다른 시드로 대신하지 않는다.
  모델: Proposed = 반감기 혼합 신경망 (p_neural) / DeepSurv = 나이만 쓰는 Cox 신경망 (p_deepsurv) /
        LogCL + head = LogCL 인코더에 생존 head (p_logcl) / Object degree = 대상의 차수로 줄 세우는 규칙 (p_object_degree, 학습 없음, 1회)
"""
#%%
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- SETTINGS
RES = Path(os.environ.get("FACT_SURVIVAL_RESULTS", Path(__file__).resolve().parents[1] / "results"))   # 예측 파일(test_predictions.csv.gz)이 든 폴더들이 있는 곳.  MODELS 안의 폴더 이름은 저자가 돌린 실행 이름이다
SEEDS = [42, 43, 44, 45, 46]
HORIZON = {"finwiki": 12, "finsurvkg": 12, "icewswiki": 12, "yago": 1}     # "12개월 안에 종료"에 해당하는 horizon. YAGO는 단위가 년이라 1년 = 12개월
FIXED_K = [25, 50, 100, 250, 500, 1000]                        # origin마다 확인하는 사실 수

# 데이터셋 -> 모델 -> (점수 열, RES 아래 파일 경로).  {s} = 시드, * = 아무 글자.  "Object degree"는 학습이 없어서 시드 42 하나만 쓴다.
MODELS = {
    "finwiki": {
        "Proposed":      ("p_neural", "finwiki_trial25_ruleB_test_s42_46_20261104/seed_{s}/attempt_*/test_predictions.csv.gz"),
        "DeepSurv":      ("p_deepsurv", "finwiki_featladder_test_deepsurv_feat-age_seed{s}_20261101/test_predictions.csv.gz"),
        "LogCL + head":  ("p_logcl", "finwiki_logcl_seed{s}_1051295/test_predictions.csv.gz"),       # 시드 42만 경로가 달라서 아래 SPECIAL
        "Object degree": ("p_object_degree", "finwiki_repeat_v2/regcn_official_encoder_seed42/test_predictions.csv.gz"),
    },
    "finsurvkg": {
        "Proposed":      ("p_neural", "finsurvkg_ruleB_test_proposed_seed{s}_20261101/test_predictions.csv.gz"),
        "DeepSurv":      ("p_deepsurv", "finsurvkg_featladder_test_deepsurv_feat-age_seed{s}_20261101/test_predictions.csv.gz"),
        "LogCL + head":  ("p_logcl", "original_encoder_suite_20261023/finsurvkg_logcl_seed{s}/test_predictions.csv.gz"),
        "Object degree": ("p_object_degree", "finsurvkg_ruleB_test_proposed_seed42_20261101/test_predictions.csv.gz"),
    },
    "icewswiki": {                                             # 파일 하나가 855만 행: 계산 노드에서 돌릴 것
        "Proposed":      ("p_neural", "icewswiki_ruleB_test_proposed_seed{s}_20261101/test_predictions.csv.gz"),
        "DeepSurv":      ("p_deepsurv", "icewswiki_featladder_test_deepsurv_feat-age_seed{s}_20261101/test_predictions.csv.gz"),
        "LogCL + head":  ("p_logcl", "original_encoder_suite_20261023/icewswiki_logcl_seed{s}_merged/test_predictions.csv.gz"),
        "Object degree": ("p_object_degree", "icewswiki_ruleB_test_proposed_seed42_20261101/test_predictions.csv.gz"),
    },
    "yago": {                                                  # YAGO는 열린 사실이 없다: 논문에 밝힐 것
        "Proposed":      ("p_neural", "yago_ruleB_test_proposed_seed{s}_20261101/test_predictions.csv.gz"),      # 논문 Table 1의 YAGO 값과 같은 폴더
        "Object degree": ("p_object_degree", "yago_ruleB_test_proposed_seed42_20261101/test_predictions.csv.gz"),
        "DeepSurv":     ("p_deepsurv", "yago_featladder_test_deepsurv_feat-age_seed{s}_20261101/test_predictions.csv.gz"),
        "LogCL + head":  ("p_logcl", "original_encoder_suite_20261023/yago_logcl_seed{s}/test_predictions.csv.gz"),
    },
}
FEWER_SEEDS = {("icewswiki", "LogCL + head"): [42, 43]}        # 파일이 있는 시드만 쓰는 모델 (ICEWSWiki LogCL은 시드 2개: 논문에 n=2라고 밝힐 것)
SPECIAL = {("finwiki", "LogCL + head", 42): "original_encoder_suite_20261023/finwiki_logcl_seed42/test_predictions.csv.gz"}
# ----------------------------------------------------------------------------


#%%
def find_file(dataset, model, pattern, seed):
    """이 모델·시드의 파일 하나. 없으면 None (그 모델을 뺀다).   [5]"""
    hits = glob.glob(str(RES / SPECIAL.get((dataset, model, seed), pattern).format(s=seed)))
    if len(hits) > 1:
        raise FileNotFoundError(f"{pattern} (seed {seed}): 파일이 {len(hits)}개 맞음: {hits}")
    return Path(hits[0]) if hits else None


def load(path, column, horizon):
    """이 horizon(12개월, YAGO는 1년)의 행만 읽는다. terminated = 1 - survived (= 곧 종료되는 사실)."""
    rows = pd.read_csv(path, keep_default_na=False, dtype={"fact_id": str}, usecols=["fact_id", "origin", "horizon", "survived", column])
    rows = rows[rows.horizon == horizon].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"{path}: horizon {horizon} 행이 없다")
    rows["terminated"] = 1 - rows["survived"].astype(int)
    return rows


def signature(rows):
    """이 실행이 어떤 (사실, origin, 정답) 행을 담았는지 나타내는 값. 모델끼리 같아야 한다.   [4]"""
    key = pd.util.hash_pandas_object(rows[["fact_id", "origin", "terminated"]].astype({"origin": str}), index=False)
    return len(rows), int(key.sum() % (2**63))


def budget_rows(model, seed, rows, column):
    """k마다 한 줄: 맞힌 수(found), 고른 수(picks), precision, lift.   [2]"""
    found, picks = dict.fromkeys(FIXED_K, 0.0), dict.fromkeys(FIXED_K, 0)
    for _, g in rows.groupby("origin", sort=False):
        n, ended = len(g), g["terminated"].to_numpy(dtype=int)
        ranked = ended[np.argsort(g[column].to_numpy(dtype=float), kind="mergesort")]      # 생존확률 낮은 순   [1]
        
        for k in FIXED_K:
            take = min(k, n)
            found[k] += take * ended.sum() / n if model == "Random" else ranked[:take].sum()   # Random은 기댓값   [3]
            picks[k] += take
    
    events, total = rows["terminated"].sum(), len(rows)
    return [dict(model=model, seed=seed, budget=k, picks=picks[k], found=found[k], events=events,
                 precision=100 * found[k] / picks[k], lift=(found[k] / picks[k]) / (events / total), event_rate=100 * events / total)
            for k in FIXED_K] #모든 원점의 맞힌수를더한뒤 고른수의 합으로 나눔. 원점마다 고른수가 k로 같으니 결과는 평균과 유사. 


#%%
def main(dataset):
    out = RES / f"budget_selection_{dataset}"
    out.mkdir(exist_ok=True)

    table, signatures = [], {}
    for model, (column, pattern) in MODELS[dataset].items():
        files = {s: find_file(dataset, model, pattern, s) for s in ([42] if model == "Object degree" else FEWER_SEEDS.get((dataset, model), SEEDS))}
        if None in files.values():
            print(f"LEFT OUT: {model}  (파일 없는 시드: {[s for s, f in files.items() if f is None]})")
            continue
        
        for seed, path in files.items():
            rows = load(path, column, HORIZON[dataset])
            signatures[(model, seed)] = signature(rows)
            table += budget_rows(model, seed, rows, column)

            print(f"{model:14s} seed {seed}: {len(rows):,}행, origin {rows.origin.nunique()}개  <- {path.relative_to(RES)}", flush=True)
    
    if len(set(signatures.values())) != 1:
        raise AssertionError(f"모델마다 보는 행이 다르다: {signatures}")
    table += budget_rows("Random", 42, rows, column)             # Random은 정답만 쓰므로 어느 모델의 행이든 같다   [3]

    table = pd.DataFrame(table)
    table.to_csv(out / "budget_curve_absolute.csv", index=False)
    
    print("\n평균 (시드 평균), k = 50 / 100 / 250 에서 found | precision")
    
    for model, g in table[table.budget.isin([50, 100, 250])].groupby("model", sort=False):#원점마다 따로
        mean = g.groupby("budget")[["found", "precision"]].mean()#생존확률 낮은 순
        print(f"  {model:14s}", "   ".join(f"{r.found:7.1f} | {r.precision:5.2f}" for r in mean.itertuples())) 
    print("\n저장:", out / "budget_curve_absolute.csv")


#%%
if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("finwiki", "finsurvkg", "icewswiki", "yago"):
        raise SystemExit(f"사용법: python make_budget_selection.py {{{'|'.join(('finwiki', 'finsurvkg', 'icewswiki', 'yago'))}}}")
    main(sys.argv[1])

# %%
