"""Calibration: "종료할 확률 p 라고 예측한 사실이 실제로 p 만큼 종료하는가?"   (4개 데이터셋 x 모든 기간 x 예산 실험과 같은 모델 + KM)

    python calibration_plot.py finwiki yago finsurvkg icewswiki     # 구간별 값을 계산해 calibration_<데이터셋>.csv 저장
    python calibration_plot.py figure                               # 그 CSV 로 그림 + 표 (ECE) 만들기

예측 종료확률 = 1 - 예측 생존확률.  모델마다·시드마다 예측을 같은 수의 구간 10개로 나누고 (예측값 기준 분위수),
구간마다 "예측 평균"과 "실제 종료 비율"을 구한다.  점이 대각선 위에 있으면 잘 맞는 것.
ECE_equal_mass = 위 구간(같은 개수)별 |예측 평균 - 실제 비율| 의 가중 평균.  ECE_standard = 표준 ECE: 확률 0~1 을 같은 폭 10개로 나눠 같은 계산 (둘 다 %포인트, 작을수록 좋다).   Ratio = 예측 종료 수 / 실제 종료 수 (1 이면 개수가 맞는다).
점수가 같은 사실이 많은 단순 모델은 구간이 10개보다 적게 나온다 (같은 값은 한 구간).
"""
import sys

import numpy as np
import pandas as pd

import budget_curve as K

BINS = 10


def models(dataset):
    """예산 실험과 같은 모델 + KM (Table 1 에 있는 기준 모델.  classical_rolling 파일의 p_km 열, 무작위성 없음)."""
    m = {"KM": ("p_km", f"classical_rolling_20261030/{dataset}/test_predictions.csv.gz")}
    m.update(K.C.MODELS[dataset])
    return m


def bins_of(predicted, observed):
    """구간별 (예측 평균, 실제 비율, 행 수).  예측값이 같은 행은 같은 구간에 둔다."""
    part = pd.DataFrame({"predicted": predicted, "observed": observed})
    part["bin"] = pd.qcut(part.predicted, BINS, duplicates="drop", labels=False)
    out = part.groupby("bin").agg(predicted=("predicted", "mean"), observed=("observed", "mean"), rows=("observed", "size"))
    return out.reset_index()


def standard_ece(predicted, observed):
    """표준 ECE: 확률 0~1 을 같은 폭 BINS 개로 나눠 구간별 |예측 평균 - 실제 비율| 을 행 수로 가중 평균 (%포인트)."""
    part = pd.DataFrame({"predicted": predicted, "observed": observed, "bin": np.minimum((predicted * BINS).astype(int), BINS - 1)})
    g = part.groupby("bin").agg(predicted=("predicted", "mean"), observed=("observed", "mean"), rows=("observed", "size"))
    return 100 * ((g.predicted - g.observed).abs() * g.rows).sum() / g.rows.sum()


def run(dataset):
    parts, standard = [], []
    for model, (column, pattern) in models(dataset).items():
        for seed in [42] if model in K.C.NO_SEED or model == "KM" else K.C.FEWER_SEEDS.get((dataset, model), K.B.SEEDS):
            d = pd.read_csv(K.B.find_file(dataset, model, pattern, seed), keep_default_na=False, usecols=["horizon", "survived", column])
            for horizon, g in d.groupby("horizon"):
                predicted, observed = 1 - g[column].to_numpy(float), 1 - g.survived.astype(int).to_numpy()
                t = bins_of(predicted, observed)
                standard.append(dict(model=model, seed=seed, horizon=horizon, ECE_standard=standard_ece(predicted, observed)))
                parts.append(t.assign(model=model, seed=seed, horizon=horizon))
        print(f"  {dataset} {model} done", flush=True)
    pd.concat(parts).replace({"model": {"Proposed": "LaST"}}).to_csv(K.C.HERE / f"calibration_{dataset}.csv", index=False)
    pd.DataFrame(standard).replace({"model": {"Proposed": "LaST"}}).to_csv(K.C.HERE / f"calibration_standard_{dataset}.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1:] == ["figure"]:
        import calibration_figure
        calibration_figure.main()
    else:
        for d in [a for a in sys.argv[1:] if a in K.C.MODELS] or list(K.C.MODELS):
            run(d)
