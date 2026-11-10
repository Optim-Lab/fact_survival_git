# Reproducibility guide

모든 명령은 저장소 루트에서 실행합니다. 기존 결과 디렉터리를 덮어쓰지 말고 새 경로를 사용하십시오.

## 1. 환경 만들기

검증 기준 환경은 Python 3.13.5, NumPy 2.3.4, pandas 2.3.1, SciPy 1.16.3, scikit-learn 1.8.0, PyTorch 2.11.0입니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python modules/neural_hazard.py
```

현재 연구 서버에서만 conda의 `libstdc++` 경로가 필요했습니다.

```bash
export LD_LIBRARY_PATH=/home1/pz29075/miniconda3/lib
```

일반적인 깨끗한 설치에서는 이 줄을 사용하지 않습니다.

## 2. 데이터 배치 및 검증

[`data/README.md`](data/README.md)의 네 파일을 정확한 경로에 둡니다.

```bash
sha256sum \
  data/FinWiki/train_valid_test.pkl \
  data/raw/ICEWSWiki.pkl \
  data/YAGO/train_valid_test.pkl \
  data/YAGO/processed/yago_survival_rows.csv
```

YAGO processed table이 없으면 원본 pickle에서 다시 만듭니다.

```bash
python datasets/yago_survival_table.py \
  --data data/YAGO/train_valid_test.pkl \
  --output data/YAGO/processed
```

`yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv`는 label 감사용이며 모델 특징으로 사용하지 않습니다.

## 3. 실행 가능한 공통 protocol

모든 새 실행은 `datasets/task.py`의 `landmark-survival-v1` 계약을 기록합니다. Baseline과 proposed run의 `run_config.json`, 최종 `report_manifest.json`에 같은 version과 규칙이 없으면 `reproduce_report.py`가 실패합니다. 이 변경 전의 legacy artifact는 새 protocol 결과로 간주하지 않으며 다시 실행해야 합니다.

### 3.1 Label과 경계

Fact의 유효 구간은 `[begin, stop)`입니다. `event_observed=False`이면 `stop`은 실제 종료가 아니라 마지막 관측 경계입니다. Landmark `c`에서 active인 행만 만들며 반드시

```text
begin <= c < stop
```

이어야 합니다. Horizon `h>0`의 공통 label은 다음과 같습니다.

```text
1   if stop > c+h
0   if stop <= c+h and event_observed
NaN if stop <= c+h and not event_observed
```

따라서 실제 종료가 정확히 `c+h`이면 0이고, 검열 경계가 정확히 `c+h`이면 unknown입니다. FinWiki/ICEWSWiki loader, YAGO table builder, YAGO adapter와 proposed experiment는 모두 `make_survival_labels()`의 결과를 사용합니다. YAGO adapter는 별도 endpoint audit CSV로 label을 다시 계산하고 저장 label과 exact/equal-NaN 비교한 뒤 endpoint를 폐기합니다.

### 3.2 Risk set과 feature 순서

각 landmark에서 다음 순서를 바꾸면 안 됩니다.

```text
전체 current active risk set R(c)
  -> age/degree/HALO-as-of-c 특징 계산
  -> horizon별 label observed/maturity mask
  -> 학습 또는 평가 행 선택
```

즉, label이 unknown이라는 이유로 degree 계산 전에 행을 제거하지 않습니다. YAGO의 object degree는 현재 snapshot의 object-role count이고, FinWiki/ICEWSWiki의 object degree는 기존 loader 정의대로 양방향 incident count입니다. Proposed의 subject degree는 subject-role count입니다. 서로 다른 정의는 dataset adapter에 명시하지만, 같은 dataset 안의 비교 모형은 동일하게 생성된 landmark table을 사용합니다.

종료시점과 `event_observed`는 label 감사에만 사용하며 feature allowlist에는 들어가지 않습니다. HALO 이력도 각 행의 `c` 이하에서 관측된 시작시점만 사용합니다.

### 3.3 Rolling-origin 학습 조건

Origin `v`, 행 `i`, horizon `h_k`에 대해 proposed joint model의 학습 mask는

```text
m[i,k;v] = observed(label[i,k])
           and landmark[i] < v
           and landmark[i] + h[k] <= v
```

입니다. 짧은 horizon만 성숙했다면 그 label만 loss에 들어가고 긴 horizon의 NaN은 산술 전에 제외됩니다. 평가 population은 `landmark == v`이면서 해당 horizon label이 관측된 행입니다. Baseline은 같은 조건을 horizon별로 적용합니다.

각 origin에서는 다음을 수행합니다.

1. 과거의 성숙 label만 선택합니다.
2. scaler와 category vocabulary를 그 training fold에서만 적합합니다.
3. unseen category는 ID 0으로 보냅니다.
4. 모델을 이전 origin checkpoint에서 warm-start하지 않고 처음부터 학습합니다.
5. origin `v`의 전체 risk set을 예측한 뒤 horizon별 observed evaluation mask를 적용합니다.
6. 이전 test origin의 label도 `landmark+h <= v`가 된 이후에는 다음 origin의 학습에 사용할 수 있습니다.

Development에서는 고정 후보 epoch `{5,15,30}`를 전체 development OOF population의 horizon별 row Brier 산술평균으로 비교합니다. 선택 결과를 `selection.json`에 먼저 고정하며, test에서는 epoch 재선택이나 early stopping을 하지 않습니다. 부분 origin 실행은 모든 조각을 합쳐 원래 population과 1:1 일치할 때만 complete run으로 승격됩니다.

### 3.4 동일 비교 population

Baseline과 proposed prediction은 `(fact_id, origin, horizon)`으로 one-to-one outer join합니다. 누락/추가 key, 중복 key 또는 label 불일치가 하나라도 있으면 실패합니다. 모든 확률은 finite `[0,1]`이어야 하고 neural survival은 공통 행에서 horizon에 따라 단조감소해야 합니다. 이 검사를 통과한 row만 동일 Brier 및 paired fact-cluster bootstrap에 사용합니다.

## 4. 비교 baseline 생성

Proposed experiment가 아래 디렉터리 이름을 사용해 baseline을 1:1 join합니다.

```bash
python experiments/yago_bootstrap_significance.py \
  --evaluate-test \
  --output results/landmark-survival-v1/yago_halo_rolling_test

python main.py --dataset finwiki --evaluate-test \
  --output results/landmark-survival-v1/finwiki_rolling_test

python main.py --dataset icewswiki --evaluate-test \
  --output results/landmark-survival-v1/icewswiki_rolling_test
```

YAGO shared logistic baseline만 별도로 확인하려면 다음을 사용합니다.

```bash
python main.py --dataset yago --evaluate-test \
  --output results/landmark-survival-v1/yago_shared_riskset_rolling_test
```

## 5. Development에서 epoch 고정

Test를 보기 전에 각 dataset의 development 결과를 완성합니다.

```bash
python main.py --model proposed --dataset yago \
  --epochs 5 15 30 \
  --output results/landmark-survival-v1/proposed_yago_v2_development

python main.py --model proposed --dataset finwiki \
  --epochs 5 15 30 \
  --output results/landmark-survival-v1/proposed_finwiki_v2_development

python main.py --model proposed --dataset icewswiki \
  --epochs 5 15 30 \
  --output results/landmark-survival-v1/proposed_icewswiki_development
```

큰 실행은 `--development-origins`로 나눈 뒤 `--merge-parts`로 합칠 수 있습니다. 선택은 전체 development population을 합친 뒤 한 번만 수행합니다.

## 6. Frozen rolling test

YAGO seed 42는 원래 결과 구조와 맞추기 위해 세 variant를 분리합니다.

```bash
python main.py --model proposed --dataset yago --variants neural --seed 42 \
  --selection-from results/landmark-survival-v1/proposed_yago_v2_development --evaluate-test \
  --output results/landmark-survival-v1/proposed_yago_final_seed42

python main.py --model proposed --dataset yago --variants cloglog --seed 42 \
  --selection-from results/landmark-survival-v1/proposed_yago_v2_development --evaluate-test \
  --output results/landmark-survival-v1/proposed_yago_cloglog_seed42

python main.py --model proposed --dataset yago --variants half_life --seed 42 \
  --selection-from results/landmark-survival-v1/proposed_yago_v2_development --evaluate-test \
  --output results/landmark-survival-v1/proposed_yago_half_life_seed42
```

Seed 43과 44는 development 설정을 그대로 사용합니다.

```bash
python main.py --model proposed --dataset yago --variants neural --seed 43 \
  --selection-from results/landmark-survival-v1/proposed_yago_v2_development --evaluate-test \
  --output results/landmark-survival-v1/proposed_yago_final_seed43
python main.py --model proposed --dataset yago --variants neural --seed 44 \
  --selection-from results/landmark-survival-v1/proposed_yago_v2_development --evaluate-test \
  --output results/landmark-survival-v1/proposed_yago_final_seed44
```

FinWiki와 ICEWSWiki도 같은 구조를 사용합니다. Seed 42에서는 `--variants neural cloglog half_life`, seed 43/44에서는 `--variants neural`을 사용합니다.

```bash
python main.py --model proposed --dataset finwiki \
  --variants neural cloglog half_life --seed 42 \
  --selection-from results/landmark-survival-v1/proposed_finwiki_v2_development --evaluate-test \
  --output results/landmark-survival-v1/proposed_finwiki_final_seed42

python main.py --model proposed --dataset icewswiki \
  --variants neural cloglog half_life --seed 42 \
  --selection-from results/landmark-survival-v1/proposed_icewswiki_development --evaluate-test \
  --output results/landmark-survival-v1/proposed_icewswiki_final_seed42
```

Seed 43/44 명령은 dataset과 seed/output만 바꾸고 `--variants neural`을 사용합니다.

`--test-origins`로 origin을 분할 실행한 경우 모든 part를 `--merge-parts`로 합쳐야 합니다. 부분 결과를 전체 test 성능으로 보고하면 안 됩니다.

## 7. 보고서와 표 재생성

`configs/reproduction_runs.json`에 지정된 run directory를 모두 만든 뒤 실행합니다.

```bash
python reproduce_report.py \
  --output results/landmark-survival-v1/reproduced_report
```

이 명령은 다음을 수행합니다.

- seed 간 `(fact_id, origin, horizon)`과 label 일치 확인
- 확률 범위와 neural survival 단조성 확인
- row/fact Brier 재계산
- seed 요약 재계산
- 27개 paired fact-cluster bootstrap과 dataset별 Holm 보정
- `results/reference/`의 공개 기준 CSV와 수치 대조
- `benchmark_tables.tex` 생성

같은 output 경로가 이미 있으면 실패하도록 되어 있습니다. 새 경로를 지정하십시오.

## 8. 발표자료 컴파일

XeLaTeX와 한국어 TeX 패키지가 설치된 환경에서 실행합니다.

```bash
xelatex "proposed model.tex"
xelatex "proposed model.tex"
```

## 9. 공개 artifact 구성

Git에는 코드와 `results/reference/`만 포함합니다. 다음은 GitHub Release 또는 Zenodo archive에 넣습니다.

- 원자료 또는 공식 다운로드 지침과 checksum
- `configs/reproduction_runs.json`에 명시된 각 run directory
- 각 run의 `run_config.json`, `selection.json`, checkpoint, `test_predictions.csv.gz`
- 로그와 Slurm script는 선택 사항

공개 전에는 archive를 빈 디렉터리에 풀고 7단계가 통과하는지 확인합니다.
