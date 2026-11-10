# Data layout

원자료는 크기와 라이선스 때문에 Git에 포함하지 않습니다. 공식 배포처에서 받은 파일을 아래 경로에 둡니다.

| Dataset | Required path | SHA-256 |
|---|---|---|
| FinWiki | `data/FinWiki/train_valid_test.pkl` | `09cbfbc154e1052730036eb92715787a94867645487e5ac8567624416ca12d5e` |
| ICEWSWiki | `data/raw/ICEWSWiki.pkl` | `7cabceb2ec0723ea3c5598de9fb0bb00f80354208f13b6f4069fe124d96209da` |
| YAGO raw | `data/YAGO/train_valid_test.pkl` | `fd81ba3fd6867400405c8c136d4aa339b1172433473659b228a99e864df47f31` |
| YAGO landmark table | `data/YAGO/processed/yago_survival_rows.csv` | `cf9cd07b6987301cc104479f1c4636df8bb5c19d979bbf7149b63d580f172e57` |

YAGO landmark table은 raw pickle에서 재생성할 수 있습니다.

```bash
python datasets/yago_survival_table.py \
  --data data/YAGO/train_valid_test.pkl \
  --output data/YAGO/processed
```

`yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv`는 종료시점 감사용입니다. 모델 입력이나 특징 생성에 사용하지 않습니다.

새 데이터셋을 추가할 때는 `datasets/task.py`의 `[begin, stop)` 및 `make_survival_labels()` 계약을 사용하고, degree를 전체 현재 risk set에서 계산한 다음 horizon 관측 mask를 적용합니다.
