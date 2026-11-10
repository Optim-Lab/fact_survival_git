"""Stage 1 생존예측의 공통 라벨 정의.

Temporal fact:
    (subject, relation, object, [begin, stop), event_observed)

stop은 실제 종료 또는 마지막 관측 시점이다. event_observed=False이면
stop에서 우측 검열된 것이다. 데이터셋별 loader는 시간 단위만 정규화하고
라벨 의미와 경계 조건은 이 모듈을 공통으로 사용한다.

한 행의 의미:
    landmark 시점에 살아 있는 fact 하나

출력 라벨:
    1    = horizon 이후에도 fact가 살아 있음
    0    = horizon 안에 실제로 종료됨
    None = horizon까지 관측하지 못해 정답을 모름
"""

#%%
from __future__ import annotations

import numpy as np


PROTOCOL_VERSION = "landmark-survival-v1"
PROTOCOL_RULES = {
    "interval": "[begin, stop)",
    "label": "1=stop>landmark+h; 0=observed stop<=landmark+h; unknown otherwise",
    "feature_order": "full current risk set -> feature computation -> observed-label mask",
    "train_mask": "landmark<origin and landmark+horizon<=origin and label observed",
    "evaluation_mask": "landmark=origin and label observed",
    "vocabulary": "fit on rolling training rows; unseen category=0",
    "selection": "development only; frozen epoch reused for test",
    "refit": "from scratch at every rolling origin",
}


#%%
def is_active(
    begin_time: int,
    stop_time: int,
    landmark_time: int,
) -> bool:#enabled가 참/거짓 값이라는 타입 힌트
    """Fact가 landmark 시점에 살아 있는지 확인한다.

    구간은 [begin, stop)이다.

    예:
        begin=10, stop=20이면
        10~19에는 active
        20부터는 inactive
    """
    return begin_time <= landmark_time < stop_time

def start(begin_time: int):
    print(begin_time)

start(10)

#%%
def make_survival_labels(
    begin_times,
    stop_times,
    landmark_times,
    horizon: int,
    *,
    event_observed,
) -> np.ndarray:
    """공통 fixed-horizon 라벨을 벡터화해 만든다.

    반환값은 1(생존), 0(실제 종료), NaN(검열)이다. 구간은
    [begin, stop)이므로 stop == landmark + horizon이면 실제 종료는 0,
    검열은 NaN이다.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    begin, stop, landmark, event = np.broadcast_arrays(
        np.asarray(begin_times),
        np.asarray(stop_times),
        np.asarray(landmark_times),
        np.asarray(event_observed),
    )
    if not np.isin(event, (False, True)).all():
        raise ValueError("event_observed must contain only booleans")
    if np.any(stop <= begin):
        raise ValueError("stop must be later than begin")
    if np.any((landmark < begin) | (landmark >= stop)):
        raise ValueError("fact must be active at landmark")

    target = landmark + horizon
    return np.where(stop > target, 1.0, np.where(event.astype(bool), 0.0, np.nan))


def make_survival_label(
    begin_time: int,
    stop_time: int,
    landmark_time: int,
    horizon: int,
    *,
    event_observed: bool,
) -> int | None:
    """모든 데이터셋과 동일한 경계 규칙으로 단일 라벨을 만든다."""
    value = make_survival_labels(
        begin_time, stop_time, landmark_time, horizon, event_observed=event_observed
    ).item()
    return None if np.isnan(value) else int(value)


def mature_label_mask(
    labels,
    landmark_times,
    horizons,
    origin: int,
) -> np.ndarray:
    """Return labels knowable at a rolling origin.

    A training label is usable only when its landmark precedes ``origin``, its
    full prediction window ends no later than ``origin``, and the common label
    contract did not mark it as right-censored/unknown.
    """
    label_array = np.asarray(labels, dtype=float)
    landmark = np.asarray(landmark_times)
    horizon = np.asarray(horizons)
    if label_array.ndim != 2:
        raise ValueError("labels must have shape (rows, horizons)")
    if landmark.shape != (len(label_array),):
        raise ValueError("landmark_times must have one value per row")
    if horizon.shape != (label_array.shape[1],):
        raise ValueError("horizons must have one value per label column")
    return (
        np.isfinite(label_array)
        & (landmark[:, None] < origin)
        & (landmark[:, None] + horizon[None, :] <= origin)
    )


#%%
if __name__ == "__main__":
    print("Example 1: horizon까지 생존")
    label_1 = make_survival_label(
        begin_time=10,
        stop_time=20,
        landmark_time=12,
        horizon=3,
        event_observed=True,
    )
    print("label =", label_1)
    assert label_1 == 1

    print("\nExample 2: horizon 안에 종료")
    label_2 = make_survival_label(
        begin_time=10,
        stop_time=15,
        landmark_time=12,
        horizon=3,
        event_observed=True,
    )
    print("label =", label_2)
    assert label_2 == 0

    print("\nExample 3: right censoring")
    label_3 = make_survival_label(
        begin_time=10,
        stop_time=30,
        landmark_time=25,
        horizon=5,
        event_observed=False,
    )
    print("label =", label_3)
    assert label_3 is None

    print("\nExample 4: 종료월에는 이미 inactive")
    active = is_active(
        begin_time=10,
        stop_time=20,
        landmark_time=20,
    )
    print("active =", active)
    assert active is False

    print("\nTASK LABEL TESTS PASSED")