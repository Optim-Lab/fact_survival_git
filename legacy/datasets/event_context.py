"""Load observed point events without turning them into duration targets."""

from datetime import date
from pathlib import Path
import re

import pandas as pd

from datasets.interval_landmark import month_id, safe_load_dataset


def load_point_events(path: Path, *, time_unit: str = "month") -> pd.DataFrame:
    """Return unique (subject, relation, object, origin) event edges.

    ``origin`` is the raw event month or year, with no endpoint rounding.
    The history builder must apply its observation cutoff before using events.
    Five-field state facts are excluded. Multiple occurrences of the same edge
    in one time unit become one graph edge.
    """
    if time_unit not in {"month", "year"}:
        raise ValueError("point-event time_unit must be 'month' or 'year'")
    data = safe_load_dataset(Path(path))
    pattern = r"[0-9]{4}-[0-9]{2}(?:-[0-9]{2})?" if time_unit == "month" else r"[0-9]{4}"
    timestamps: dict[str | int, int] = {}
    events = set()
    for split in data[2:5]:
        for fact in split:
            if len(fact) != 4:
                continue
            subject, relation, obj, timestamp = fact
            if not isinstance(timestamp, (str, int)):
                raise ValueError(f"invalid {time_unit} point-event timestamp {timestamp!r} in {path}")
            if timestamp not in timestamps:
                text = str(timestamp)
                try:
                    if re.fullmatch(pattern, text) is None:
                        raise ValueError("unexpected date format")
                    suffix = "-01" if len(text) == 7 else "-01-01" if len(text) == 4 else ""
                    date.fromisoformat(text + suffix)
                    timestamps[timestamp] = month_id(text) if time_unit == "month" else int(text)
                except ValueError as error:
                    raise ValueError(
                        f"invalid {time_unit} point-event timestamp {timestamp!r} in {path}"
                    ) from error
            events.add((subject, relation, obj, timestamps[timestamp]))
    frame = pd.DataFrame(events, columns=["subject", "relation", "object", "origin"])
    frame["origin"] = frame["origin"].astype("int64")
    return frame.sort_values(["origin", "relation", "subject", "object"]).reset_index(drop=True)
