"""Checks that point events preserve their time and stay separate from states."""

import pickle
from pathlib import Path
import tempfile
import unittest

from datasets.event_context import load_point_events
from datasets.interval_landmark import month_id


class EventContextTest(unittest.TestCase):
    def load_events(self, splits, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.pkl"
            path.write_bytes(pickle.dumps(["2000-12", "2001-12", *splits, {}]))
            return load_point_events(path, **kwargs)

    def test_mixed_container_deduplicates_edges_without_rounding_days(self):
        events = self.load_events([
            [
                ("S", "event", "O", "2000-02-28"),
                ("S", "state", "O", "1999-01", "2002-01"),
                ("A", "event", "B", "2000-01"),
            ],
            [("S", "event", "O", "2000-02-28")],
            [("S", "event", "O", "2000-02-01"), ("Z", "event", "O", "2005-01")],
        ])
        self.assertEqual(events.to_records(index=False).tolist(), [
            ("A", "event", "B", month_id("2000-01")),
            ("S", "event", "O", month_id("2000-02")),
            ("Z", "event", "O", month_id("2005-01")),
        ])
        self.assertEqual(events.columns.tolist(), ["subject", "relation", "object", "origin"])

    def test_years_and_empty_point_event_sets(self):
        events = self.load_events([[('S', 'event', 'O', 1982)], [], []], time_unit="year")
        self.assertEqual(events.origin.tolist(), [1982])
        empty = self.load_events([[('S', 'state', 'O', '2000-01', '2002-01')], [], []])
        self.assertTrue(empty.empty)
        self.assertEqual(str(empty.origin.dtype), 'int64')

    def test_invalid_dates_fail_with_source_timestamp(self):
        for timestamp in ("2000-13", "2001-02-29", "2000-01-32", "2000-2", "2000", None, []):
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, 'point-event timestamp'):
                self.load_events([[('S', 'event', 'O', timestamp)], [], []])
        with self.assertRaisesRegex(ValueError, 'time_unit'):
            self.load_events([[], [], []], time_unit='day')


if __name__ == "__main__":
    unittest.main()
