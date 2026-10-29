"""Tests for partial group means, checkpoint integrity and empty data."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from test_pipeline_execution import CountingSource, demo

import aidisclose
from aidisclose import Pipeline
from aidisclose.aggregate import group_panel
from aidisclose.pipeline import Checkpoint, DocumentOutcome
from aidisclose.validate.external import eurostat_ai_share_by_industry, parse_jsonstat


class TestFeatureIntegrity(unittest.TestCase):
    def panel(self):
        return pd.DataFrame(
            {
                "g": ["X", "X"],
                "keyword_any": [1, 1],
                "claim_any": pd.array([1, pd.NA], dtype="Int64"),
                "substantive_any": pd.array([1, pd.NA], dtype="Int64"),
                "classification_complete": [True, False],
                "keyword_per_10k_words": [10, 10],
                "substantive_per_10k_words": [10, np.nan],
                "substantive_share": [1, np.nan],
                "size": [2, 3],
            }
        )

    def test_wheel_import_path(self):
        root = os.environ.get("AIDISCLOSE_EXPECT_INSTALL_ROOT")
        if root:
            self.assertTrue(Path(aidisclose.__file__).resolve().is_relative_to(Path(root).resolve()))

    def test_group_means_expose_missingness_and_observed_denominator(self):
        df = self.panel()
        row = group_panel(df, by=["g"], weight="size").iloc[0]
        self.assertTrue(pd.isna(row.mean_substantive_per_10k_words))
        self.assertTrue(pd.isna(row.mean_substantive_share))
        self.assertEqual(row.mean_substantive_per_10k_words_observed, 10)
        self.assertEqual(row.n_firms_substantive_per_10k_words_observed, 1)
        self.assertEqual(row.mean_keyword_per_10k_words, 10)
        df.loc[1, "size"] = 0
        row = group_panel(df, by=["g"], weight="size").iloc[0]
        self.assertEqual(row.mean_substantive_per_10k_words, 10)

    def test_empty_group_output_retains_headers(self):
        empty = group_panel(self.panel(), by=["g"], min_firms=3)
        self.assertEqual(len(empty), 0)
        self.assertIn("g", empty.columns)
        self.assertIn("mean_substantive_share_observed", empty.columns)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.csv"
            empty.to_csv(path, index=False)
            self.assertEqual(len(pd.read_csv(path)), 0)
        self.assertIn("share_substantive", group_panel(self.panel().iloc[:0], by=["g"]).columns)

    def test_empty_jsonstat_keeps_its_dimensions(self):
        obj = {
            "id": ["size_emp", "nace_r2", "geo", "time"],
            "size": [1, 1, 1, 1],
            "dimension": {
                "size_emp": {"category": {"index": {"GE10": 0}}},
                "nace_r2": {"category": {"index": {"C": 0}}},
                "geo": {"category": {"index": {"DK": 0}}},
                "time": {"category": {"index": {"2023": 0}}},
            },
            "value": {},
        }
        obj["id"] = ["indic_is", "unit"] + obj["id"]
        obj["size"] = [1, 1] + obj["size"]
        obj["dimension"].update(
            {"indic_is": {"category": {"index": {"E_AI_TANY": 0}}}, "unit": {"category": {"index": {"PC_ENT": 0}}}}
        )
        parsed = parse_jsonstat(obj)
        self.assertTrue({"size_emp", "nace_r2", "geo", "time", "value"}.issubset(parsed.columns))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.json"
            path.write_text(json.dumps(obj))
            result = eurostat_ai_share_by_industry(path=path)
        self.assertEqual(len(result), 0)
        self.assertEqual(list(result.columns), ["country", "nace_section", "year", "eurostat_share"])

    def test_checkpoint_content_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as td:
            c = Checkpoint(td, "same")
            c.write("a", DocumentOutcome([], {"filing_id": "a", "status": "ok", "n_substantive": 0}))
            self.assertEqual(c.statuses(), {"a": "ok"})
            path = c._path("a")
            data = json.loads(path.read_text())
            data["doc_row"]["n_substantive"] = 99
            path.write_text(json.dumps(data))
            self.assertEqual(c.statuses(), {})
            self.assertIsNone(c.read("a"))

    def test_resume_reprocesses_corrupted_valid_json(self):
        source, filings = demo()
        with tempfile.TemporaryDirectory() as td:
            original = Pipeline().run(filings[:2], CountingSource(source), checkpoint_dir=td)
            path = sorted((Path(td) / "parts").glob("*.json"))[0]
            data = json.loads(path.read_text())
            fid = data["filing_id"]
            data["doc_row"]["n_substantive"] = 999
            path.write_text(json.dumps(data))
            counted = CountingSource(source)
            resumed = Pipeline().run(filings[:2], counted, checkpoint_dir=td, resume=True)
            self.assertEqual(counted.fetched, [fid])
            pd.testing.assert_frame_equal(original.documents, resumed.documents)

    def test_checkpoint_metadata_replacement_failure_preserves_old_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            Checkpoint(td, "same")
            path = Path(td) / "checkpoint.json"
            before = path.read_bytes()
            with mock.patch("aidisclose.pipeline.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    Checkpoint(td, "same", resume=True)
            self.assertEqual(path.read_bytes(), before)
