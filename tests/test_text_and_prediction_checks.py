"""Tests for text extraction, classifier errors and evaluation."""

import contextlib
import io
import json
import sys
import tempfile
import textwrap
import unittest
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from helpers import FakeResponse, fixture

from aidisclose import LocalSource, Pipeline
from aidisclose.cli import main
from aidisclose.lexicon import Lexicon
from aidisclose.models import Document, Filing
from aidisclose.text import html_to_text, split_10k_items
from aidisclose.validate import bootstrap_report, classification_report, cohen_kappa

DEMO = Path(str(resources.files("aidisclose.demo").joinpath("data")))


def run_cli(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        main(list(args))
    return out.getvalue(), err.getvalue()


class TenKSplitterTests(unittest.TestCase):
    def test_irregular_headings(self):
        secs = split_10k_items(html_to_text(fixture("hard_10k.htm").read_bytes()))
        self.assertEqual(list(secs), ["10-K:cover", "10-K:1", "10-K:1A", "10-K:6", "10-K:7", "10-K:16"])
        self.assertIn("forecast freight volumes", secs["10-K:1"])  # combined "Items 1 and 2" heading
        self.assertIn("Item 7 of this report", secs["10-K:1"])  # cross-reference stays in Item 1
        self.assertIn("FORWARD-LOOKING STATEMENTS", secs["10-K:cover"])  # not mistaken for Item 16
        self.assertIn("reduced fuel consumption", secs["10-K:7"])  # long run-in heading
        self.assertNotIn("Business and Properties", secs["10-K:6"])  # table of contents skipped
        self.assertEqual(secs["10-K:16"], "ITEM 16. FORM 10-K SUMMARY None.")


class LexiconReasonTests(unittest.TestCase):
    def test_filter_reasons(self):
        lex = Lexicon.load("default", tiers=None)
        self.assertEqual(lex.find("The ML division grew.").reasons, {"ml_acronym": "requires_cooccurrence"})
        self.assertEqual(lex.find("artificial insemination (AI) revenue").reasons, {"ai_acronym": "negative_context"})
        hits = lex.find("AI programs grew.", disabled_terms={"ai_acronym"})
        self.assertEqual(hits.reasons, {"ai_acronym": "document_definition"})


class SentenceIdTests(unittest.TestCase):
    def test_ids_do_not_depend_on_section_filter_or_tiers(self):
        src = LocalSource(DEMO, metadata=DEMO / "metadata.csv")
        f = [x for x in src.list_filings() if x.firm_id == "northwind" and x.fiscal_year == 2023]
        full = Pipeline().run(f, source=src).sentences
        only7 = Pipeline(sections=["10-K:7"], tiers=None).run(f, source=src).sentences
        self.assertTrue(len(only7) >= 1)
        merged = only7.merge(full, on="sentence_id", suffixes=("_a", "_b"))
        self.assertEqual(len(merged), len(only7))
        self.assertTrue((merged["text_a"] == merged["text_b"]).all())
        self.assertTrue(full["sentence_id"].str.contains("#10-K:").all())

    def test_filtered_rows_record_reason(self):
        src = LocalSource(DEMO, metadata=DEMO / "metadata.csv")
        f = [x for x in src.list_filings() if x.firm_id == "bluefield" and x.fiscal_year == 2023]
        s = Pipeline().run(f, source=src).sentences
        row = s[s["text"].str.startswith("AI programs")].iloc[0]
        self.assertEqual(json.loads(row["details"])["filter_reasons"], {"ai_acronym": "document_definition"})


class ClassifierErrorTests(unittest.TestCase):
    def test_unparseable_llm_reply_is_counted(self):
        from aidisclose.classify.llm import LLMClassifier

        class Chat:
            def post(self, url, json=None, headers=None, timeout=None):
                import json as _j

                return FakeResponse(_j.dumps({"choices": [{"message": {"content": "no idea"}}]}).encode())

        clf = LLMClassifier(model="m", cache_path=None, session=Chat())
        doc = Document(
            filing=Filing(source="x", filing_id="x1", firm_id="x", firm_name="X Corp."),
            text="We use machine learning to detect fraud in 2 million payments every day across our network.",
            language="en",
        )
        res = Pipeline(backend=clf).run_documents([doc])
        self.assertEqual(int(res.documents.iloc[0]["n_classifier_errors"]), 1)


class WeightedMetricTests(unittest.TestCase):
    yt = ["SUBSTANTIVE", "SUBSTANTIVE", "WEAK", "WEAK", "NO_CLAIM", "NO_CLAIM", "NO_CLAIM", "WEAK"]
    yp = ["SUBSTANTIVE", "WEAK", "WEAK", "NO_CLAIM", "NO_CLAIM", "NO_CLAIM", "SUBSTANTIVE", "WEAK"]
    w = [1.0, 1.0, 5.0, 5.0, 10.0, 10.0, 10.0, 5.0]

    def test_weighted_kappa_matches_sklearn(self):
        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:  # pragma: no cover
            self.skipTest("scikit-learn not installed")
        self.assertAlmostEqual(
            cohen_kappa(self.yt, self.yp, weights=self.w), cohen_kappa_score(self.yt, self.yp, sample_weight=self.w)
        )
        rep = classification_report(self.yt, self.yp, weights=self.w)
        self.assertFalse(np.isnan(rep["kappa"]))

    def test_bootstrap_uses_weights(self):
        unweighted = bootstrap_report(self.yt, self.yp, n_boot=200)
        weighted = bootstrap_report(self.yt, self.yp, weights=self.w, n_boot=200)
        self.assertNotAlmostEqual(unweighted["macro_f1"][0], weighted["macro_f1"][0])
        self.assertAlmostEqual(
            weighted["macro_f1"][0], classification_report(self.yt, self.yp, weights=self.w)["macro_f1"]
        )
        for key in ("SUBSTANTIVE.precision", "WEAK.recall", "NO_CLAIM.f1", "kappa", "substantive_vs_rest.f1"):
            pt, lo, hi = weighted[key]
            self.assertLessEqual(lo, hi)
        strat = bootstrap_report(self.yt, self.yp, weights=self.w, strata=["a"] * 4 + ["b"] * 4, n_boot=100)
        self.assertIn("accuracy", strat)

    def test_cross_validation_weighted(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("scikit-learn not installed")
        from aidisclose.classify.supervised import SupervisedClassifier
        from aidisclose.cli import _gold_candidates

        gold = pd.read_csv(DEMO / "demo_gold.csv")
        cands = _gold_candidates(gold)
        w = np.where(gold["gold_state"] == "SUBSTANTIVE", 1.0, 3.0)
        rep = SupervisedClassifier.cross_validate(cands, gold["gold_state"].tolist(), k=3, weights=w, n_boot=50)
        self.assertTrue(rep["weighted"])
        self.assertIn("macro_f1", rep["ci"])
        self.assertEqual(len(rep["oof_predictions"]), len(gold))


class CLIBehaviourTests(unittest.TestCase):
    def test_weighted_without_key_is_rejected_and_unweighted_mismatch_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = Path(tmp) / "res"
            run_cli(
                "run",
                "--source",
                "local",
                "--dir",
                str(DEMO),
                "--metadata",
                str(DEMO / "metadata.csv"),
                "--out",
                str(res),
                "-q",
            )
            s = pd.read_csv(res / "sentences.csv")
            gold = s[["sentence_id", "text", "state"]].head(10).rename(columns={"state": "gold_state"})
            gold.loc[gold.index[0], "text"] = "a different sentence"
            gold.to_csv(Path(tmp) / "gold.csv", index=False)
            with self.assertRaisesRegex(SystemExit, "requires --key"):
                run_cli(
                    "evaluate",
                    "--gold",
                    str(Path(tmp) / "gold.csv"),
                    "--results",
                    str(res),
                    "--weighted",
                    "--n-boot",
                    "20",
                )
            out, err = run_cli(
                "evaluate", "--gold", str(Path(tmp) / "gold.csv"), "--results", str(res), "--n-boot", "20"
            )
            self.assertIn("1 annotated rows match a prediction with different text", err)
            self.assertIn("n = 9", out)

    def test_multiple_results_and_plugin_source(self):
        plugin = textwrap.dedent(
            '''
            from aidisclose.models import Filing
            from aidisclose.sources import LocalSource, register_source


            class DemoRegistry(LocalSource):
                """A custom adapter wrapping a folder, configured through --source-option."""

                name = "demoreg"

                def __init__(self, folder, meta=None):
                    super().__init__(folder, metadata=meta)

                def list_filings(self, start_year=None, end_year=None, firm=None):
                    out = super().list_filings()
                    return [f for f in out if firm is None or f.firm_id == firm]


            register_source("demoreg", DemoRegistry)
            '''
        )
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "my_test_plugin.py").write_text(plugin, encoding="utf-8")
            sys.path.insert(0, tmp)
            try:
                out, _ = run_cli(
                    "--plugin",
                    "my_test_plugin",
                    "run",
                    "--source",
                    "demoreg",
                    "--source-option",
                    f"folder={DEMO}",
                    "--source-option",
                    f"meta={DEMO / 'metadata.csv'}",
                    "--query",
                    "firm=kestrel",
                    "--end",
                    "2023",
                    "--out",
                    str(Path(tmp) / "r1"),
                    "-q",
                )
                self.assertIn("processed 2/2 documents", out)
            finally:
                sys.path.remove(tmp)
            run_cli(
                "run",
                "--source",
                "local",
                "--dir",
                str(DEMO),
                "--metadata",
                str(DEMO / "metadata.csv"),
                "--start",
                "2024",
                "--out",
                str(Path(tmp) / "r2"),
                "-q",
            )
            out, _ = run_cli(
                "panel",
                "--results",
                str(Path(tmp) / "r1"),
                str(Path(tmp) / "r2"),
                "--level",
                "firm-year",
                "--out",
                str(Path(tmp) / "p.csv"),
                "-q",
            )
            panel = pd.read_csv(Path(tmp) / "p.csv")
            self.assertIn("kestrel", set(panel["firm_id"]))
            self.assertIn("northwind", set(panel["firm_id"]))
            self.assertIn("n_classifier_errors", panel.columns)


if __name__ == "__main__":
    unittest.main()
