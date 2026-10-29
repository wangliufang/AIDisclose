"""Tests for classifier and dataset integrity."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from test_backends import FakeChat, cand

from aidisclose import Document, Filing, Pipeline, Results
from aidisclose.classify.llm import LLMClassifier
from aidisclose.classify.supervised import SupervisedClassifier
from aidisclose.sources.base import Source
from aidisclose.validate.sampling import merge_annotations


class TestFeatureIntegrity(unittest.TestCase):
    def test_partial_votes_and_ties_remain_missing(self):
        c = cand("We deployed artificial intelligence for logistics.")
        with tempfile.TemporaryDirectory() as d:
            chat = FakeChat(['{"state":"SUBSTANTIVE","reason":"DEPLOYMENT"}'] + ["bad"] * 4)
            clf = LLMClassifier(
                model="fake", n_votes=5, temperature=0.7, session=chat, cache_path=Path(d) / "c.sqlite", max_retries=0
            )
            p = clf.classify_one(c)
            self.assertEqual(p.as_row()["classification_status"], "error")
            self.assertEqual(p.as_row()["state"], "")
            self.assertEqual(p.details["valid_votes"], 1)
            self.assertEqual(p.details["requested_votes"], 5)
            tie = FakeChat(['{"state":"WEAK","reason":"VAGUE"}', '{"state":"SUBSTANTIVE","reason":"DEPLOYMENT"}'])
            clf = LLMClassifier(model="fake", n_votes=2, temperature=0.7, session=tie, cache_path=None, max_retries=0)
            self.assertEqual(clf.classify_one(c).as_row()["classification_status"], "error")

    def test_different_fitted_models_cannot_resume(self):
        texts = [
            "We deployed artificial intelligence to route trucks in 2024.",
            "We launched artificial intelligence to route ships in 2024.",
            "We plan to test artificial intelligence for routing in 2025.",
            "We plan to test artificial intelligence for shipping in 2025.",
        ]
        c = [cand(x) for x in texts]
        a = SupervisedClassifier.train(c, ["SUBSTANTIVE", "SUBSTANTIVE", "WEAK", "WEAK"])
        b = SupervisedClassifier.train(c, ["WEAK", "WEAK", "SUBSTANTIVE", "SUBSTANTIVE"])
        self.assertNotEqual(a.config()["fitted_model_sha256"], b.config()["fitted_model_sha256"])
        p = Pipeline()
        p.classifier = a
        q = Pipeline()
        q.classifier = b
        f = Filing(source="fake", filing_id="f", firm_id="a", firm_name="Acme", fiscal_year=2024)

        class FakeSource(Source):
            name = "fake"

            def list_filings(self, **kwargs):
                return [f]

            def fetch_document(self, filing):
                return Document(filing=filing, language="en", text=texts[0])

        with tempfile.TemporaryDirectory() as d:
            p.run([f], FakeSource(), checkpoint_dir=d)
            with self.assertRaisesRegex(ValueError, "different configuration"):
                q.run([f], FakeSource(), checkpoint_dir=d, resume=True)
        with tempfile.TemporaryDirectory() as d:
            path = a.save(Path(d) / "m.joblib")
            self.assertEqual(
                a.config()["fitted_model_sha256"], SupervisedClassifier.load(path).config()["fitted_model_sha256"]
            )

    def test_missing_provenance_never_silent(self):
        f = Filing(source="x", filing_id="f", firm_id="a", firm_name="Acme", fiscal_year=2024)
        r = Pipeline().run_documents(
            [Document(filing=f, language="en", text="We deployed artificial intelligence in logistics.")]
        )
        with tempfile.TemporaryDirectory() as d:
            r.save(d)
            (Path(d) / "provenance.json").unlink()
            with self.assertRaisesRegex(ValueError, "provenance is missing"):
                Results.load(d)
            self.assertEqual(len(Results.load(d, allow_unverified=True).documents), 1)

    def test_gold_cannot_override_predictions(self):
        g = pd.DataFrame({"sentence_id": ["x"], "gold_state": ["SUBSTANTIVE"], "state": ["SUBSTANTIVE"]})
        p = pd.DataFrame({"sentence_id": ["x"], "state": ["NO_CLAIM"], "classification_status": ["ok"]})
        with self.assertRaisesRegex(ValueError, "prediction columns"):
            merge_annotations(g, p)

    def test_baseline_only_is_in_recall_population(self):
        with tempfile.TemporaryDirectory() as d:
            lex = Path(d) / "custom.yaml"
            lex.write_text(
                'name: custom\nversion: "1"\nlanguage: en\nterms:\n  - id: agentic\n    pattern: agentic\n    category: general\n    tier: core\n'
            )
            f = Filing(source="x", filing_id="f", firm_id="a", firm_name="Acme", fiscal_year=2024)
            p = Pipeline(lexicon=lex, recall_random=10, recall_probe_cap=10)
            r = p.run_documents(
                [
                    Document(
                        filing=f,
                        language="en",
                        text="We deployed AI in our logistics systems. We launched agentic workflows for routing goods.",
                    )
                ]
            )
            base = r.sentences[r.sentences["raw_terms"] == ""]
            self.assertEqual(len(base), 1)
            self.assertIn(base.iloc[0].sentence_id, set(r.recall_pool.sentence_id))

    def test_dataset_digest_changes_with_result_data(self):
        f = Filing(source="x", filing_id="f", firm_id="a", firm_name="Acme", fiscal_year=2024)
        r = Pipeline().run_documents(
            [Document(filing=f, language="en", text="We deployed artificial intelligence in logistics.")]
        )
        h = r.dataset_digest()
        r.sentences.loc[0, "state"] = "WEAK"
        self.assertNotEqual(h, r.dataset_digest())


class CLIDatasetBindingTests(unittest.TestCase):
    def test_evaluate_rejects_mixed_corpora_and_predicted_columns(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        from aidisclose.cli import main

        def run(*args):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main(list(map(str, args)))

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for key in ("a", "b"):
                f = Filing(source="x", filing_id=key, firm_id=key, firm_name=key, fiscal_year=2024)
                text = f"We deployed artificial intelligence to improve logistics in 2024 for firm {key}.\n\nOur staff work in markets around the world."
                r = Pipeline(recall_random=10, recall_probe_cap=10).run_documents(
                    [Document(filing=f, language="en", text=text)]
                )
                r.save(d / key)
                run("sample", "--results", d / key, "--n", "10", "--out", d / f"{key}_sample.csv")
                sample = pd.read_csv(d / f"{key}_sample.csv", keep_default_na=False)
                source = pd.read_csv(d / f"{key}_sample_key.csv", keep_default_na=False)
                sample = sample.merge(source[["sentence_id", "state"]], on="sentence_id")
                sample["gold_state"] = sample.pop("state")
                sample.to_csv(d / f"{key}_gold.csv", index=False)
            run("sample-recall", "--results", d / "a", "--n", "10", "--out", d / "r.csv")
            recall = pd.read_csv(d / "r.csv", keep_default_na=False)
            recall["gold_state"] = "NO_CLAIM"
            recall.to_csv(d / "r_gold.csv", index=False)
            with self.assertRaisesRegex(SystemExit, "different datasets"):
                run(
                    "evaluate-recall",
                    "--gold",
                    d / "r_gold.csv",
                    "--key",
                    d / "r_key.csv",
                    "--main-gold",
                    d / "b_gold.csv",
                    "--main-key",
                    d / "b_sample_key.csv",
                    "--n-boot",
                    "3",
                )
            forged = pd.read_csv(d / "a_gold.csv", keep_default_na=False)
            forged["state"] = "NO_CLAIM"
            forged.to_csv(d / "forged.csv", index=False)
            with self.assertRaisesRegex(SystemExit, "prediction columns"):
                run(
                    "evaluate",
                    "--gold",
                    d / "forged.csv",
                    "--results",
                    d / "a",
                    "--key",
                    d / "a_sample_key.csv",
                    "--weighted",
                    "--cluster-by",
                    "none",
                    "--n-boot",
                    "3",
                )
            with self.assertRaisesRegex(SystemExit, "different datasets"):
                run(
                    "evaluate",
                    "--gold",
                    d / "a_gold.csv",
                    "--results",
                    d / "b",
                    "--key",
                    d / "a_sample_key.csv",
                    "--weighted",
                    "--cluster-by",
                    "none",
                    "--n-boot",
                    "3",
                )
