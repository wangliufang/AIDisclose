"""Regressions for term positions, lexicon identity and invalid dates."""

import os
import tempfile
import unittest
from pathlib import Path

import aidisclose
from aidisclose import Document, Filing, Pipeline, Results
from aidisclose.classify.base import Classifier
from aidisclose.lexicon import Lexicon, Term
from aidisclose.models import AdoptionState, Prediction, Reason
from aidisclose.sources.base import fiscal_year_from_period_end


class TestFeatureIntegrity(unittest.TestCase):
    def test_installed_package_when_requested(self):
        root = os.environ.get("AIDISCLOSE_EXPECT_INSTALL_ROOT")
        if root:
            self.assertTrue(Path(aidisclose.__file__).resolve().is_relative_to(Path(root).resolve()))

    def document(self):
        filing = Filing(source="x", filing_id="f", firm_id="x", firm_name="Acme", fiscal_year=2024)
        return Document(filing=filing, language="en", text="We deployed AIPX in our logistics warehouses in 2024.")

    def test_programmatic_lexicons_have_distinct_fingerprints(self):
        a = Lexicon("custom", "1", [Term("custom", "AIPX", "general")])
        b = Lexicon("custom", "1", [Term("custom", "BIPX", "general")])
        self.assertNotEqual(a.describe()["terms_sha256"], b.describe()["terms_sha256"])
        self.assertNotEqual(Pipeline(lexicon=a).fingerprint(), Pipeline(lexicon=b).fingerprint())

    def test_custom_term_spans_survive_save_load_and_relabel(self):
        lex = Lexicon("custom", "1", [Term("custom", "AIPX", "general")])
        original = Pipeline(lexicon=lex).run_documents([self.document()])
        self.assertEqual(original.sentences.state.tolist(), ["SUBSTANTIVE"])
        with tempfile.TemporaryDirectory() as td:
            original.save(td)
            loaded = Results.load(td)
            relabelled = Pipeline().relabel(loaded)
        self.assertEqual(relabelled.sentences.state.tolist(), original.sentences.state.tolist())
        self.assertEqual(relabelled.sentences.reason.tolist(), original.sentences.reason.tolist())
        changed = loaded.sentences.copy()
        changed.loc[0, "term_spans"] = "[[0, 99999]]"
        with self.assertRaisesRegex(ValueError, "invalid recorded term spans"):
            Pipeline().relabel(Results(changed, loaded.documents, loaded.provenance, loaded.recall_pool))

    def test_legacy_relabel_requires_matching_reconstructable_lexicon(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lex.yaml"
            path.write_text(
                'name: custom\nversion: "1"\nlanguage: en\nterms:\n  - id: custom\n    pattern: AIPX\n    category: general\n    tier: core\n'
            )
            pipe = Pipeline(lexicon=path)
            r = pipe.run_documents([self.document()])
            r.sentences = r.sentences.drop(columns="term_spans")
            r.provenance["lexicon"].pop("terms_sha256")  # Exercise a result without term digests.
            with self.assertRaisesRegex(ValueError, "original en lexicon"):
                Pipeline().relabel(r)
            self.assertEqual(pipe.relabel(r).sentences.state.tolist(), r.sentences.state.tolist())
            r.provenance["lexicon"]["sha256"] = ""
            with self.assertRaisesRegex(ValueError, "original en lexicon"):
                pipe.relabel(r)

    def test_invalid_period_ends_do_not_produce_fiscal_years(self):
        for period in ["2024-00-12", "2024-13-12", "2024-02-31", "2024/01/31", "2024-01"]:
            self.assertIsNone(fiscal_year_from_period_end(period), period)
        self.assertEqual(fiscal_year_from_period_end("2024-02-29"), 2023)
        self.assertEqual(fiscal_year_from_period_end("2024-02-29", "period_end_year"), 2024)

    def test_legacy_multilingual_uses_document_language_lexicon(self):
        class Stub(Classifier):
            name = "stub"

            def classify(self, candidates):
                return [Prediction(AdoptionState.WEAK, Reason.VAGUE, backend=self.name) for _ in candidates]

        de = Lexicon("de", "1", [Term("ki", "KI", "general", case_sensitive=True)], language="de")
        p = Pipeline(backend=Stub(), languages=["en", "de"], lexicons={"de": de})
        filing = Filing(source="x", filing_id="de1", firm_id="de", firm_name="Beispiel", fiscal_year=2024)
        result = p.run_documents([Document(filing=filing, language="de", text="Wir nutzen KI in unseren Betrieben.")])
        result.sentences = result.sentences.drop(columns="term_spans")
        self.assertEqual(p.candidates_from_results(result)[0].term_spans, [(11, 13)])
        result.documents.loc[0, "language"] = ""
        with self.assertRaisesRegex(ValueError, "needs each document.s language"):
            p.candidates_from_results(result)
