import tempfile
import unittest
from importlib import resources
from pathlib import Path

import pandas as pd
from helpers import fixture

from aidisclose import LocalSource, Pipeline, Results, firm_year_panel
from aidisclose.models import Document, Filing
from aidisclose.sources.base import Source


def demo_dir() -> Path:
    return Path(str(resources.files("aidisclose.demo").joinpath("data")))


class DemoPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = demo_dir()
        cls.src = LocalSource(d, metadata=d / "metadata.csv")
        cls.res = Pipeline().run(cls.src.list_filings(), source=cls.src)
        cls.gold = pd.read_csv(d / "demo_gold.csv")

    def test_documents(self):
        docs = self.res.documents
        self.assertEqual(len(docs), 21)
        self.assertEqual((docs["status"] == "ok").sum(), 20)
        german = docs[docs["firm_id"] == "rheinwerk"].iloc[0]
        self.assertEqual(german["status"], "skipped_non_english")
        self.assertEqual(german["language"], "de")

    def test_every_designed_sentence_is_found(self):
        m = self.gold.merge(self.res.sentences, on=["firm_id", "fiscal_year", "text"], how="left")
        self.assertEqual(int(m["state"].isna().sum()), 0)

    def test_document_level_acronym_definition(self):
        s = self.res.sentences
        blue = s[(s["firm_id"] == "bluefield") & s["text"].str.startswith("AI programs")]
        self.assertTrue(len(blue) >= 1)
        self.assertTrue((blue["reason"] == "FALSE_POSITIVE").all())

    def test_hidden_ixbrl_not_counted(self):
        s = self.res.sentences
        self.assertFalse(s["text"].str.contains("hidden tagged facts").any())

    def test_sections_recorded(self):
        s = self.res.sentences
        self.assertTrue(set(s[s["firm_id"] == "harbor"]["section"]) <= {"10-K:1", "10-K:1A", "10-K:7"})
        # Free-form annual reports are split into parts.
        self.assertTrue(set(s[s["firm_id"] == "kestrel"]["section"]) <= {"AR:front", "AR:management", "AR:risk"})

    def test_20f_demo_document(self):
        s = self.res.sentences
        t = s[s["firm_id"] == "tasman"]
        self.assertTrue(set(t["section"]) <= {"20-F:3", "20-F:4", "20-F:5"})
        rag = t[t["text"].str.contains("RAG rating")]
        self.assertEqual(rag["reason"].tolist(), ["FALSE_POSITIVE"])

    def test_provenance(self):
        prov = self.res.provenance
        for key in ("aidisclose_version", "lexicon", "baseline_lexicon", "classifier", "created_utc"):
            self.assertIn(key, prov)
        self.assertEqual(len(prov["lexicon"]["sha256"]), 64)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.res.save(tmp)
            again = Results.load(tmp)
            self.assertEqual(len(again.sentences), len(self.res.sentences))
            p1 = firm_year_panel(self.res)
            p2 = firm_year_panel(again)
            pd.testing.assert_frame_equal(p1.reset_index(drop=True), p2.reset_index(drop=True), check_dtype=False)

    def test_evidence(self):
        ev = self.res.evidence(firm_id="northwind", fiscal_year=2023)
        self.assertEqual(len(ev), 2)
        self.assertTrue((ev["state"] == "SUBSTANTIVE").all())

    def test_section_filter(self):
        pipe = Pipeline(sections=["10-K:7"])
        f = [x for x in self.src.list_filings() if x.firm_id == "northwind" and x.fiscal_year == 2023]
        res = pipe.run(f, source=self.src)
        self.assertEqual(set(res.sentences["section"]), {"10-K:7"})


class ErrorHandlingTests(unittest.TestCase):
    def test_failures_are_recorded(self):
        class Broken(Source):
            name = "broken"

            def list_filings(self, **q):
                return [Filing(source="broken", filing_id="b1", firm_id="b", firm_name="B")]

            def fetch_document(self, filing):
                raise RuntimeError("network down")

        src = Broken()
        res = Pipeline().run(src.list_filings(), source=src)
        self.assertEqual(res.documents.iloc[0]["status"], "error")
        self.assertIn("network down", res.documents.iloc[0]["error"])

    def test_run_documents(self):
        f = Filing(source="x", filing_id="x1", firm_id="x", firm_name="Xylo Corp.", fiscal_year=2024)
        text = (
            "Xylo Corp. makes industrial pumps for customers in many countries around the world. "
            "We use machine learning to predict pump failures at 300 customer sites. "
            "Our engineers are among the best in the industry and we are proud of them."
        )
        res = Pipeline().run_documents([Document(filing=f, text=text)])
        self.assertEqual(res.documents.iloc[0]["n_substantive"], 1)
        self.assertEqual(res.documents.iloc[0]["n_baseline_sentences"], 1)

    def test_english_only_can_be_disabled(self):
        f = Filing(source="x", filing_id="x1", firm_id="x", firm_name="X")
        doc = Document(filing=f, text=fixture("sample_10k.htm").read_text(), language="xx")
        self.assertEqual(Pipeline().run_documents([doc]).documents.iloc[0]["status"], "skipped_non_english")
        self.assertEqual(
            Pipeline(english_only=False).run_documents([doc]).documents.iloc[0]["status"], "skipped_no_lexicon"
        )


if __name__ == "__main__":
    unittest.main()
