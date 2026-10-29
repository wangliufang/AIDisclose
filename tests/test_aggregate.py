import unittest

import pandas as pd
from helpers import FIXTURES  # noqa: F401

from aidisclose.aggregate import firm_year_panel, group_panel, state_mix
from aidisclose.industry import sic_to_nace_section
from aidisclose.pipeline import DOCUMENT_COLUMNS, SENTENCE_COLUMNS, Results


def doc(fid, firm, year, country, words, base, kw, sub, weak, no, status="ok", code="7372", scheme="SIC"):
    row = {c: "" for c in DOCUMENT_COLUMNS}
    row.update(
        filing_id=fid,
        firm_id=firm,
        firm_name=firm.upper(),
        fiscal_year=year,
        country=country,
        industry_code=code,
        industry_scheme=scheme,
        source="t",
        status=status,
        n_words=words,
        n_sentences=10,
        n_baseline_sentences=base,
        n_keyword_sentences=kw,
        n_candidates=kw,
        n_substantive=sub,
        n_weak=weak,
        n_no_claim=no,
        n_false_positive=0,
        extra="{}",
    )
    return row


def sent(sid, fid, state, conf=0.8):
    row = {c: "" for c in SENTENCE_COLUMNS}
    row.update(
        sentence_id=sid,
        filing_id=fid,
        state=state,
        classification_status="ok",
        confidence=conf,
        raw_terms="ai_acronym",
        reason="VAGUE",
    )
    return row


class AggregateTests(unittest.TestCase):
    def setUp(self):
        docs = pd.DataFrame(
            [
                doc("f1", "a", 2023, "US", 10000, 4, 5, 2, 1, 2),
                doc("f1b", "a", 2023, "US", 500, 1, 1, 0, 1, 0),  # shorter duplicate (e.g. 10-K/A)
                doc("f2", "b", 2023, "US", 20000, 2, 2, 0, 2, 0),
                doc("f3", "c", 2023, "DK", 5000, 0, 0, 0, 0, 0, code="G", scheme="NACE"),
                doc("f4", "d", 2023, "DK", 5000, 3, 3, 0, 0, 3, status="error"),
            ]
        )
        sents = pd.DataFrame(
            [
                sent("s1", "f1", "SUBSTANTIVE", 0.9),
                sent("s2", "f1", "SUBSTANTIVE", 0.7),
                sent("s3", "f1", "WEAK"),
                sent("s4", "f2", "WEAK"),
            ]
        )
        self.res = Results(sentences=sents, documents=docs)

    def test_firm_year_panel(self):
        p = firm_year_panel(self.res)
        self.assertEqual(len(p), 3)  # duplicate firm-year collapsed, error document dropped
        a = p[p["firm_id"] == "a"].iloc[0]
        self.assertEqual(a["filing_id"], "f1")
        self.assertEqual((a["keyword_any"], a["claim_any"], a["substantive_any"]), (1, 1, 1))
        self.assertAlmostEqual(a["keyword_per_10k_words"], 4.0)
        self.assertAlmostEqual(a["substantive_share"], 0.4)
        self.assertEqual(a["evidence"], "s1;s2")  # ordered by confidence
        self.assertEqual(a["nace_section"], "J")
        c = p[p["firm_id"] == "c"].iloc[0]
        self.assertEqual(c["nace_section"], "G")
        self.assertTrue(pd.isna(c["substantive_share"]))

    def test_group_panel(self):
        g = group_panel(firm_year_panel(self.res), by=["country", "fiscal_year"])
        us = g[g["country"] == "US"].iloc[0]
        self.assertEqual(us["n_firms"], 2)
        self.assertAlmostEqual(us["share_keyword"], 1.0)
        self.assertAlmostEqual(us["share_substantive"], 0.5)
        self.assertAlmostEqual(us["gap_keyword_substantive_pp"], 50.0)
        dk = g[g["country"] == "DK"].iloc[0]
        self.assertAlmostEqual(dk["share_keyword"], 0.0)

    def test_weighted_and_min_firms(self):
        p = firm_year_panel(self.res)
        g = group_panel(p, by=["country"], weight="n_words")
        us = g[g["country"] == "US"].iloc[0]
        self.assertAlmostEqual(us["share_substantive"], 10000 / 30000, places=4)
        self.assertEqual(len(group_panel(p, by=["country"], min_firms=2)), 1)

    def test_state_mix(self):
        mix = state_mix(self.res, by=["filing_id"])
        f1 = mix[mix["filing_id"] == "f1"]
        self.assertAlmostEqual(f1["share"].sum(), 1.0)


class IndustryTests(unittest.TestCase):
    def test_mapping(self):
        cases = {
            "7372": "J",
            "4213": "H",
            "6331": "K",
            "6798": "L",
            "2834": "C",
            "2711": "J",
            "0200": "A",
            "4911": "D",
            "4953": "E",
            "5812": "I",
            "8731": "M",
            "7361": "N",
            "9999": None,
            None: None,
            "xx": None,
        }
        for sic, nace in cases.items():
            self.assertEqual(sic_to_nace_section(sic), nace, sic)


if __name__ == "__main__":
    unittest.main()
