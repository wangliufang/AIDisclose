import unittest

import yaml
from helpers import fixture

from aidisclose.classify.rules import RuleClassifier, firm_alias_regex
from aidisclose.lexicon import Lexicon
from aidisclose.models import AdoptionState, Candidate, Reason

LEX = Lexicon.load("default", tiers=None)


def make(text, section="10-K:1", prev="", firm="Fictional Corp."):
    hits = LEX.find(text)
    return Candidate(
        filing_id="f",
        sentence_id="s",
        section=section,
        text=text,
        prev_text=prev,
        firm_name=firm,
        terms=hits.term_ids("valid"),
        term_spans=[(h.start, h.end) for h in hits.valid],
        filtered=hits.term_ids("filtered"),
    )


class RuleCaseTests(unittest.TestCase):
    """Developer-written codebook examples (fixtures/rule_cases.yaml)."""

    def test_cases(self):
        clf = RuleClassifier()
        cases = yaml.safe_load(fixture("rule_cases.yaml").read_text(encoding="utf-8"))
        failures = []
        for c in cases:
            p = clf.classify_one(
                make(c["text"], c.get("section", "10-K:1"), c.get("prev", ""), c.get("firm", "Fictional Corp."))
            )
            if p.state.value != c["state"] or ("reason" in c and p.reason.value != c["reason"]):
                failures.append(
                    f"{c['text']!r}: expected {c['state']}/{c.get('reason')} got {p.state.value}/{p.reason.value}"
                )
        self.assertEqual(failures, [])


class RuleBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.clf = RuleClassifier()

    def test_false_positive_without_valid_terms(self):
        p = self.clf.classify_one(make("Revenue from artificial insemination (AI) services rose 5%."))
        self.assertEqual((p.state, p.reason), (AdoptionState.NO_CLAIM, Reason.FALSE_POSITIVE))

    def test_details_are_auditable(self):
        p = self.clf.classify_one(make("We use machine learning to detect fraud in card payments."))
        self.assertEqual(p.state, AdoptionState.SUBSTANTIVE)
        self.assertIn("linked_action", p.details)
        self.assertIn("application", p.details)
        self.assertEqual(p.backend, "rules")
        self.assertTrue(0 < p.confidence < 1)

    def test_strongest_hit_wins(self):
        p = self.clf.classify_one(
            make("Competitors use AI, and we use machine learning for 80% of our pricing decisions.")
        )
        self.assertEqual(p.state, AdoptionState.SUBSTANTIVE)
        self.assertEqual(p.details["n_hits"], 2)

    def test_firm_alias(self):
        rx = firm_alias_regex("Northwind Logistics Inc.")
        self.assertTrue(rx.search("Northwind's AI routing engine"))
        rx2 = firm_alias_regex("First National Bank Corp.")
        self.assertTrue(rx2.search("First National deployed AI"))
        self.assertIsNone(firm_alias_regex(""))

    def test_risk_section_context(self):
        text = "We use AI in our products and services."
        self.assertEqual(self.clf.classify_one(make(text, "10-K:1")).reason, Reason.VAGUE)
        self.assertEqual(self.clf.classify_one(make(text, "10-K:1A")).reason, Reason.RISK_BOILERPLATE)

    def test_terms_located_when_spans_missing(self):
        c = make("We use machine learning to forecast demand at 85 depots.")
        c.term_spans = []
        self.assertEqual(self.clf.classify_one(c).state, AdoptionState.SUBSTANTIVE)

    def test_config(self):
        cfg = self.clf.config()
        self.assertEqual(cfg["backend"], "rules")
        self.assertIn("version", cfg)


class StressCaseTests(unittest.TestCase):
    """Illustrative sentence cases, including documented rule limitations."""

    def test_stress_cases(self):
        clf = RuleClassifier()
        cases = yaml.safe_load(fixture("stress_cases.yaml").read_text(encoding="utf-8"))
        unexpected, limitations_fixed = [], []
        for c in cases:
            p = clf.classify_one(make(c["text"], c["section"], firm="Example Holdings Inc."))
            ok = p.state.value == c["state"]
            if c.get("known_limitation"):
                if ok:
                    limitations_fixed.append(c["text"])
            elif not ok:
                unexpected.append(f"{c['text']!r}: expected {c['state']} got {p.state.value}/{p.reason.value}")
        self.assertEqual(unexpected, [])
        # If a known limitation starts passing, update the fixture to reflect current behaviour.
        self.assertEqual(limitations_fixed, [])


if __name__ == "__main__":
    unittest.main()
