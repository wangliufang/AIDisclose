import unittest

from helpers import fixture  # noqa: F401  (sets sys.path)

from aidisclose.lexicon import Lexicon, conflicting_acronyms


class LexiconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = Lexicon.load("default", tiers=None)
        cls.core = Lexicon.load("default")

    def ids(self, text, lex=None, **kw):
        return (lex or self.lex).find(text, **kw).term_ids("valid")

    def test_core_terms(self):
        self.assertIn("artificial_intelligence", self.ids("We invest in Artificial Intelligence."))
        self.assertIn("machine_learning", self.ids("machine-learning models"))
        self.assertIn("large_language_model", self.ids("large language models"))
        self.assertIn("generative_ai", self.ids("generative AI tools"))

    def test_acronym_is_case_sensitive_and_bounded(self):
        self.assertEqual(self.ids("he said the plan was fair"), [])
        self.assertEqual(self.ids("We partner with OpenAI."), ["openai"])  # 'AI' inside a name is not the acronym
        self.assertIn("ai_acronym", self.ids("AI-powered search"))

    def test_requires_cooccurrence(self):
        self.assertEqual(self.ids("The ML division grew."), [])
        hits = self.lex.find("The ML division grew.")
        self.assertEqual(hits.term_ids("filtered"), ["ml_acronym"])
        self.assertIn("ml_acronym", self.ids("our AI/ML platform"))

    def test_negative_context(self):
        hits = self.lex.find("Revenue from artificial insemination (AI) services rose.")
        self.assertEqual(hits.term_ids("valid"), [])
        self.assertEqual(hits.term_ids("filtered"), ["ai_acronym"])
        # an unambiguous AI term in the same sentence overrides the negative context
        self.assertIn("ai_acronym", self.ids("We use AI and machine learning to schedule artificial insemination."))

    def test_disabled_terms(self):
        self.assertEqual(self.ids("AI programs grew 5%.", disabled_terms={"ai_acronym"}), [])

    def test_tiers(self):
        self.assertIn("chatbot", self.ids("our chatbot"))
        self.assertEqual(self.ids("our chatbot", lex=self.core), [])
        self.assertTrue(self.core.version.endswith("[core]"))

    def test_conflicting_acronyms(self):
        self.assertEqual(conflicting_acronyms('We provide artificial insemination ("AI") services.'), {"ai_acronym"})
        self.assertEqual(conflicting_acronyms("We use artificial intelligence (AI)."), set())
        mixed = 'artificial intelligence ("AI") ... aromatase inhibitor (AI)'
        self.assertEqual(conflicting_acronyms(mixed), set())

    def test_describe_and_custom_yaml(self):
        desc = self.lex.describe()
        self.assertEqual(len(desc["sha256"]), 64)
        custom = Lexicon.from_yaml_bytes(
            b"name: t\nversion: '9'\nterms:\n  - {id: robo, pattern: 'robo-?advis\\w+', category: general}\n",
            tiers=None,
        )
        self.assertEqual(custom.find("Our robo-advisor").term_ids(), ["robo"])
        with self.assertRaises(ValueError):
            Lexicon.from_yaml_bytes(b"terms:\n  - {id: x}\n")
        with self.assertRaises(FileNotFoundError):
            Lexicon.load("no-such-lexicon")

    def test_baseline(self):
        base = Lexicon.load("keyword-baseline-4", tiers=None)
        # the plain baseline does not disambiguate
        self.assertEqual(base.find("Revenue from artificial insemination (AI) services").term_ids(), ["ai_acronym"])


if __name__ == "__main__":
    unittest.main()
