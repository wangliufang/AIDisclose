import unittest

from helpers import fixture

from aidisclose.text import detect_language, html_to_text, split_10k_items, split_sentences
from aidisclose.text.extract import extract_text


class ExtractTests(unittest.TestCase):
    def test_html_cleaning(self):
        text = html_to_text(fixture("sample_10k.htm").read_bytes())
        self.assertNotIn("Hidden AI facts", text)  # ix:header and display:none removed
        self.assertNotIn("var ai", text)  # scripts removed
        self.assertNotIn("2,310", text)  # numeric financial table dropped
        self.assertIn("We use machine learning models to forecast freight demand", text)
        self.assertIn("Item 1A.", text)  # text tables (TOC) are kept

    def test_ixbrl_hidden_facts_removed(self):
        text = extract_text(fixture("esef_report.xhtml").read_bytes(), filename="r.xhtml")
        self.assertNotIn("AI AI AI", text)
        self.assertIn("We use machine learning to set prices", text)
        self.assertIn("4,210 million", text)  # visible tagged facts stay

    def test_plain_text(self):
        self.assertEqual(extract_text(b"<DOCUMENT>\nHello   world\n</DOCUMENT>"), "Hello world")


class SegmentTests(unittest.TestCase):
    def test_abbreviations(self):
        text = "Acme Inc. operates in the U.S. and Canada. It sells e.g. tools. Item 1. Business"
        sents = [s.text for s in split_sentences(text)]
        self.assertEqual(
            sents, ["Acme Inc. operates in the U.S. and Canada.", "It sells e.g. tools.", "Item 1. Business"]
        )

    def test_blocks_are_not_joined(self):
        sents = [s.text for s in split_sentences("Heading without period\nFirst sentence. Second one.")]
        self.assertEqual(sents, ["Heading without period", "First sentence.", "Second one."])

    def test_numbers(self):
        sents = [s.text for s in split_sentences("Revenue rose 4.5% to $1.2 billion. Costs fell.")]
        self.assertEqual(len(sents), 2)

    def test_long_sentence_split(self):
        long = "; ".join([f"clause number {i} with several words" for i in range(60)]) + "."
        self.assertGreater(len(split_sentences(long, max_chars=300)), 1)


class SectionTests(unittest.TestCase):
    def test_10k_items_skip_toc(self):
        text = html_to_text(fixture("sample_10k.htm").read_bytes())
        secs = split_10k_items(text)
        self.assertIn("10-K:1", secs)
        self.assertIn("10-K:1A", secs)
        self.assertIn("10-K:7", secs)
        self.assertIn("forecast freight demand", secs["10-K:1"])  # body, not the table of contents
        self.assertIn("Competitors may use AI", secs["10-K:1A"])
        self.assertNotIn("Competitors may use AI", secs["10-K:1"])
        self.assertIn("reduced fuel consumption", secs["10-K:7"])

    def test_non_10k(self):
        self.assertEqual(split_10k_items("Just a letter.\nNothing else."), {"full": "Just a letter.\nNothing else."})


class LanguageTests(unittest.TestCase):
    def test_english_and_german(self):
        en = "We are a leading provider of logistics services and we operate in the United States and in Canada with our partners."
        de = "Wir sind ein führender Anbieter von Logistikdienstleistungen und sind in Deutschland und in Österreich mit unseren Partnern tätig, die auch für die Region wichtig sind."
        self.assertEqual(detect_language(en * 3)[0], "en")
        self.assertEqual(detect_language(de * 3)[0], "de")
        self.assertIsNone(detect_language("AI ML")[0])


class ReflowTests(unittest.TestCase):
    def test_line_per_div_layout_is_rejoined(self):
        lines = [
            "In 2024 we deployed a machine learning model that",
            "optimises replenishment for all 42 stores, and our",
            "generative AI assistant now answers 30% of customer",
            "service enquiries.",
        ]
        html = (
            "<html><body>"
            + "".join(
                f'<div style="position:absolute;top:{100 + i * 14}px">{line}</div>' for i, line in enumerate(lines)
            )
            + "</body></html>"
        )
        sents = [s.text for s in split_sentences(html_to_text(html))]
        self.assertEqual(len(sents), 1)
        self.assertTrue(sents[0].endswith("customer service enquiries."))

    def test_page_break_inside_sentence(self):
        html = (
            "<p>We use machine learning models to forecast demand at</p><hr/><p>23</p>"
            "<p><a href='#toc'>Table of Contents</a></p><p>85 distribution centers, which reduced inventory by 12%.</p>"
        )
        self.assertEqual(
            html_to_text(html),
            "We use machine learning models to forecast demand at 85 distribution centers, which reduced inventory by 12%.",
        )

    def test_headings_items_and_bullets_are_not_merged(self):
        text = html_to_text(
            "<p>ITEM 1A. RISK FACTORS</p><p>competitors may use AI.</p><h2>Our strategy</h2><p>We grew 4%.</p>"
            "<p>Our tools include</p><p>• chatbots for customer service</p>"
        )
        self.assertIn("ITEM 1A. RISK FACTORS\n", text)
        self.assertIn("Our strategy\n", text)
        self.assertIn("\n• chatbots", text)

    def test_hyphenated_line_break(self):
        self.assertEqual(
            html_to_text("<p>We apply machine-</p><p>learning to pricing.</p>"), "We apply machine-learning to pricing."
        )

    def test_running_headers_removed(self):
        page = "<p>Acme Group Annual Report 2024</p><p>Revenue grew strongly in every region this year.</p><p>17</p>"
        text = html_to_text(page * 5)
        self.assertNotIn("Acme Group Annual Report 2024", text)
        self.assertNotIn("17", text)
        self.assertEqual(text.count("Revenue grew strongly"), 5)

    def test_reflow_can_be_disabled(self):
        html = "<div>We use AI to</div><div>route trucks.</div>"
        self.assertEqual(len(html_to_text(html, reflow=False).split("\n\n")), 2)
        self.assertEqual(html_to_text(html), "We use AI to route trucks.")

    def test_embedded_images_are_stripped(self):
        html = '<p>We use AI.</p><img src="data:image/png;base64,' + "A" * 10000 + '"/>'
        self.assertEqual(html_to_text(html), "We use AI.")

    def test_pdf_reflow(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("pypdf not installed")
        from aidisclose.text.extract import postprocess

        page_text = "Our underwriting platform uses machine\nlearning to price 60% of new policies.\n12\n"
        self.assertEqual(
            postprocess(page_text), "Our underwriting platform uses machine learning to price 60% of new policies."
        )


if __name__ == "__main__":
    unittest.main()
