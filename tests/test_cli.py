import contextlib
import io
import tempfile
import unittest
from importlib import resources
from pathlib import Path

import pandas as pd
from helpers import fixture

from aidisclose.cli import main


def run(*args) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(list(args))
    return buf.getvalue()


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.demo = Path(str(resources.files("aidisclose.demo").joinpath("data")))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_demo(self):
        text = run("demo", "--out", str(self.out / "demo"))
        self.assertIn("SYNTHETIC DEMO CORPUS", text)
        self.assertIn("NOT a validation", text)
        self.assertTrue((self.out / "demo" / "firm_year_panel.csv").exists())

    def test_02_workflow(self):
        res = self.out / "res"
        text = run(
            "run",
            "--source",
            "local",
            "--dir",
            str(self.demo),
            "--metadata",
            str(self.demo / "metadata.csv"),
            "--out",
            str(res),
            "-q",
        )
        self.assertIn("processed 20/21 documents", text)
        run(
            "list",
            "--source",
            "local",
            "--dir",
            str(self.demo),
            "--metadata",
            str(self.demo / "metadata.csv"),
            "--start",
            "2024",
            "--out",
            str(self.out / "filings.csv"),
        )
        self.assertEqual(len(pd.read_csv(self.out / "filings.csv")), 7)
        run("panel", "--results", str(res), "--level", "country-year", "--out", str(self.out / "cy.csv"), "-q")
        cy = pd.read_csv(self.out / "cy.csv")
        self.assertIn("share_substantive", cy.columns)
        run("sample", "--results", str(res), "--n", "30", "--min-per-stratum", "5", "--out", str(self.out / "ann.csv"))
        ann = pd.read_csv(self.out / "ann.csv")
        key = pd.read_csv(self.out / "ann_key.csv")
        self.assertNotIn("state", ann.columns)
        # simulate annotators who agree with the classifier
        ann["gold_state"] = key.set_index("sentence_id").loc[ann["sentence_id"], "state"].values
        ann.to_csv(self.out / "ann_done.csv", index=False)
        report = run(
            "evaluate",
            "--gold",
            str(self.out / "ann_done.csv"),
            "--results",
            str(res),
            "--key",
            str(self.out / "ann_key.csv"),
            "--weighted",
            "--n-boot",
            "50",
            "--out",
            str(self.out / "report.md"),
        )
        self.assertIn("Accuracy: 1.000", report)
        self.assertTrue((self.out / "report.json").exists())
        text = run("train", "--gold", str(self.out / "ann_done.csv"), "--model-out", str(self.out / "m.joblib"))
        self.assertIn("model trained", text)
        text = run(
            "run",
            "--source",
            "local",
            "--dir",
            str(self.demo),
            "--metadata",
            str(self.demo / "metadata.csv"),
            "--backend",
            "supervised",
            "--model-path",
            str(self.out / "m.joblib"),
            "--limit",
            "3",
            "--out",
            str(self.out / "res_sup"),
            "-q",
        )
        self.assertIn("processed 3/3", text)

    def test_03_eurostat(self):
        panel = pd.DataFrame(
            {
                "country": ["DK", "NL", "RO"],
                "fiscal_year": [2024] * 3,
                "n_firms": [8] * 3,
                "share_keyword": [1.0, 0.9, 0.8],
                "share_claim": [0.7, 0.6, 0.3],
                "share_substantive": [0.5, 0.4, 0.1],
            }
        )
        panel.to_csv(self.out / "panel.csv", index=False)
        text = run(
            "eurostat",
            "--panel",
            str(self.out / "panel.csv"),
            "--eurostat-file",
            str(fixture("eurostat_isoc_eb_ai_synthetic.json")),
            "--out",
            str(self.out / "eu.csv"),
        )
        self.assertIn("share_substantive", text)

    def test_04_lexicon(self):
        text = run("lexicon", "--test", "Our AI/ML platform", "-q")
        self.assertIn("ml_acronym", text)

    def test_05_version(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--version"])


if __name__ == "__main__":
    unittest.main()
