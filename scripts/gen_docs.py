"""Generate the reference pages of the documentation site.

Writes docs/cli.md (from the argparse definitions), docs/api.md (from the
docstrings of the public API), and copies the codebook into docs/ so that
mkdocs can render it.

    python scripts/gen_docs.py && mkdocs build --strict
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

API = [
    ("Pipeline and results", "aidisclose.pipeline", ["Pipeline", "Results", "DocumentOutcome"]),
    ("Indicators", "aidisclose.aggregate", ["firm_year_panel", "group_panel", "state_mix", "read_industry_map"]),
    (
        "Sources",
        "aidisclose.sources",
        ["Source", "HttpClient", "EdgarSource", "XbrlFilingsSource", "LocalSource", "register_source"],
    ),
    ("EDGAR helpers", "aidisclose.sources.edgar", ["country_from_edgar", "parse_filing_index"]),
    (
        "Text",
        "aidisclose.text",
        [
            "extract_text",
            "html_to_text",
            "split_sentences",
            "split_sections",
            "split_items",
            "split_annual_report",
            "SectionFilter",
            "detect_language",
        ],
    ),
    ("Lexicons", "aidisclose.lexicon", ["Lexicon", "conflicting_acronyms"]),
    ("Backends", "aidisclose.classify", ["get_classifier", "RuleClassifier"]),
    ("LLM backend", "aidisclose.classify.llm", ["LLMClassifier", "parse_response"]),
    ("Supervised backend", "aidisclose.classify.supervised", ["SupervisedClassifier"]),
    ("Transformer backend (experimental)", "aidisclose.classify.transformer", ["TransformerClassifier"]),
    (
        "Validation: sampling and metrics",
        "aidisclose.validate",
        ["sample_for_annotation", "blind", "classification_report", "bootstrap_report", "cohen_kappa"],
    ),
    (
        "Validation: annotation files and agreement",
        "aidisclose.validate.annotation",
        [
            "write_annotation_file",
            "read_annotation_file",
            "to_label_studio",
            "from_label_studio",
            "agreement_report",
            "fleiss_kappa",
            "krippendorff_alpha_nominal",
        ],
    ),
    ("Validation: recall audit", "aidisclose.validate.recall", ["sample_recall", "estimate_recall"]),
    (
        "Validation: external benchmarks",
        "aidisclose.validate.external",
        ["eurostat_ai_share", "eurostat_ai_share_by_industry", "load_benchmark", "compare_with_external"],
    ),
    ("Quality checks", "aidisclose.qa", ["quality_report"]),
    ("Data model", "aidisclose.models", ["AdoptionState", "Reason", "Filing", "Document", "Candidate", "Prediction"]),
]


def _doc(obj) -> str:
    text = inspect.getdoc(obj) or ""
    # numpy-style section underlines render as horizontal rules in Markdown
    text = re.sub(r"\n([A-Z][A-Za-z ]+)\n-{3,}\n", r"\n**\1**\n\n", "\n" + text).lstrip("\n")
    return text.replace("``", "`")


def _sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def api_page() -> str:
    lines = ["# Python API", "", "Generated from the docstrings by `scripts/gen_docs.py`.", ""]
    for title, mod_name, names in API:
        mod = importlib.import_module(mod_name)
        lines += [f"## {title}", "", f"`{mod_name}`", ""]
        for name in names:
            obj = getattr(mod, name)
            kind = "class" if inspect.isclass(obj) else "function"
            sig = "" if inspect.isclass(obj) and issubclass(obj, (str,)) else _sig(obj)
            lines += [f"### `{name}`", "", f"```python\n{kind} {name}{sig}\n```", "", _doc(obj), ""]
            if inspect.isclass(obj):
                for mname, member in inspect.getmembers(obj):
                    if mname.startswith("_") or not (inspect.isfunction(member) or inspect.ismethod(member)):
                        continue
                    if member.__qualname__.split(".")[0] != obj.__name__:
                        continue
                    doc = _doc(member)
                    if doc:
                        lines += [f"**`{name}.{mname}{_sig(member)}`**", "", doc, ""]
    return "\n".join(lines)


def cli_page() -> str:
    from aidisclose.cli import build_parser

    parser = build_parser()
    lines = [
        "# Command line",
        "",
        "Generated from `aidisclose --help` by `scripts/gen_docs.py`.",
        "",
        "```text",
        parser.format_help().strip(),
        "```",
        "",
    ]
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for name, p in sub.choices.items():
        lines += [f"## `aidisclose {name}`", "", "```text", p.format_help().strip(), "```", ""]
    return "\n".join(lines)


def main() -> None:
    (DOCS / "api.md").write_text(api_page(), encoding="utf-8")
    (DOCS / "cli.md").write_text(cli_page(), encoding="utf-8")
    codebook = (ROOT / "annotation" / "codebook.md").read_text(encoding="utf-8")
    (DOCS / "codebook.md").write_text(codebook, encoding="utf-8")
    print("wrote docs/api.md, docs/cli.md, docs/codebook.md")


if __name__ == "__main__":
    main()
