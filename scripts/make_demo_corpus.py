"""Generate the synthetic demo corpus shipped with AIDisclose.

All companies, people and numbers in this corpus are FICTIONAL. The corpus
exists so that the software can be installed and exercised offline in a few
minutes; it is not data and must not be used to draw empirical conclusions.

Each AI-related sentence is written together with the label it was designed to
illustrate (``demo_gold.csv``). Because the same developers wrote the sentences,
the labels and the rule backend, agreement on this corpus says nothing about
real-world accuracy; use ``aidisclose sample`` and human annotation for that.

Run from the repository root:  python scripts/make_demo_corpus.py
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "aidisclose" / "demo" / "data"

FILLER = {
    "business": [
        "{name} was founded in 1987 and is headquartered in {city}.",
        "We employed approximately {emp} people at the end of the fiscal year.",
        "Our revenue is generated primarily from {line}.",
        "We compete on the basis of price, service quality and reliability.",
        "Seasonal demand typically peaks in the fourth quarter.",
        "We operate through two reportable segments.",
    ],
    "risk": [
        "Our results of operations may fluctuate significantly from period to period.",
        "We depend on key personnel, and the loss of their services could harm our business.",
        "Changes in interest rates could adversely affect our financing costs.",
        "We are exposed to the credit risk of our counterparties.",
    ],
    "mdna": [
        "Revenue increased compared with the prior year, driven by higher volumes.",
        "Operating expenses grew more slowly than revenue.",
        "Cash flows from operating activities remained positive.",
        "We believe our existing liquidity is sufficient for the next twelve months.",
    ],
}

# (firm, country, scheme, code, form, city, employees, business line)
FIRMS = {
    "northwind": (
        "Northwind Logistics Inc.",
        "US",
        "SIC",
        "4213",
        "10-K",
        "Columbus, Ohio",
        "8,200",
        "freight transportation services",
    ),
    "harbor": (
        "Harbor Mutual Insurance Co.",
        "US",
        "SIC",
        "6331",
        "10-K",
        "Hartford, Connecticut",
        "3,100",
        "property and casualty insurance premiums",
    ),
    "bluefield": (
        "Bluefield Agritech Corp.",
        "US",
        "SIC",
        "0200",
        "10-K",
        "Des Moines, Iowa",
        "640",
        "livestock genetics and breeding services",
    ),
    "kestrel": (
        "Kestrel Software plc",
        "GB",
        "NACE",
        "J",
        "annual-report",
        "Leeds",
        "1,450",
        "subscriptions to our workflow software",
    ),
    "vantage": (
        "Vantage Retail Group A/S",
        "DK",
        "NACE",
        "G",
        "annual-report",
        "Aarhus",
        "5,600",
        "sales in our department stores and online",
    ),
    "tasman": (
        "Tasman Freight Ltd",
        "AU",
        "NACE",
        "H",
        "20-F",
        "Sydney",
        "4,700",
        "container freight forwarding and customs brokerage",
    ),
    "crescent": (
        "Crescent Pharma N.V.",
        "NL",
        "NACE",
        "C",
        "annual-report",
        "Leiden",
        "2,300",
        "sales of oncology medicines",
    ),
}

# AI sentences per firm-year: (section, text, gold_state, gold_reason)
AI = {
    ("northwind", 2022): [
        (
            "business",
            "We plan to use artificial intelligence to optimize delivery routes in the coming years.",
            "WEAK",
            "FORWARD_LOOKING",
        ),
        ("business", "Artificial intelligence is transforming the logistics industry.", "NO_CLAIM", "MARKET_TREND"),
        ("risk", "Competitors with greater resources may use AI to offer lower prices.", "NO_CLAIM", "THIRD_PARTY"),
    ],
    ("northwind", 2023): [
        (
            "business",
            "In 2023 we deployed an AI-powered routing engine across 120 distribution centers.",
            "SUBSTANTIVE",
            "DEPLOYMENT",
        ),
        (
            "mdna",
            "Our AI-powered routing engine reduced fuel consumption by 7% compared with 2022.",
            "SUBSTANTIVE",
            "DEPLOYMENT",
        ),
        (
            "risk",
            "If our AI models contain errors, our business could be adversely affected.",
            "WEAK",
            "RISK_BOILERPLATE",
        ),
        ("business", "Artificial intelligence is transforming the logistics industry.", "NO_CLAIM", "MARKET_TREND"),
    ],
    ("northwind", 2024): [
        (
            "business",
            "Our AI-powered routing engine now plans 95% of daily delivery routes.",
            "SUBSTANTIVE",
            "DEPLOYMENT",
        ),
        (
            "business",
            "We use machine learning models to forecast freight demand at the depot level.",
            "SUBSTANTIVE",
            "DEPLOYMENT",
        ),
        (
            "risk",
            "Our use of artificial intelligence may result in reputational harm or liability.",
            "WEAK",
            "RISK_BOILERPLATE",
        ),
        (
            "risk",
            "We are subject to emerging laws governing the use of artificial intelligence.",
            "NO_CLAIM",
            "REGULATORY",
        ),
    ],
    ("harbor", 2022): [
        ("business", "Insurtech start-ups are using machine learning to price policies.", "NO_CLAIM", "THIRD_PARTY"),
        ("risk", "Regulators have issued guidance on the use of AI in insurance pricing.", "NO_CLAIM", "REGULATORY"),
    ],
    ("harbor", 2023): [
        (
            "business",
            "We are piloting a machine learning tool for claims triage in two regions.",
            "WEAK",
            "FORWARD_LOOKING",
        ),
        ("business", "We intend to invest in AI capabilities over the coming years.", "WEAK", "FORWARD_LOOKING"),
        ("risk", "Regulators have issued guidance on the use of AI in insurance pricing.", "NO_CLAIM", "REGULATORY"),
    ],
    ("harbor", 2024): [
        (
            "business",
            "Machine learning models developed by our data science team now price approximately 60% of new personal auto policies.",
            "SUBSTANTIVE",
            "DEPLOYMENT",
        ),
        (
            "business",
            "In 2024 we launched Harbor Assist, a generative AI assistant that helps policyholders file claims online.",
            "SUBSTANTIVE",
            "PRODUCT",
        ),
        (
            "risk",
            "If our AI models contain errors or biases, our business could be adversely affected.",
            "WEAK",
            "RISK_BOILERPLATE",
        ),
        (
            "risk",
            "We are subject to emerging laws governing the use of artificial intelligence.",
            "NO_CLAIM",
            "REGULATORY",
        ),
    ],
    ("bluefield", 2022): [
        (
            "business",
            'Our breeding services rely on artificial insemination ("AI") using semen from proven sires.',
            "NO_CLAIM",
            "FALSE_POSITIVE",
        ),
        ("business", "AI programs accounted for 64% of our service revenue.", "NO_CLAIM", "FALSE_POSITIVE"),
        ("mdna", "Revenue from AI services increased 5% due to higher herd counts.", "NO_CLAIM", "FALSE_POSITIVE"),
    ],
    ("bluefield", 2023): [
        (
            "business",
            'Our breeding services rely on artificial insemination ("AI") using semen from proven sires.',
            "NO_CLAIM",
            "FALSE_POSITIVE",
        ),
        ("business", "AI programs accounted for 66% of our service revenue.", "NO_CLAIM", "FALSE_POSITIVE"),
        ("mdna", "Revenue from AI services increased 4% due to higher herd counts.", "NO_CLAIM", "FALSE_POSITIVE"),
    ],
    ("bluefield", 2024): [
        (
            "business",
            'Our breeding services rely on artificial insemination ("AI") using semen from proven sires.',
            "NO_CLAIM",
            "FALSE_POSITIVE",
        ),
        ("business", "AI programs accounted for 67% of our service revenue.", "NO_CLAIM", "FALSE_POSITIVE"),
        (
            "business",
            "We are exploring whether computer vision could help detect lameness in dairy herds.",
            "WEAK",
            "FORWARD_LOOKING",
        ),
    ],
    ("kestrel", 2022): [
        (
            "business",
            "Our search feature uses machine learning to rank documents by relevance for over 9,000 customers.",
            "SUBSTANTIVE",
            "PRODUCT",
        ),
        ("business", "We hold 12 patents related to natural language processing.", "SUBSTANTIVE", "INVESTMENT"),
    ],
    ("kestrel", 2023): [
        (
            "business",
            "In 2023 we launched Kestrel Copilot, a generative AI assistant that drafts workflow documents.",
            "SUBSTANTIVE",
            "PRODUCT",
        ),
        (
            "business",
            "Our search feature uses machine learning to rank documents by relevance for over 10,500 customers.",
            "SUBSTANTIVE",
            "PRODUCT",
        ),
        ("risk", "Our use of generative AI may expose us to intellectual property claims.", "WEAK", "RISK_BOILERPLATE"),
    ],
    ("kestrel", 2024): [
        (
            "business",
            "Kestrel Copilot, our generative AI assistant, drafted 1.2 million documents for customers during the year.",
            "SUBSTANTIVE",
            "PRODUCT",
        ),
        (
            "business",
            "We acquired Lexora Ltd., a natural language processing start-up, for $38 million.",
            "SUBSTANTIVE",
            "INVESTMENT",
        ),
        (
            "risk",
            "The EU AI Act imposes new obligations on providers of general-purpose AI models.",
            "NO_CLAIM",
            "REGULATORY",
        ),
    ],
    ("vantage", 2022): [
        ("business", "Artificial intelligence will reshape retail over the next decade.", "NO_CLAIM", "MARKET_TREND"),
        ("business", "We are an AI-first retailer.", "WEAK", "VAGUE"),
    ],
    ("vantage", 2023): [
        ("business", "Artificial intelligence will reshape retail over the next decade.", "NO_CLAIM", "MARKET_TREND"),
        ("business", "We are an AI-first retailer.", "WEAK", "VAGUE"),
        ("business", "We use AI across our business.", "WEAK", "VAGUE"),
        ("business", "We expect generative AI to create significant opportunities for us.", "WEAK", "FORWARD_LOOKING"),
    ],
    ("vantage", 2024): [
        ("business", "Artificial intelligence will reshape retail over the next decade.", "NO_CLAIM", "MARKET_TREND"),
        ("business", "We are an AI-first retailer.", "WEAK", "VAGUE"),
        ("business", "We use AI across our business.", "WEAK", "VAGUE"),
        ("business", "AI is central to our strategy.", "WEAK", "VAGUE"),
        (
            "business",
            "We are exploring how generative AI could improve productivity across the organization.",
            "WEAK",
            "FORWARD_LOOKING",
        ),
    ],
    ("crescent", 2022): [
        (
            "business",
            "Our lead product is an aromatase inhibitor (AI) for hormone receptor-positive breast cancer.",
            "NO_CLAIM",
            "FALSE_POSITIVE",
        ),
        (
            "business",
            "We do not currently use artificial intelligence in our research programs.",
            "NO_CLAIM",
            "NEGATED",
        ),
    ],
    ("crescent", 2023): [
        (
            "business",
            "Our lead product is an aromatase inhibitor (AI) for hormone receptor-positive breast cancer.",
            "NO_CLAIM",
            "FALSE_POSITIVE",
        ),
        ("business", "We have begun evaluating machine learning for target identification.", "WEAK", "FORWARD_LOOKING"),
    ],
    ("crescent", 2024): [
        (
            "business",
            "We use machine learning models to prioritize compounds in our drug discovery pipeline.",
            "SUBSTANTIVE",
            "DEPLOYMENT",
        ),
        (
            "business",
            "Our machine learning platform screened 2.4 million virtual compounds during 2024.",
            "SUBSTANTIVE",
            "DEPLOYMENT",
        ),
    ],
}

AI[("tasman", 2023)] = [
    (
        "business",
        "We are piloting AI agents to answer shipper enquiries in our Sydney contact centre.",
        "WEAK",
        "FORWARD_LOOKING",
    ),
    (
        "risk",
        "The development and use of AI presents risks that may result in reputational harm.",
        "WEAK",
        "RISK_BOILERPLATE",
    ),
    (
        "mdna",
        "We cannot predict how artificial intelligence will affect demand for freight forwarding.",
        "NO_CLAIM",
        "MARKET_TREND",
    ),
]
AI[("tasman", 2024)] = [
    (
        "business",
        "In 2024 our agentic AI dispatch system scheduled 70% of container pickups.",
        "SUBSTANTIVE",
        "DEPLOYMENT",
    ),
    (
        "business",
        "We license large language models from an external vendor and use them to draft customs declarations.",
        "SUBSTANTIVE",
        "DEPLOYMENT",
    ),
    (
        "risk",
        "Progress of the dispatch roll-out is reported to the board using a RAG rating.",
        "NO_CLAIM",
        "FALSE_POSITIVE",
    ),
]

GERMAN_REPORT = """Die Rheinwerk Maschinenbau AG ist ein Hersteller von Werkzeugmaschinen mit Sitz in Köln.
Im Geschäftsjahr 2023 setzten wir künstliche Intelligenz (KI) in der Qualitätsprüfung ein.
Der Umsatz stieg gegenüber dem Vorjahr um 4 Prozent. Die Zahl der Mitarbeiter lag bei 1.900.
Wir erwarten für das kommende Jahr eine stabile Entwicklung der Nachfrage in Europa und Asien.
Die Investitionen in die Digitalisierung der Produktion wurden fortgesetzt und werden auch im nächsten Jahr weitergeführt."""


def p(text: str) -> str:
    return f"<p>{html.escape(text)}</p>\n"


def build_10k(key: str, year: int) -> tuple[str, list[tuple[str, str, str, str]]]:
    name, country, scheme, code, form, city, emp, line = FIRMS[key]
    ai = AI[(key, year)]
    fill = {k: [s.format(name=name, city=city, emp=emp, line=line) for s in v] for k, v in FILLER.items()}
    body = [f"<html><head><title>{html.escape(name)} Form 10-K {year}</title></head><body>\n"]
    body.append(p("SYNTHETIC DEMO DOCUMENT - fictional company, for software testing only."))
    body.append(
        p(f"UNITED STATES SECURITIES AND EXCHANGE COMMISSION. FORM 10-K. Fiscal year ended December 31, {year}.")
    )
    body.append(
        "<table><tr><td>Item 1.</td><td>Business</td><td>3</td></tr><tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr>"
        "<tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>21</td></tr>"
        "<tr><td>Item 8.</td><td>Financial Statements</td><td>35</td></tr></table>\n"
    )
    sections = [
        ("1", "Business", "business"),
        ("1A", "Risk Factors", "risk"),
        ("7", "Management's Discussion and Analysis of Financial Condition and Results of Operations", "mdna"),
    ]
    for item, title, kind in sections:
        body.append(f"<div><b>Item {item}. {html.escape(title)}</b></div>\n")
        sents = list(fill[kind])
        ai_here = [a[1] for a in ai if a[0] == kind]
        mixed = []
        for i, s in enumerate(sents):
            mixed.append(s)
            if i < len(ai_here):
                mixed.append(ai_here[i])
        mixed.extend(ai_here[len(sents) :])
        body.append(p(" ".join(mixed[: len(mixed) // 2])))
        body.append(p(" ".join(mixed[len(mixed) // 2 :])))
    body.append("<div><b>Item 8. Financial Statements and Supplementary Data</b></div>\n")
    body.append(
        "<table><tr><td>Revenue</td><td>1,204</td><td>1,150</td></tr><tr><td>Net income</td><td>88</td><td>71</td></tr>"
        "<tr><td>Total assets</td><td>2,310</td><td>2,205</td></tr></table>\n"
    )
    body.append("</body></html>\n")
    return "".join(body), ai


def build_20f(key: str, year: int) -> tuple[str, list[tuple[str, str, str, str]]]:
    name, country, scheme, code, form, city, emp, line = FIRMS[key]
    ai = AI[(key, year)]
    fill = {k: [s.format(name=name, city=city, emp=emp, line=line) for s in v] for k, v in FILLER.items()}
    body = [f"<html><head><title>{html.escape(name)} Form 20-F {year}</title></head><body>\n"]
    body.append(p("SYNTHETIC DEMO DOCUMENT - fictional company, for software testing only."))
    body.append(
        p(f"UNITED STATES SECURITIES AND EXCHANGE COMMISSION. FORM 20-F. Fiscal year ended December 31, {year}.")
    )
    body.append(
        "<table><tr><td>Item 1.</td><td>Identity of Directors, Senior Management and Advisers</td><td>3</td></tr>"
        "<tr><td>Item 3.</td><td>Key Information</td><td>4</td></tr>"
        "<tr><td>Item 4.</td><td>Information on the Company</td><td>12</td></tr>"
        "<tr><td>Item 5.</td><td>Operating and Financial Review and Prospects</td><td>25</td></tr></table>\n"
    )
    body.append("<div><b>Item 1. Identity of Directors, Senior Management and Advisers</b></div>\n")
    body.append(p("Not applicable."))
    body.append("<div><b>Item 2. Offer Statistics and Expected Timetable</b></div>\n")
    body.append(p("Not applicable."))
    for item, title, kind in (
        ("3", "Key Information", "risk"),
        ("4", "Information on the Company", "business"),
        ("5", "Operating and Financial Review and Prospects", "mdna"),
    ):
        body.append(f"<div><b>Item {item}. {html.escape(title)}</b></div>\n")
        if item == "3":
            body.append("<div><b>3.D. Risk Factors</b></div>\n")
        sents = list(fill[kind]) + [a[1] for a in ai if a[0] == kind]
        body.append(p(" ".join(sents)))
    body.append("<div><b>Item 18. Financial Statements</b></div>\n")
    body.append("<table><tr><td>Revenue</td><td>842</td><td>801</td></tr></table>\n")
    body.append("</body></html>\n")
    return "".join(body), ai


def build_annual_report(key: str, year: int) -> tuple[str, list[tuple[str, str, str, str]]]:
    name, country, scheme, code, form, city, emp, line = FIRMS[key]
    ai = AI[(key, year)]
    fill = {k: [s.format(name=name, city=city, emp=emp, line=line) for s in v] for k, v in FILLER.items()}
    body = [
        f'<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><title>{html.escape(name)} Annual Report {year}</title></head><body>\n'
    ]
    body.append(
        '<div style="display:none"><ix:header><ix:hidden>hidden tagged facts that must not be read as text: AI AI AI</ix:hidden></ix:header></div>\n'
    )
    body.append(p("SYNTHETIC DEMO DOCUMENT - fictional company, for software testing only."))
    body.append(f"<h2>{html.escape(name)} Annual Report {year}</h2>\n<h3>Strategic report</h3>\n")
    for kind, heading in (
        ("business", "Our business"),
        ("risk", "Principal risks and uncertainties"),
        ("mdna", "Financial review"),
    ):
        body.append(f"<h3>{heading}</h3>\n")
        sents = list(fill[kind]) + [a[1] for a in ai if a[0] == kind]
        body.append(p(" ".join(sents)))
    body.append("</body></html>\n")
    return "".join(body), ai


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    gold_rows = []
    for key, year in sorted(AI):
        name, country, scheme, code, form, *_ = FIRMS[key]
        if form == "10-K":
            doc, ai = build_10k(key, year)
            fname = f"{key}_{year}_10-K.htm"
        elif form == "20-F":
            doc, ai = build_20f(key, year)
            fname = f"{key}_{year}_20-F.htm"
        else:
            doc, ai = build_annual_report(key, year)
            fname = f"{key}_{year}_annual-report.xhtml"
        (OUT / fname).write_text(doc, encoding="utf-8")
        meta_rows.append(
            {
                "file": fname,
                "firm_id": key,
                "firm_name": name,
                "fiscal_year": year,
                "country": country,
                "industry_code": code,
                "industry_scheme": scheme,
                "form": form,
                "period_end": f"{year}-12-31",
            }
        )
        for _sec, text, state, reason in ai:
            gold_rows.append(
                {
                    "file": fname,
                    "firm_id": key,
                    "fiscal_year": year,
                    "text": text,
                    "gold_state": state,
                    "gold_reason": reason,
                    "annotator": "developers (synthetic)",
                }
            )
    (OUT / "rheinwerk_2023_geschaeftsbericht.html").write_text(
        "<html><body>" + "".join(p(line) for line in GERMAN_REPORT.splitlines()) + "</body></html>", encoding="utf-8"
    )
    meta_rows.append(
        {
            "file": "rheinwerk_2023_geschaeftsbericht.html",
            "firm_id": "rheinwerk",
            "firm_name": "Rheinwerk Maschinenbau AG",
            "fiscal_year": 2023,
            "country": "DE",
            "industry_code": "C",
            "industry_scheme": "NACE",
            "form": "annual-report",
            "period_end": "2023-12-31",
        }
    )
    with open(OUT / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta_rows[0]))
        w.writeheader()
        w.writerows(meta_rows)
    with open(OUT / "demo_gold.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gold_rows[0]))
        w.writeheader()
        w.writerows(gold_rows)
    (OUT / "README.txt").write_text(
        "SYNTHETIC DEMO CORPUS. All companies, people and figures are fictional.\n"
        "Generated by scripts/make_demo_corpus.py for offline testing of AIDisclose.\n"
        "demo_gold.csv lists the label each AI sentence was written to illustrate; it is\n"
        "NOT an independent validation set.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(meta_rows)} documents and {len(gold_rows)} labelled sentences to {OUT}")


if __name__ == "__main__":
    main()
