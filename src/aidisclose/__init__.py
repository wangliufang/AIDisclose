"""AIDisclose: measure substantive AI adoption in corporate disclosures.

Quick start::

    import aidisclose as aid

    src = aid.LocalSource("reports/")               # or EdgarSource / XbrlFilingsSource
    results = aid.Pipeline().run(src.list_filings(), source=src)
    panel = aid.firm_year_panel(results)
    countries = aid.group_panel(panel, by=["country", "fiscal_year"])
"""

from ._version import __version__
from .aggregate import firm_year_panel, group_panel, state_mix
from .classify import RuleClassifier, get_classifier
from .lexicon import Lexicon
from .models import AdoptionState, Candidate, Document, Filing, Prediction, Reason
from .pipeline import Pipeline, Results
from .sources import EdgarSource, LocalSource, XbrlFilingsSource, get_source, register_source

__all__ = [
    "__version__",
    "AdoptionState",
    "Candidate",
    "Document",
    "Filing",
    "Prediction",
    "Reason",
    "Lexicon",
    "Pipeline",
    "Results",
    "RuleClassifier",
    "get_classifier",
    "EdgarSource",
    "LocalSource",
    "XbrlFilingsSource",
    "get_source",
    "register_source",
    "firm_year_panel",
    "group_panel",
    "state_mix",
]
