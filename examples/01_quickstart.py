"""Quick start on the bundled synthetic corpus (runs offline in seconds).

    python examples/01_quickstart.py

The corpus contains fictional companies only; it demonstrates the workflow,
not empirical results.
"""

from importlib import resources
from pathlib import Path

import aidisclose as aid

data = Path(str(resources.files("aidisclose.demo").joinpath("data")))

# 1. a source: here a local folder with a metadata file
src = aid.LocalSource(data, metadata=data / "metadata.csv")
filings = src.list_filings()
print(f"{len(filings)} filings")

# 2. the pipeline: lexicon retrieval + three-state classification (rule backend)
results = aid.Pipeline(backend="rules").run(filings, source=src)
print(results.documents[["firm_name", "fiscal_year", "status", "n_baseline_sentences", "n_substantive"]].head())

# 3. indicators
panel = aid.firm_year_panel(results)
yearly = aid.group_panel(panel, by=["fiscal_year"])
print(yearly[["fiscal_year", "n_firms", "share_keyword", "share_claim", "share_substantive"]])

# 4. the evidence behind one indicator value
print(results.evidence(firm_id="kestrel", fiscal_year=2024)[["reason", "text"]])

# 5. what keyword counts are made of
mix = aid.state_mix(results, by=["fiscal_year"])
print(mix.pivot_table(index="fiscal_year", columns="state", values="share", aggfunc="sum").round(2))

results.save("quickstart-results")
