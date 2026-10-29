# AIDisclose

**Measure substantive AI adoption in corporate disclosures, beyond keyword counts.**

AIDisclose retrieves annual reports from regulatory sources, finds sentences
that mention artificial intelligence and classifies each of them into one of
three states:

| State | Meaning | Example |
|---|---|---|
| `SUBSTANTIVE` | the reporting firm uses, sells or has invested in a concrete AI system, with a specific element | "We use machine learning models to forecast freight demand at 85 depots." |
| `WEAK` | the firm's own AI is mentioned in a forward-looking, hedged, boilerplate or vague way | "We plan to use artificial intelligence to enhance our customer experience." |
| `NO_CLAIM` | no valid claim about the firm's own AI (third parties, markets, regulation, negation, false positives) | "Competitors may use AI to offer lower prices." |

The labels are aggregated into firm-year and country/industry-year indicators
next to a plain keyword baseline, so researchers can see how much a keyword
measure overstates substantiated adoption. Every indicator can be traced back
to the sentences behind it, and the validation kit measures precision,
agreement, recall and convergent validity on the user's own corpus.

* [Getting started](getting-started.md): install, demo, a first study
* [Sources](sources.md): SEC EDGAR (10-K, 20-F, 40-F and exhibits), ESEF/UKSEF via filings.xbrl.org, local folders, new adapters
* [Sections](sections.md): Item and annual-report sections, filters and presets
* [Backends](backends.md): rules, supervised, LLM, experimental transformer; relabelling
* [Indicators](indicators.md): firm-year and group-year measures, novelty and onset
* [Validation](validation.md): annotation, agreement, recall audit, external benchmarks, quality checks
* [Reproducibility and large runs](reproducibility.md): provenance, versions, checkpoints, parallel runs, languages
* [Codebook](codebook.md): label definitions and decision rules

AIDisclose ships English lexicons and English rule cues. Other languages require
user-supplied and independently validated lexicons and compatible backends;
documents without a matching lexicon are skipped and reported.
