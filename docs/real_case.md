# Reproducible real-filing demonstration

This is a deliberately selected, single-document functional demonstration,
not a representative sample or an accuracy study. The input is the SEC-hosted
2024 annual report on Form 10-K of SES AI Corporation:

https://www.sec.gov/Archives/edgar/data/1819142/000181914225000017/ses-20241231x10k.htm

On 27 October 2026 the HTML response was 2,204,666 bytes with SHA-256
`0719fda6d04e08b041396f193243f1149881ee20ad2ab779fc37e7c8f9dd1861`.
This selected example is intended to illustrate the local-file workflow.
Classifier predictions are not human validation and cannot establish
accuracy or actual AI adoption.

The original report and text-bearing outputs are not redistributed here.
Download the filing from the link above using the SEC's access rules. Install
this package, then run:

```bash
python examples/05_sec_real_case.py --input /path/to/ses-20241231x10k.htm --out /path/to/new-case
```

The script rejects a different input digest and a nonempty output directory.
It runs the local-file workflow and creates a 60-sentence stratified blind
annotation sample, a sealed key, two independent blank copies, predictions,
and a machine-readable aggregate summary. Versions and local input hashes are
recorded in results/provenance.json. In addition to local installation, use
an appropriately identifying SEC User-Agent when fetching reports through the
SEC adapter, and respect the regulator's current access policy. The case
shows local-file processing of one downloaded report, not a live SEC adapter
test. Third-party filing text should be handled under applicable terms.

## Independent annotation and analysis

1. Give `annotator_A.csv` and `annotator_B.csv` to two people separately with
   `annotation/codebook.md`. Do not share the key or predictions. Each labels
   `gold_state`, `gold_reason`, and `annotator` independently.
2. Run `aidisclose agreement --annotations annotator_A.csv annotator_B.csv
   --out agreement.md` and review disagreements. Adjudicate into a new gold
   CSV; record who adjudicated and the protocol. Never use the classifier's
   predicted labels as the gold standard.
3. Run `aidisclose evaluate --gold adjudicated.csv --results results --key
   sample_key.csv --weighted --out validation.md`. This estimates performance
   only for the sampled candidate population of this selected report. Use
   multiple independently selected firms and filings for broader inference.
4. The recall audit is a separate stage. The one-document pool and very small
   strata are inadequate for a general recall estimate; use a wider corpus,
   sufficient independent annotation and report the sampling design. Do not
   report a classifier metric until actual independent annotation is complete.

