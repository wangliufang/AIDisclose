# AIDisclose annotation codebook (version 1.1)

This codebook defines the labels that AIDisclose assigns to AI-related
sentences in corporate disclosures. The same definitions are used by the rule
backend (rules 1.1.0), by the language-model prompt (`codebook-v1.1`) and by
human annotators, so that machine labels and validation labels measure the
same construct.

## Unit of annotation

One **target sentence** from an annual report that contains at least one term
of the AI lexicon. Annotators see the firm name, the report section and the
previous and next sentences. Use the context only to resolve references
("it", "this system", passive verbs); judge the target sentence.

**Focal firm** = the company that published the report, including its
subsidiaries, products, employees and platforms.

**AI** follows the OECD (2024) definition: a machine-based system that infers
from its inputs how to generate outputs such as predictions, content,
recommendations or decisions. Keywords used in another sense (e.g. "AI" for
artificial insemination or aromatase inhibitor) are false positives.

## States

| State | Definition | Typical cues |
|---|---|---|
| **SUBSTANTIVE** | The focal firm states that it **currently or previously** uses, deploys, sells or has invested in a concrete AI system, product or capability, **and** the sentence contains at least one **specific element**: an application or business process, a named product, a quantity, a date, or an investment (acquisition, patents, dedicated team, compute). | "we use machine learning to detect fraud", "our AI routing engine reduced fuel use by 7%", "we acquired X, an AI start-up" |
| **WEAK** | The focal firm refers to **its own** AI in a forward-looking, exploratory, hedged, risk-boilerplate or unspecific way. | "we plan to", "we are exploring", "pilot", "may", "we are an AI-first company", "our use of AI may result in reputational harm" |
| **NO_CLAIM** | No valid claim about the focal firm's own AI adoption. | competitors, customers, vendors, attackers; industry trends; laws and regulation; explicit non-use; false positives; headings |

If a sentence contains several AI mentions, label the **strongest** state
(SUBSTANTIVE > WEAK > NO_CLAIM).

## Reasons

| State | Reason | Use when |
|---|---|---|
| SUBSTANTIVE | DEPLOYMENT | AI is used in the firm's own operations |
| SUBSTANTIVE | PRODUCT | AI is part of a product or service offered to customers, or AI-related revenue is reported |
| SUBSTANTIVE | INVESTMENT | acquisitions, patents, dedicated AI teams, AI compute or capital expenditure |
| WEAK | FORWARD_LOOKING | plans, intentions, expectations, pilots, trials, evaluations |
| WEAK | HEDGED | modal or conditional claims outside risk language ("we may use AI") |
| WEAK | RISK_BOILERPLATE | generic risk language about the firm's AI ("our use of AI may expose us to ...") |
| WEAK | VAGUE | present-tense claim without any specific element ("we use AI across our business") |
| NO_CLAIM | THIRD_PARTY | the AI belongs to someone else (competitors, customers, vendors, criminals) |
| NO_CLAIM | MARKET_TREND | statements about industries, markets or the economy |
| NO_CLAIM | REGULATORY | laws, regulation, supervisory guidance about AI |
| NO_CLAIM | NEGATED | the firm states that it does not use AI |
| NO_CLAIM | FALSE_POSITIVE | the keyword does not mean artificial intelligence |
| NO_CLAIM | GENERIC | no identifiable agent (headings, definitions) |

## Decision rules for hard cases

1. **Pilots and trials are WEAK.** Operational use requires that the system is
   in production or has produced results ("now prices 60% of policies").
2. **Specificity is required for SUBSTANTIVE.** "We use AI in our products" is
   WEAK/VAGUE; "we use AI to rank search results for 9,000 customers" is
   SUBSTANTIVE.
3. **"Continue to use/invest"** describes current use, not a plan.
4. **Risk factors are not automatically WEAK.** A risk-factor sentence that
   states concrete current use ("we use machine learning in underwriting, and
   errors could harm us") is SUBSTANTIVE.
5. **Services to clients count as PRODUCT** ("we help clients deploy machine
   learning for AML monitoring").
6. **Agentless financial statements** in the report ("AI-related revenue was
   12% of total revenue") refer to the focal firm.
7. **Governance without use** (AI policies, committees, principles) is WEAK/VAGUE.
8. **Predictions about AI's effect** on the firm's markets, demand, industry
   or business are NO_CLAIM/MARKET_TREND unless they describe the firm's own
   AI, also when phrased in the firm's voice ("we cannot predict how AI will
   affect demand for our services").
9. **Definitions and headings** ("Artificial Intelligence (AI) refers to ...")
   are NO_CLAIM/GENERIC.
10. **Agentless risk boilerplate in a risk section** ("Issues in the
    development and use of AI may result in reputational harm") is the firm's
    own risk disclosure: WEAK/RISK_BOILERPLATE. The same sentence outside a
    risk section, or a sentence that only describes regulation ("the use of AI
    is subject to increasing regulatory scrutiny"), is NO_CLAIM.
11. **Licensing in is use.** "We license models from X" or "AI provided by our
    vendor" describes the firm's own use (DEPLOYMENT); "we license our models
    to banks" is a PRODUCT.
12. **A named AI model or platform is a specific element** ("our engineers use
    GitHub Copilot", "we build on Azure OpenAI").

## Annotation procedure (recommended)

1. Draw a stratified sample with `aidisclose sample` (blind file without
   predictions plus a key file with sampling weights).
2. Two annotators label independently (`gold_state`, `gold_reason`); compute
   Cohen's kappa on the overlap; resolve disagreements by discussion and record
   the final label.
3. Run `aidisclose evaluate --key <key file> --weighted` to estimate
   population-level accuracy; the report gives per-state precision, recall
   and F1, accuracy, macro-F1 and Cohen's kappa with stratified bootstrap
   confidence intervals.
4. Optionally train the supervised backend on the adjudicated labels
   (`aidisclose train --cv 5`) and evaluate it on a separate held-out sample.

## Known limitations of the rule backend (rules 1.1.0)

The sentence cases in `tests/fixtures/stress_cases.yaml` illustrate rule
behaviour. Some examples were used during rule design; these cases are not an
independent validation benchmark. Three known limitations are recorded in the
fixture:

* A tool described only by what it does for staff ("advisors use a
  retrieval-augmented generation tool that answers questions from our research
  library") lacks an application cue the rules recognise and is labelled WEAK.
* "Fine-tune" contains the risk word "fine", and "claims data" is not an
  application cue, so "we license LLMs ... and fine-tune them on our claims
  data" is labelled WEAK.
* In "we do not currently use AI ..., but we may do so in the future" the
  negation wins over the forward-looking clause (labelled NO_CLAIM, codebook:
  WEAK).

Further limitations:

* AI supplied by others and used by the firm is recognised in active sentences
  ("we use third-party AI models"), but not in passive ones whose subject is
  the supplier ("our vendors' AI models are used in our call centres":
  labelled THIRD_PARTY).
* Named products without any lexicon term are not retrieved (lexicon recall;
  see the recall audit in `docs/validation.md`).
* Reason codes are less reliable than states; PRODUCT versus DEPLOYMENT in
  particular depends on knowledge of who the users are.
* The developer-written examples (`rule_cases.yaml`) and stress sentences are
  unit tests, not a validation benchmark; accuracy on real reports must be
  measured with the validation kit.
