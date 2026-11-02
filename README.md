# Headache & Migraine Frequency Extraction from Clinical Notes

A hybrid clinical NLP pipeline for extracting **current headache and
migraine frequency** from unstructured neurology notes using a local
large language model (LLM) plus deterministic clinical rules.

The pipeline standardizes heterogeneous documentation into
headache/migraine days per **28-day period**, while distinguishing
current frequency from historical frequency, medication schedules,
symptom duration, and unrelated clinical events.

## Overview

Headache frequency can be documented in many forms:

``` text
Frequency: 10-12/month
Currently has 2-3 per week
# headache days per month: 11/30 days
Chronic daily headache
Headache is almost daily
Migraines are better controlled, ~monthly
```

A pure keyword/regex approach can miss these variations, while
unrestricted LLM extraction can confuse medication schedules, historical
values, or unrelated symptoms with headache frequency.

The pipeline combines:

-   **LLM-based extraction** for flexible interpretation of clinical
    language.
-   **Note-level consolidation** for resolving candidate frequencies
    across chunks.
-   **Deterministic clinical rules** for high-confidence documentation
    patterns.
-   **Auditable evidence fields** for reviewing why a frequency was
    selected.
-   **Reference-based evaluation** when a `CC` annotation is available.

## Extraction Pipeline

``` text
Clinical Note
     |
     v
Text Cleaning
     |
     v
Overlapping Note Chunking
     |
     v
LLM Frequency Extraction
     |
     v
Candidate Frequency Detection
     |
     v
Note-Level Consolidation
     |
     v
Deterministic Clinical Rules
     |
     v
Final Headache / Migraine Frequency
     |
     v
Reference Comparison & Evaluation
```

## Model

The current implementation uses:

``` text
NousResearch/Nous-Hermes-2-Mistral-7B-DPO
```

Inference uses Hugging Face Transformers with 4-bit quantization through
`bitsandbytes`. Generation is deterministic (`do_sample=False`). A
CUDA-capable GPU is recommended.

## Frequency Normalization

The current pipeline uses a **28-day normalization convention**.

  Clinical documentation       Normalized output
  -------------------------- -------------------
  `1/week`                             `4/month`
  `1-2/week`                         `4-8/month`
  `2-3/week`                        `8-12/month`
  `10 headache days/month`            `10/month`
  `daily`                             `28/month`
  `every day`                         `28/month`
  `chronic daily headache`            `28/month`
  `almost daily`                      `28/month`
  `weekly`                             `4/month`
  `monthly`                            `1/month`
  `11/30 headache days`               `11/month`

## Structured Headache Templates

The pipeline recognizes common structured neurology documentation:

``` text
Summary of headache:
Aura: none
Location: right occipital
Duration of headache: hours to a day
Frequency: 10-12/month
Associated symptoms: photophobia, phonophobia, nausea
```

This produces:

``` text
Headache frequency: 10-12/month
```

When the same phenotype is explicitly identified as migraine, the
frequency can populate both headache and migraine outputs. A generic
`Frequency:` field outside a headache/migraine-specific block is not
automatically treated as headache frequency.

## Headache-Day Templates

Dedicated headache-day fields receive high extraction priority.

``` text
# headache days per month: 11/30 days
```

becomes `11/month`, while:

``` text
# headache days per month: daily
```

becomes `28/month`.

The pipeline prevents a field such as `# severe headache days per month`
from overwriting the overall headache-day count.

## Current vs. Historical Frequency

When multiple frequencies are present, the extraction strategy
prioritizes:

``` text
Structured current frequency
        >
Current narrative frequency
        >
Historical frequency
```

Historical expressions such as `used to have`, `previously`,
`at one point`, and `before treatment` should not override clearly
documented current frequency.

## Headache vs. Migraine Frequency

The pipeline maintains separate outputs:

``` text
headache_frequency_IB
migraine_frequency_IB
```

This is important when overall headache frequency differs from the
migraine subset. A diagnosis of **chronic migraine alone is not
automatically converted to 28 migraine days/month**. Migraine frequency
requires quantified migraine evidence or a clear link between a
quantified headache phenotype and the migraine diagnosis.

## Deterministic Recovery Rules

V6 prioritizes high-confidence evidence approximately as follows:

1.  Dedicated `# headache days per month:` fields
2.  Structured `Summary of headache/migraine -> Frequency`
3.  Explicit current narrative numeric frequency
4.  Qualitative current frequency
5.  Daily/continuous headache language
6.  LLM-derived extraction and note-level consolidation

Supported qualitative patterns include `almost daily`,
`most of the time`, `~monthly`, `weekly`, `every day`,
`continuous headache`, and `constant headache`.

## Excluded Frequencies

The pipeline attempts not to interpret the following as headache
frequency:

-   Medication dosing schedules
-   Preventive treatment schedules
-   Acute-treatment limits
-   Symptom duration
-   Follow-up intervals
-   Unrelated symptom frequencies
-   Family-history or social-history frequencies
-   Historical headache frequency when a current frequency is available

## Output Columns

The output CSV retains the original data and adds extraction/auditing
fields including:

``` text
headache_frequency_IB
migraine_frequency_IB
num_chunks_IB
num_candidate_chunks_IB
deterministic_evidence_IB
chunk_candidates_IB
consolidation_output_IB
CC_IB_status
CC_IB_match
CC_IB_strict_match
```

`deterministic_evidence_IB` records evidence responsible for
deterministic overrides and supports error analysis.

## Evaluation

The current development evaluation used a manually annotated **30-note
test set**, with `CC` as the reference headache-frequency annotation.

### V6 Performance

  Metric                                       V6
  ----------------------------------- -----------
  Extraction sensitivity / coverage     **96.7%**
  Range-aware accuracy                  **76.7%**
  Strict exact accuracy                 **60.0%**
  Agreement among extracted cases       **79.3%**
  Missed extractions                     **1/30**

The primary development objective was improving extraction sensitivity:

``` text
Old pipeline: 66.7%
V6 pipeline:  96.7%
```

This represents a **30 percentage-point improvement** in extraction
sensitivity/coverage on the development evaluation set.

### Metric Definitions

**Extraction sensitivity / coverage:** percentage of evaluable reference
cases for which the pipeline returns a usable headache-frequency
prediction.

**Range-aware accuracy:** a prediction is correct when the reference
value falls within the predicted interval. For example, reference `12`
and prediction `8-12/month` is a match.

**Strict exact accuracy:** the prediction must be a single value exactly
equal to the reference.

## Installation

``` bash
pip install pandas torch transformers accelerate bitsandbytes tqdm
```

Exact package requirements may vary by the local CUDA/PyTorch
environment.

## Configuration

Update the paths near the beginning of the script:

``` python
INPUT_FILE = "/path/to/input.csv"
OUTPUT_FILE = "/path/to/output.csv"
NOTE_COLUMN = "Full Note"
```

Generation settings can also be configured:

``` python
BATCH_SIZE = 8
MAX_INPUT_TOKENS = 3000
MAX_NEW_TOKENS = 100
```

## Running V6

``` bash
python headache_migraine_frequency_extraction_full_v6_recall.py
```

The pipeline loads the clinical-note CSV, cleans and chunks notes,
performs batched LLM extraction, consolidates candidates, applies
deterministic recovery rules, evaluates against `CC` when available, and
saves the output CSV.

## Auditing Predictions

Useful fields for error analysis include:

``` text
Full Note
CC
headache_frequency_IB
migraine_frequency_IB
deterministic_evidence_IB
chunk_candidates_IB
consolidation_output_IB
CC_IB_status
```

These fields help identify errors arising from missing extraction,
temporal interpretation, phenotype assignment, deterministic overrides,
normalization, or candidate consolidation.

## Current Limitations

The current evaluation set is small (`n=30`), so reported performance
should be considered preliminary. Residual challenges include:

-   Historical and current frequencies occurring close together
-   Multiple headache phenotypes in the same note
-   Daily headache with a less frequent severe/migraine subset
-   Incomplete distinction between headache and migraine frequency
-   Contradictory frequencies across note sections
-   Older daily/chronic descriptions competing with newer numeric
    frequencies

Validation on a larger, independent clinical-note cohort is required to
assess generalizability.

## Development Approach

The pipeline combines **LLM extraction** for heterogeneous clinical
language with **deterministic clinical rules** for reproducible handling
of recurring documentation patterns and improved auditability.

The goal is to transform documented headache-frequency information into
a standardized representation suitable for downstream clinical NLP and
research workflows.

## Disclaimer

This software is intended for **research use only**.

It is not a medical device and should not be used independently for
diagnosis, treatment decisions, or other clinical decision-making.
Extracted values should be validated against source documentation when
used for research or clinical analysis.
