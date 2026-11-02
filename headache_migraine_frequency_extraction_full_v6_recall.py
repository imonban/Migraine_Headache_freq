import pandas as pd
import torch
import re
from tqdm.auto import tqdm

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ======================================================
# SETTINGS
# ======================================================

# Project convention: frequencies are normalized to a 28-day window.
# Therefore daily/every-day headache = 28 headache days per 28 days.
DAYS_PER_MONTH = 28


MODEL_NAME = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

INPUT_FILE = "/media/Datacenter_storage/LLM_apigee/SampleNotes_HdFreq_MigraineFreq_Annotated_P3_LLM_output.csv"
OUTPUT_FILE = "/media/Datacenter_storage/LLM_apigee/SampleNotes_HdFreq_MigraineFreq_Annotated_P3_LLM_outputIB.csv"

NOTE_COLUMN = "Full Note"

# Batch size for generation.
BATCH_SIZE = 8

# Each note is split into overlapping chunks (see CHUNKING below)
# before extraction, so individual prompts are much shorter than a
# full note -- this token budget is mostly a safety net.
MAX_INPUT_TOKENS = 3000
MAX_NEW_TOKENS = 80

# ------------------------------------------------------
# CHUNKING
# ------------------------------------------------------
# Long notes are split into overlapping character-based chunks. Each
# chunk is sent to the LLM independently ("chunk extraction"). Chunks
# that come back with an actual frequency (rather than
# "Not mentioned") are the candidates that get reconciled into one
# FINAL current frequency per note ("consolidation"). This avoids:
#   - sending the whole note at once, where irrelevant sections
#     (med sigs, social history, etc.) can drown out the real signal
#   - a single small keyword-window snippet that can cut a relevant
#     phrase off mid-sentence
# The overlap between consecutive chunks reduces the chance that a
# frequency phrase gets split across a chunk boundary.
CHUNK_CHARS = 3000
CHUNK_OVERLAP_CHARS = 300


# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv(INPUT_FILE)
print(f"Number of rows: {len(df):,}")

if NOTE_COLUMN not in df.columns:
    raise ValueError(
        f"Column '{NOTE_COLUMN}' not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ======================================================
# CLEAN TEXT (remove non-ASCII characters)
# ======================================================

def clean_text(text):
    """
    Remove non-ASCII characters from a string.

    Non-string / NaN values are returned unchanged so downstream
    `isinstance(text, str)` checks still behave correctly.
    """

    if not isinstance(text, str):
        return text

    # Strip anything outside the standard ASCII range (0-127).
    # This removes smart quotes, accented letters, emoji, special
    # dashes/bullets, etc. that sometimes appear in clinical notes
    # and can break tokenization or downstream regex matching.
    return text.encode("ascii", "ignore").decode("ascii")


print("Cleaning non-ASCII characters from notes...")
df[NOTE_COLUMN] = df[NOTE_COLUMN].apply(clean_text)


# ======================================================
# CHUNK EACH NOTE INTO OVERLAPPING PIECES
# ======================================================

def chunk_text(text, chunk_size=CHUNK_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    """
    Split `text` into a list of overlapping chunks of roughly
    `chunk_size` characters each. The `overlap` characters shared
    between consecutive chunks reduce the chance that a relevant
    phrase (e.g. a frequency mention) gets split across a boundary.

    Short notes (<= chunk_size) are returned as a single "chunk" so
    the rest of the pipeline doesn't need a special case for them.
    """

    if not isinstance(text, str) or not text.strip():
        return [text if isinstance(text, str) else ""]

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = end - overlap

    return chunks


# ======================================================
# PROMPTS
# ======================================================

DEFAULT_PROMPT = """
You are a clinical NLP model extracting CURRENT headache and migraine frequency.

Your task is to identify the CURRENT frequency from the clinical text.

IMPORTANT:
- Extract explicit frequency expressions even when the surrounding text is incomplete.
- A frequency does not need to be immediately next to "headache" or "migraine".
- Recognize:
  - 1x/week
  - 1-2x/week
  - 1x a week
  - 1-2x a week
  - once a week
  - twice a week
  - 3 times per month
  - 10 headache days/month
  - 4 migraine days in the past month
  - daily
  - every 2 days

STRUCTURED HEADACHE TEMPLATES:
- Clinical notes may contain a block such as "Summary of headache:" followed
  several lines later by "Frequency: 10-12/month".
- Within a headache/migraine summary block, the "Frequency:" field describes
  the headache phenotype even though the word headache is not repeated on
  that line.
- Prefer this dedicated structured Frequency field over a less precise nearby
  narrative approximation.
- Example: "currently 2-3/week" plus later "Summary of headache ... Frequency:
  10-12/month" -> use 10-12/month.
- Never treat a generic "Frequency:" field outside a headache/migraine-specific
  block as headache frequency.

CURRENT FREQUENCY:
- Use the most recent/current frequency.
- Ignore historical frequencies if a current frequency is available.
- Do not add frequencies together.
- TRANSITION PHRASES: When the note describes a change over time using
  phrasing like "was having X but now Y", "previously X, now down to Y",
  "used to be X, currently Y", or similar before/after language, ALWAYS
  use the LATER value (Y) as the current frequency. Do not use the
  earlier value (X) and do not average X and Y.
  Example: "was having 4+/week but now down to 1-2/week in the past
  month" -> current frequency is 1-2/week (i.e. 4-8/month), NOT 4+/week.

NON-HEADACHE FREQUENCIES:
- Only extract frequencies that describe headache or migraine episodes
  themselves.
- Do NOT use frequencies that describe a different, separate symptom
  or event, even if it appears near headache/migraine text or its
  relationship to headache/migraine is uncertain (e.g. "~4/month of
  unusual body posturing", "dizziness 2x/week", "nausea daily" when
  these are described as their own distinct symptom rather than a
  headache/migraine episode).
- If the note itself says the relationship between another symptom
  and headache/migraine is "unclear" or "unrelated", do not use that
  symptom's frequency for headache or migraine frequency.

HEADACHE VS MIGRAINE:
- If headache and migraine refer to the same underlying clinical event,
  use the same current frequency for both.
- If the note clearly distinguishes headache frequency from migraine
  frequency, report them separately.
- Do not create separate frequencies merely because both words appear.

TEMPORAL / EVIDENCE PRIORITY:
- Extract the patient's CURRENT or most recently documented actual frequency.
- Prefer assessment/HPI statements about current burden over copied problem-list,
  diagnosis, medication instructions, templated education, or historical text.
- "Before treatment", "previously", "used to", "at baseline", "prior to Botox",
  and similar phrases are historical unless explicitly stated to still be current.
- "Now", "currently", "since starting", "over the last/past 4 weeks/month",
  "at present", and the latest follow-up statement indicate current status.
- Do NOT use medication-use limits or instructions as disease frequency:
  examples: "Nurtec every other day", "sumatriptan max 2 days/week",
  "Botox every 12 weeks", "take BID", "may repeat in 2 hours".
- Do NOT use planned/target frequencies ("goal <4/month", "should limit to
  2 days/week") as observed frequency.
- Do NOT use family history or another person's frequency.
- If multiple current values conflict and the text does not establish which is
  newer or which refers to headache vs migraine, return Not mentioned rather
  than guessing.

HEADACHE-DAY VS MIGRAINE-DAY SEMANTICS:
- "headache days" / overall headache burden -> Headache frequency.
- "migraine days", "migraines", or a clearly migraine/severe subset -> Migraine frequency.
- Do NOT automatically copy a headache-day value into migraine frequency.
- Do NOT automatically copy a migraine-day value into headache frequency unless
  the note clearly uses headache and migraine interchangeably for the same events.
- If the note says "20 headache days, 8 migraine days per month", output 20 and 8.
- If it says "daily headache, with 8 migraines/month", output 30 and 8.
- If it says only "8 migraines/month", migraine=8/month; headache is Not mentioned
  unless the note establishes that every headache is a migraine.

NORMALIZATION PRIORITY:
- Prefer an explicit numeric frequency when it refers to the SAME event class
  and SAME current time frame. Do not let a numeric severe/migraine subset
  override an overall daily headache pattern. Example: "daily headache; severe
  migraines 8/month" means headache=30/month and migraine=8/month.
- Only apply the generic normalizations below when NO explicit
  numeric frequency is stated for that headache type.

NORMALIZATION (use only when no explicit number is given):
1/week = 4/month
1-2/week = 4-8/month
2-3/week = 8-12/month
1/day = 28/month
2/day = 56/month
X/month = X/month
X over 4 weeks = X/month
X over 28 days = X/month
chronic daily headache (no other number given) = 28/month
daily headache (no other number given) = 28/month
daily (no other number given) = 28/month
weekly (no number given) = 4/month
monthly (no number given) = 1/month

VAGUE OR NON-NUMERIC FREQUENCY DESCRIPTIONS:
Clinical notes often describe frequency with qualitative phrases
instead of a number. Some of these imply a specific frequency and
should be converted; others do not quantify frequency at all and must
be left as "Not mentioned" rather than guessed.

- Phrases that describe the headache/migraine itself as happening
  continuously or nearly every day are equivalent to daily and should
  be normalized to 30/month. This includes:
  - "continuous" / "constant" / "all the time"
  - "most of the time" / "most of the day"
  - "hemicrania continua" (continuous by definition)
- IMPORTANT: "chronic migraine" alone is NOT a numeric frequency. Do NOT
  convert the diagnosis "chronic migraine" to 30/month. Only quantify it
  when the note separately states a frequency (e.g. 20 headache days/month,
  8 migraine days/month, daily headache).
- Phrases that describe treatment response, severity, or clinical
  course -- NOT how often the headache/migraine occurs -- do NOT imply
  a frequency and must be left as "Not mentioned" unless an actual
  frequency is stated elsewhere in the text. Do not infer a frequency
  from phrases like:
  - "well controlled" / "well managed" / "stable on current regimen"
  - "occasional breakthrough" / "rare flare" with no number given
  - "improved" / "worsened" / "recently recurred" with no number given
- If a single passage combines both (e.g. "well controlled, but she
  still has a mild headache most of the time"), use only the part that
  actually describes occurrence ("most of the time" -> 28/month); do
  not let the control/severity language change that number or cause
  you to output "Not mentioned" instead.

MIXED CASES:
- A note may describe an underlying near-daily/chronic headache
  PLUS a distinct, more severe subset with its own explicit frequency
  (e.g. "headache almost daily... about 3-4 days a week he gets a
  severe headache he calls migraine" or "daily, severe ones 12-16/month").
  In this case:
  - Headache frequency: use daily/chronic normalization (28/month)
    for the overall headache pattern, since it is described as
    daily/near-daily.
  - Migraine frequency: use the explicit number given for the
    severe/migraine subset (e.g. 12-16/month, or 3-4/week -> 12-16/month).
- Do not average or combine the daily pattern and the explicit
  severe-subset number into a single figure.
chronic daily headache = 28/month
daily headache = 28/month

Do not confuse duration with frequency.

"headaches last 4 hours" = NOT frequency
"headaches occur 2x/week" = frequency
"She has had chronic daily headache for years." = 28/month (daily is normalized to a 28-day observation window; the duration "for years" is history, not frequency, and should be ignored)

OUTPUT EXACTLY THREE LINES:

Headache frequency: <value>/month OR Not mentioned
Migraine frequency: <value>/month OR Not mentioned
Evidence: <short exact phrase from the clinical text supporting the CURRENT values>

If truly not documented:

Headache frequency: Not mentioned
Migraine frequency: Not mentioned

Clinical text:
{context}
"""


CONSOLIDATION_PROMPT = """
You are consolidating several candidate extractions of headache and
migraine frequency. Each candidate below was extracted independently
from a different, possibly overlapping, chunk of the SAME clinical
note, so some candidates may be duplicates, historical-only,
incomplete, or in conflict with each other.

Using these rules:
- Source text is authoritative; candidate values may be wrong. Re-check each value
  against its source before selecting it.
- Reject frequencies belonging to medication schedules/limits, duration, family
  history, goals, historical baseline, or unrelated symptoms.
- A diagnosis of chronic migraine alone is not a numeric frequency.
- Prefer the CURRENT frequency over any historical-only frequency.
- If a candidate's source text describes a transition over time
  ("was having X but now Y", "previously X, now down to Y"), that
  candidate's Y value is the current one -- prefer it over an older
  X value from another candidate.
- Prefer explicit numeric frequency only within the same event class and current
  time frame. A migraine/severe-subset number must not replace an overall daily
  headache frequency.
- Do not add frequencies from different candidates together, and do
  not average them.
- Keep headache days and migraine days semantically separate. Copy a value
  between them only when the source explicitly establishes that they are the same
  events or uses the terms interchangeably.
- A candidate normalized from continuous/near-daily language
  ("continuous", "most of the time", "hemicrania continua") to
  28/month is just as valid as a candidate with an explicit number --
  do not discard it in favor of a candidate that is actually
  "Not mentioned" wearing different words (e.g. a candidate based only
  on "well controlled" or "stable", which does not describe frequency
  and should be ignored when reconciling).

CANDIDATES:
{candidates}

Determine the single best CURRENT headache frequency and migraine
frequency for this patient.

OUTPUT EXACTLY:

Headache frequency: <value>/month
Migraine frequency: <value>/month
"""




# ======================================================
# OPTIONAL DETERMINISTIC SELF-TESTS
# ======================================================

def run_deterministic_self_tests():
    tests = [
        {
            "name": "chronic daily headache, migraine subset not quantified",
            "note": (
                "She has had chronic daily headache for years. "
                "# Chronic daily headache (every day), likely has some chronic "
                "migraine and chronic tension."
            ),
            "expected_h": "28/month",
            "expected_m": None,
        },
        {
            "name": "headache days numerator denominator template",
            "note": (
                "Current headache characteristics\n"
                "# headache days per month: 11/30 days\n"
                "# severe headache days per month: 8/11 days\n"
            ),
            "expected_h": "11/month",
            "expected_m": None,
        },
        {
            "name": "daily headache days template",
            "note": (
                "Current headache characteristics\n"
                "# headache days per month: daily with associated neuralgia pain\n"
                "# severe headache days per month: 16-20/30 days\n"
            ),
            "expected_h": "28/month",
            "expected_m": None,
        },
        {
            "name": "weekly structured headache frequency",
            "note": (
                "Summary of headache:\n"
                "Duration of headache: hours to days\n"
                "Frequency: weekly\n"
            ),
            "expected_h": "4/month",
            "expected_m": None,
        },
        {
            "name": "migraine headaches structured header",
            "note": (
                "Summary of migraine headaches:\n"
                "Duration of headache: hours\n"
                "Frequency: 3-8/month\n"
            ),
            "expected_h": "3-8/month",
            "expected_m": "3-8/month",
        },
        {
            "name": "structured episodic migraine frequency",
            "note": (
                "Summary of headache:\n"
                "Aura: no aura\n"
                "Duration of headache: up to 4 hours\n"
                "Frequency: 8-12/month\n"
                "Associated symptoms: photophobia, nausea\n\n"
                "ASSESSMENT / PLAN\n"
                "# Episodic migraine without aura\n"
            ),
            "expected_h": "8-12/month",
            "expected_m": "8-12/month",
        },
        {
            "name": "structured frequency beats weekly approximation",
            "note": (
                "Her migraines started at age 13. She used to have about weekly "
                "headaches. She currently has 2-3 per week, about one severe.\n"
                "Summary of headache:\n"
                "Aura: none\n"
                "Frequency: 10-12/month\n"
                "Associated symptoms: photophobia, phonophobia, nausea\n\n"
                "ASSESSMENT / PLAN\n"
                "# Episodic migraine without aura.\n"
            ),
            "expected_h": "10-12/month",
            "expected_m": "10-12/month",
        },
    ]

    print("\nDeterministic extraction self-tests")
    print("=" * 60)

    all_ok = True
    for t in tests:
        r = deterministic_frequency_evidence_v3(t["note"])
        ok = (
            r["headache"] == t["expected_h"]
            and r["migraine"] == t["expected_m"]
        )
        all_ok = all_ok and ok
        print(
            f"{'PASS' if ok else 'FAIL'} | {t['name']}\n"
            f"  headache: {r['headache']} (expected {t['expected_h']})\n"
            f"  migraine: {r['migraine']} (expected {t['expected_m']})"
        )

    print("=" * 60)
    print("ALL TESTS PASSED" if all_ok else "ONE OR MORE TESTS FAILED")
    return all_ok


# ======================================================
# LOAD MODEL
# ======================================================

def load_model():

    print("Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Decoder-only models need a pad token and LEFT padding for
    # correct batched generation (so all sequences finish generating
    # aligned at the right edge).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        return_full_text=False,
    )

    print("Model loaded.")

    return llm


# ======================================================
# LOW-LEVEL BATCHED GENERATION
# ======================================================

def run_prompts_batch(prompts, llm):
    """
    Run the LLM on a list of ALREADY-BUILT prompts and return the
    cleaned generated text for each, in the same order. Used both for
    chunk-level extraction prompts and for consolidation prompts.
    """

    if not prompts:
        return []

    try:

        outputs = llm(
            prompts,
            batch_size=len(prompts),
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        results = []

        for output in outputs:

            if isinstance(output, list):
                generated = output[0]["generated_text"]
            else:
                generated = output["generated_text"]

            # The model can occasionally emit non-ASCII characters
            # (smart quotes, special dashes, etc.) even though the
            # input was cleaned. Strip those here too so downstream
            # parsing/regex and the saved CSV stay ASCII-only.
            results.append(clean_text(generated.strip()))

        return results

    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()

        return [
            "Headache frequency: CUDA Out of Memory\n"
            "Migraine frequency: CUDA Out of Memory"
        ] * len(prompts)

    except Exception as e:

        return [
            "Headache frequency: ERROR\n"
            "Migraine frequency: ERROR"
        ] * len(prompts)


def generate_batch(contexts, llm):
    """
    Build extraction prompts from a batch of chunk texts (using
    DEFAULT_PROMPT) and run them through the LLM. Empty/missing
    contexts are short-circuited to "Not mentioned" without being
    sent to the model.
    """

    results = [None] * len(contexts)

    to_run_idx = []
    to_run_prompts = []

    for i, context in enumerate(contexts):

        if not isinstance(context, str) or not context.strip():
            results[i] = (
                "Headache frequency: Not mentioned\n"
                "Migraine frequency: Not mentioned"
            )
            continue

        to_run_idx.append(i)
        to_run_prompts.append(DEFAULT_PROMPT.format(context=context))

    batch_results = run_prompts_batch(to_run_prompts, llm)

    for idx, result in zip(to_run_idx, batch_results):
        results[idx] = result

    return results


def run_in_batches(items, fn, batch_size, llm, desc):
    """
    Apply `fn(batch, llm) -> list[result]` to `items` in chunks of
    `batch_size`, with a progress bar, and return the flattened
    results in original order.
    """

    results = [None] * len(items)

    n_batches = (len(items) + batch_size - 1) // batch_size

    for batch_start in tqdm(
        range(0, len(items), batch_size),
        total=n_batches,
        desc=desc,
    ):
        batch_end = min(batch_start + batch_size, len(items))
        batch = items[batch_start:batch_end]
        results[batch_start:batch_end] = fn(batch, llm)

    return results



# ======================================================
# HIGH-PRECISION DETERMINISTIC FREQUENCY EVIDENCE
# ======================================================

def deterministic_frequency_evidence(note):
    """
    Capture very high-confidence headache/migraine frequency statements that
    should not depend on the LLM noticing them.

    IMPORTANT:
    - "chronic daily headache" / "daily headache" / "headache every day"
      => overall HEADACHE frequency = 28/month.
    - Do NOT infer migraine = 30/month merely because the note also contains
      "migraine", "chronic migraine", migrainous features, or a migraine
      diagnosis. Migraine needs its own quantified occurrence evidence.
    """
    if not isinstance(note, str) or not note.strip():
        return {"headache": None, "migraine": None, "evidence": []}

    t = re.sub(r"\s+", " ", note).strip()
    evidence = []
    headache = None
    migraine = None

    # Strong lexical evidence for an everyday OVERALL headache pattern.
    daily_headache_patterns = [
        r"\bchronic\s+daily\s+headaches?\b",
        r"\bdaily\s+headaches?\b",
        r"\bheadaches?\s+(?:on\s+)?(?:every|each)\s+day\b",
        r"\bheadaches?\s+everyday\b",
        r"\bheadaches?\s+daily\b",
        r"\bheadaches?\s+(?:are|is)\s+(?:constant|continuous)\b",
        r"\bconstant\s+headaches?\b",
        r"\bcontinuous\s+headaches?\b",
    ]

    for pat in daily_headache_patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            headache = f"{DAYS_PER_MONTH}/month"
            left = max(0, m.start() - 120)
            right = min(len(t), m.end() + 180)
            evidence.append("HEADACHE DAILY: " + t[left:right])
            break

    # Parenthetical/explicit clarification, e.g.
    # "Chronic daily headache (every day)".
    m = re.search(
        r"\bheadaches?\b.{0,50}\(\s*every\s+day\s*\)",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        headache = f"{DAYS_PER_MONTH}/month"
        left = max(0, m.start() - 120)
        right = min(len(t), m.end() + 180)
        evidence.append("HEADACHE EVERY DAY: " + t[left:right])

    # Intentionally no rule that maps "chronic migraine" -> 28/month.
    # A migraine value should come from explicit migraine-day/attack evidence
    # or from the LLM after reviewing such evidence.

    return {
        "headache": headache,
        "migraine": migraine,
        "evidence": evidence,
    }



def _normalize_frequency_expression(raw):
    """
    Normalize a high-confidence frequency expression to '<value>/month'.

    Returns None when the expression cannot be normalized safely.
    """
    if not isinstance(raw, str):
        return None

    s = raw.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")

    # Already monthly: 10/month, 10-12/month, 10 per month, 10-12 per month.
    m = re.fullmatch(
        r"(?:about\s+|approximately\s+|approx\.?\s+|~\s*)?"
        r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*"
        r"(?:/|per\s+)(?:month|mo)",
        s,
    )
    if m:
        value = re.sub(r"\s+", "", m.group(1))
        return f"{value}/month"

    # X headache/migraine days per month.
    m = re.fullmatch(
        r"(?:about\s+|approximately\s+|approx\.?\s+|~\s*)?"
        r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*"
        r"(?:headache|migraine)?\s*days?\s*(?:/|per\s+)(?:month|mo)",
        s,
    )
    if m:
        value = re.sub(r"\s+", "", m.group(1))
        return f"{value}/month"

    # Weekly expressions. Use 4 weeks/month to remain consistent with the
    # project's existing normalization convention.
    m = re.fullmatch(
        r"(?:about\s+|approximately\s+|approx\.?\s+|~\s*)?"
        r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*"
        r"(?:x\s*)?(?:/|per\s+)?(?:week|wk)",
        s,
    )
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else None

        def fmt(x):
            return str(int(x)) if float(x).is_integer() else f"{x:g}"

        if hi is None:
            return f"{fmt(lo * 4)}/month"
        return f"{fmt(lo * 4)}-{fmt(hi * 4)}/month"

    # Common word forms.
    if re.fullmatch(r"(?:once|one time)\s+(?:a|per)\s+week", s):
        return "4/month"
    if re.fullmatch(r"(?:twice|two times)\s+(?:a|per)\s+week", s):
        return "8/month"
    if re.fullmatch(r"(?:daily|every day|everyday)", s):
        return f"{DAYS_PER_MONTH}/month"

    # Qualitative frequencies used in structured headache templates.
    if re.fullmatch(r"(?:weekly|once weekly)", s):
        return "4/month"
    if re.fullmatch(r"(?:monthly|once monthly)", s):
        return "1/month"

    # Headache-day template notation such as 11/30 days, 25/30, 15/28 days.
    # The numerator is the number of headache days in the observation month.
    m = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*/\s*(?:28|30|31)\s*(?:days?)?",
        s,
    )
    if m:
        value = float(m.group(1))
        value = int(value) if value.is_integer() else value
        return f"{value}/month"

    return None


def _extract_structured_headache_frequency(note):
    """
    Extract frequency from dedicated headache/migraine summary templates.

    Supported examples:
      Summary of headache: ... Frequency: 10-12/month
      Summary of migraine headaches: ... Frequency: 3-8/month
      Frequency: weekly

    A generic Frequency field elsewhere in the chart is intentionally ignored.
    """
    if not isinstance(note, str):
        return None, None, None

    header = re.search(
        r"(?im)^\s*(?:"
        r"summary\s+of\s+(?:headache|headaches|migraine|migraines|"
        r"migraine\s+headaches?|headache\s+disorder)s?|"
        r"headache\s+summary|migraine\s+summary"
        r")\s*:\s*$",
        note,
    )
    if not header:
        return None, None, None

    header_text = header.group(0).lower()
    phenotype = "migraine" if "migraine" in header_text else "headache"

    start_pos = header.end()
    tail = note[start_pos:start_pos + 4000]

    boundary = re.search(
        r"(?im)^\s*(?:prior\s+medication|current\s+treatment|"
        r"medical\s+history|surgical\s+history|social\s+history|"
        r"family\s+history|current\s+medications|review\s+of\s+systems|"
        r"objective|assessment\s*/?\s*plan)\b",
        tail,
    )
    block = tail[:boundary.start()] if boundary else tail

    fm = re.search(r"(?im)^\s*frequency\s*:\s*([^\r\n]+?)\s*$", block)
    if not fm:
        return None, None, None

    raw = fm.group(1).strip()

    # Try the whole field first (handles weekly/monthly/11/30 days).
    normalized = _normalize_frequency_expression(raw)

    # Then isolate a numeric frequency if comments trail the value.
    if not normalized:
        candidates = [
            r"(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*(?:/|per\s+)(?:month|mo))",
            r"(\d+(?:\.\d+)?\s*(?:/|per\s+)(?:month|mo))",
            r"(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*(?:x\s*)?(?:/|per\s+)?(?:week|wk))",
            r"(\d+(?:\.\d+)?\s*(?:x\s*)?(?:/|per\s+)?(?:week|wk))",
            r"(\d+(?:\.\d+)?\s*/\s*(?:28|30|31)\s*(?:days?)?)",
            r"\b(weekly|monthly|daily|every\s+day|everyday)\b",
        ]
        for pat in candidates:
            cm = re.search(pat, raw, flags=re.IGNORECASE)
            if cm:
                normalized = _normalize_frequency_expression(cm.group(1))
                if normalized:
                    break

    if not normalized:
        return None, None, None

    evidence = (
        note[header.start():start_pos]
        + block[max(0, fm.start() - 250):min(len(block), fm.end() + 250)]
    )
    evidence = re.sub(r"\s+", " ", evidence).strip()
    return normalized, phenotype, evidence


def _note_identifies_current_headache_phenotype_as_migraine(note):
    """
    Decide conservatively whether the headache phenotype whose frequency is
    being summarized is explicitly identified as migraine.

    This intentionally does NOT return True merely because 'migraine' occurs
    somewhere in PMH/problem list/family history/medication names.
    """
    if not isinstance(note, str):
        return False, []

    evidence = []

    # Assessment headings are particularly strong evidence.
    assessment_patterns = [
        r"(?im)^\s*#\s*(?:episodic|chronic)?\s*migraine\b[^\r\n]*",
        r"(?im)^\s*(?:assessment|impression)\s*[:\-]?\s*"
        r"(?:episodic|chronic)?\s*migraine\b[^\r\n]*",
    ]
    for pat in assessment_patterns:
        for m in re.finditer(pat, note):
            evidence.append(re.sub(r"\s+", " ", m.group(0)).strip())

    # Narrative linkage: "her migraines..." / "current migraines..."
    narrative_patterns = [
        r"(?i)\b(?:her|his|their)\s+migraines?\b",
        r"(?i)\bcurrent(?:ly)?\b.{0,80}\bmigraines?\b",
        r"(?i)\bmigraines?\b.{0,80}\bcurrent(?:ly)?\b",
    ]
    for pat in narrative_patterns:
        m = re.search(pat, note)
        if m:
            left = max(0, m.start() - 100)
            right = min(len(note), m.end() + 180)
            evidence.append(re.sub(r"\s+", " ", note[left:right]).strip())

    return bool(evidence), evidence


def _extract_current_narrative_frequency(note):
    """
    Recover explicit CURRENT headache/migraine frequencies from narrative text.

    Examples:
      'She currently has 2-3 per week'
      'currently having 3 headaches per week'
      'now down to 1-2 migraines/week'

    This is deliberately narrower than the LLM and serves as a rescue layer.
    """
    if not isinstance(note, str):
        return None, None, None

    patterns = [
        # currently has 2-3 per week
        r"(?i)\b(?:currently|now)\s+(?:she|he|they|patient\s+)?"
        r"(?:has|have|having|gets?|experiences?|reports?)\s+"
        r"(?:about\s+|approximately\s+|~\s*)?"
        r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*"
        r"(?:x\s*)?(?:/|per\s+)?(?:week|wk|month|mo))",

        # currently 2-3/week headaches/migraines
        r"(?i)\b(?:currently|now)\b.{0,60}?"
        r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*"
        r"(?:x\s*)?(?:/|per\s+)?(?:week|wk|month|mo))"
        r".{0,50}?\b(headaches?|migraines?)\b",

        # now down to 1-2/week
        r"(?i)\b(?:now|currently)\s+(?:down\s+to\s+)?"
        r"(?:about\s+|approximately\s+|~\s*)?"
        r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*"
        r"(?:x\s*)?(?:/|per\s+)?(?:week|wk|month|mo))",
    ]

    for pat in patterns:
        m = re.search(pat, note)
        if not m:
            continue

        normalized = _normalize_frequency_expression(m.group(1))
        if not normalized:
            continue

        left = max(0, m.start() - 180)
        right = min(len(note), m.end() + 220)
        context = re.sub(r"\s+", " ", note[left:right]).strip()

        # Determine event type from nearby wording. If unclear, call it headache;
        # later migraine phenotype evidence may propagate it.
        local = note[left:right].lower()
        event_type = "migraine" if "migraine" in local else "headache"

        return normalized, event_type, context

    return None, None, None



def _extract_headache_days_template(note):
    """
    Recover common neurology template fields such as:
      # headache days per month: 11/30 days
      # headache days per month: 25/30
      # headache days per month: daily with associated neuralgia pain

    This field has very high specificity and takes precedence over severe
    headache days and acute-medication-use fields.
    """
    if not isinstance(note, str):
        return None, None

    m = re.search(
        r"(?im)^\s*#?\s*headache\s+days?\s+per\s+month\s*:\s*([^\r\n]+)",
        note,
    )
    if not m:
        return None, None

    raw = m.group(1).strip()

    # Daily/everyday in the dedicated field follows the project's 28-day rule.
    if re.search(r"(?i)\b(?:daily|every\s+day|everyday)\b", raw):
        normalized = f"{DAYS_PER_MONTH}/month"
    else:
        normalized = None
        patterns = [
            r"(\d+(?:\.\d+)?\s*/\s*(?:28|30|31)\s*(?:days?)?)",
            r"(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*/\s*(?:28|30|31)\s*(?:days?)?)",
            r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*(?:/|per\s+)(?:month|mo))",
        ]
        for pat in patterns:
            cm = re.search(pat, raw, flags=re.IGNORECASE)
            if cm:
                normalized = _normalize_frequency_expression(cm.group(1))
                if normalized:
                    break

    if not normalized:
        return None, None

    left = max(0, m.start() - 180)
    right = min(len(note), m.end() + 300)
    evidence = re.sub(r"\s+", " ", note[left:right]).strip()
    return normalized, evidence


def _extract_qualitative_current_headache_frequency(note):
    """
    Narrow rescue rules for explicit current headache occurrence phrases.
    Avoids medication schedules and unrelated symptoms by requiring headache
    language in the same local phrase.
    """
    if not isinstance(note, str):
        return None, None

    patterns = [
        # "Headache is almost daily"
        (r"(?i)\bheadaches?\s+(?:is|are|occurs?|happens?)?\s*"
         r"(?:now\s+|currently\s+)?(?:almost|nearly)\s+daily\b",
         f"{DAYS_PER_MONTH}/month"),

        # "mild headache ... most of the time"
        (r"(?i)\bheadaches?\b.{0,100}\bmost\s+of\s+(?:the\s+)?time\b",
         f"{DAYS_PER_MONTH}/month"),

        # "~monthly" in a migraine/headache sentence
        (r"(?i)\b(?:migraines?|headaches?)\b.{0,220}?"
         r"(?:better\s+controlled\s+)?(?:about\s+|approximately\s+|~\s*)monthly\b",
         "1/month"),
    ]

    for pat, normalized in patterns:
        m = re.search(pat, note)
        if m:
            left = max(0, m.start() - 160)
            right = min(len(note), m.end() + 220)
            evidence = re.sub(r"\s+", " ", note[left:right]).strip()
            return normalized, evidence

    # Reverse-order monthly wording: "... ~monthly" after a migraine paragraph.
    for m in re.finditer(r"(?i)(?:~\s*)?monthly\b", note):
        left = max(0, m.start() - 350)
        local = note[left:m.end()]
        if re.search(r"(?i)\b(?:migraine|headache)s?\b", local):
            evidence = re.sub(r"\s+", " ", note[left:min(len(note), m.end()+120)]).strip()
            return "1/month", evidence

    return None, None

def deterministic_frequency_evidence_v3(note):
    """
    High-precision rescue layer.

    Priority:
      1) Dedicated '# headache days per month:' field
      2) Structured headache/migraine Summary -> Frequency field
      3) Explicit current narrative numeric frequency
      4) Explicit qualitative headache frequency (almost daily, monthly)
      5) Daily/continuous overall headache language
    """
    result = {
        "headache": None,
        "migraine": None,
        "headache_priority": 0,
        "migraine_priority": 0,
        "evidence": [],
    }
    if not isinstance(note, str) or not note.strip():
        return result

    is_migraine_phenotype, migraine_evidence = (
        _note_identifies_current_headache_phenotype_as_migraine(note)
    )

    # 1. Dedicated headache-days field.
    hd, hd_ev = _extract_headache_days_template(note)
    if hd:
        result["headache"] = hd
        result["headache_priority"] = 400
        result["evidence"].append(
            f"HEADACHE-DAYS TEMPLATE -> {hd}: {hd_ev}"
        )
        # Propagate only when current phenotype is explicitly migraine.
        if is_migraine_phenotype:
            result["migraine"] = hd
            result["migraine_priority"] = 400
            result["evidence"].append(
                "MIGRAINE PHENOTYPE LINK: " + " | ".join(migraine_evidence[:3])
            )

    # 2. Structured Summary -> Frequency.
    structured, structured_type, structured_ev = (
        _extract_structured_headache_frequency(note)
    )
    if structured and result["headache_priority"] < 300:
        result["headache"] = structured
        result["headache_priority"] = 300
        result["evidence"].append(
            f"STRUCTURED {structured_type.upper()} FREQUENCY -> "
            f"{structured}: {structured_ev}"
        )

    if structured and (
        structured_type == "migraine" or is_migraine_phenotype
    ) and result["migraine_priority"] < 300:
        result["migraine"] = structured
        result["migraine_priority"] = 300
        result["evidence"].append(
            f"STRUCTURED MIGRAINE LINK -> {structured}"
        )

    # 3. Explicit current narrative numeric frequency.
    narrative, event_type, narrative_ev = _extract_current_narrative_frequency(note)
    if narrative and result["headache_priority"] < 200:
        result["headache"] = narrative
        result["headache_priority"] = 200
        result["evidence"].append(
            f"CURRENT NARRATIVE HEADACHE FREQUENCY -> {narrative}: {narrative_ev}"
        )
    if narrative and (
        event_type == "migraine" or is_migraine_phenotype
    ) and result["migraine_priority"] < 200:
        result["migraine"] = narrative
        result["migraine_priority"] = 200
        result["evidence"].append(
            f"CURRENT NARRATIVE MIGRAINE FREQUENCY -> {narrative}: {narrative_ev}"
        )

    # 4. Qualitative current frequency.
    qualitative, qualitative_ev = _extract_qualitative_current_headache_frequency(note)
    if qualitative and result["headache_priority"] < 150:
        result["headache"] = qualitative
        result["headache_priority"] = 150
        result["evidence"].append(
            f"QUALITATIVE CURRENT HEADACHE FREQUENCY -> "
            f"{qualitative}: {qualitative_ev}"
        )

    # 5. Existing daily/continuous rescue.
    base = deterministic_frequency_evidence(note)
    if base["headache"] and result["headache_priority"] < 100:
        result["headache"] = base["headache"]
        result["headache_priority"] = 100
        result["evidence"].extend(base["evidence"])

    return result


def apply_deterministic_overrides(note, headache, migraine):
    """
    Apply high-precision deterministic rescue rules after LLM consolidation.

    Structured/current evidence can override an LLM miss or a less precise
    LLM result. The evidence is retained for audit.
    """
    ev = deterministic_frequency_evidence_v3(note)

    if ev["headache"] is not None:
        headache = ev["headache"]

    if ev["migraine"] is not None:
        migraine = ev["migraine"]

    return headache, migraine, ev["evidence"]


# ======================================================
# PARSE LLM OUTPUT
# ======================================================

def canonicalize_frequency(value):
    """Constrain model output to a small, auditable frequency vocabulary."""
    if not isinstance(value, str):
        return "Not mentioned"
    v = value.strip().splitlines()[0].strip()
    if re.search(r"not\s+mentioned|unknown|not\s+documented|unable", v, re.I):
        return "Not mentioned"
    # Keep only numeric monthly values/ranges; prevents explanatory spillover.
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*/?\s*(?:month|mo)\s*", v, re.I)
    if m:
        return f"{m.group(1)}-{m.group(2)}/month"
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*/?\s*(?:month|mo)\s*", v, re.I)
    if m:
        return f"{m.group(1)}/month"
    return "Not mentioned"


def extract_frequencies(text):
    """Parse only the requested output lines and reject free-form spillover."""
    if not isinstance(text, str):
        return "Not mentioned", "Not mentioned"
    hm = re.search(r"(?im)^\s*Headache\s+frequency\s*:\s*([^\r\n]+)", text)
    mm = re.search(r"(?im)^\s*Migraine\s+frequency\s*:\s*([^\r\n]+)", text)
    h = canonicalize_frequency(hm.group(1)) if hm else "Not mentioned"
    m = canonicalize_frequency(mm.group(1)) if mm else "Not mentioned"
    return h, m


def build_consolidation_prompt(candidates):
    """
    Build the consolidation prompt listing each chunk-level candidate
    extraction alongside the chunk text it came from.
    """

    lines = []

    for i, c in enumerate(candidates, start=1):
        # Keep consolidation prompts bounded. Preserve both ends because current
        # status/assessment language is often near the end of a clinical chunk.
        source = re.sub(r"\s+", " ", c["chunk_text"]).strip()
        if len(source) > 1400:
            source = source[:700] + " ... [middle omitted] ... " + source[-700:]
        lines.append(
            f"Candidate {i}:\n"
            f"  Headache frequency: {c['headache']}\n"
            f"  Migraine frequency: {c['migraine']}\n"
            f"  Source text: \"{source}\""
        )

    candidates_block = "\n\n".join(lines)

    return CONSOLIDATION_PROMPT.format(candidates=candidates_block)


# ======================================================
# LOAD MODEL
# ======================================================

llm = load_model()


# ======================================================
# STEP 1: SPLIT EVERY NOTE INTO OVERLAPPING CHUNKS
# ======================================================

print("Splitting notes into chunks...")

note_chunks = [chunk_text(note) for note in df[NOTE_COLUMN].tolist()]
n_notes = len(note_chunks)

flat_chunks = []
chunk_note_idx = []

for note_i, chunks in enumerate(note_chunks):
    for c in chunks:
        flat_chunks.append(c)
        chunk_note_idx.append(note_i)

print(f"Total notes: {n_notes:,}")
print(f"Total chunks: {len(flat_chunks):,}")


# ======================================================
# STEP 2: RUN LLM ON EVERY CHUNK (BATCHED)
# ======================================================

print("Extracting frequencies from each chunk...")

chunk_raw_outputs = run_in_batches(
    flat_chunks,
    generate_batch,
    BATCH_SIZE,
    llm,
    desc="Chunk extraction",
)

chunk_headache = []
chunk_migraine = []

for raw in chunk_raw_outputs:
    h, m = extract_frequencies(raw)
    chunk_headache.append(h)
    chunk_migraine.append(m)


# ======================================================
# STEP 3: GROUP CHUNK RESULTS BACK BY NOTE
# ======================================================

per_note_candidates = [[] for _ in range(n_notes)]

for flat_i, note_i in enumerate(chunk_note_idx):

    h = chunk_headache[flat_i]
    m = chunk_migraine[flat_i]

    # A chunk that found nothing contributes no information and is
    # dropped rather than passed on to consolidation.
    if h == "Not mentioned" and m == "Not mentioned":
        continue

    per_note_candidates[note_i].append({
        "headache": h,
        "migraine": m,
        "chunk_text": flat_chunks[flat_i],
        "raw_output": chunk_raw_outputs[flat_i],
    })


# ======================================================
# STEP 4: CONSOLIDATE CANDIDATES PER NOTE
# ======================================================

final_headache = ["Not mentioned"] * n_notes
final_migraine = ["Not mentioned"] * n_notes
consolidation_raw = [""] * n_notes
num_candidate_chunks = [0] * n_notes

notes_needing_consolidation = []

for note_i, candidates in enumerate(per_note_candidates):

    num_candidate_chunks[note_i] = len(candidates)

    if len(candidates) == 0:
        # No chunk found anything -- leave as "Not mentioned".
        continue

    if len(candidates) == 1:
        # Nothing to reconcile.
        final_headache[note_i] = candidates[0]["headache"]
        final_migraine[note_i] = candidates[0]["migraine"]
        consolidation_raw[note_i] = (
            "(single candidate chunk, no consolidation needed)"
        )
        continue

    notes_needing_consolidation.append(note_i)

print(
    f"Notes with 0 candidate chunks: "
    f"{sum(1 for c in num_candidate_chunks if c == 0):,}"
)
print(
    f"Notes with 1 candidate chunk: "
    f"{sum(1 for c in num_candidate_chunks if c == 1):,}"
)
print(
    f"Notes needing consolidation (2+ candidate chunks): "
    f"{len(notes_needing_consolidation):,}"
)

if notes_needing_consolidation:

    consolidation_prompts = [
        build_consolidation_prompt(per_note_candidates[i])
        for i in notes_needing_consolidation
    ]

    consolidation_outputs = run_in_batches(
        consolidation_prompts,
        run_prompts_batch,
        BATCH_SIZE,
        llm,
        desc="Consolidating candidates",
    )

    for note_i, raw in zip(notes_needing_consolidation, consolidation_outputs):

        h, m = extract_frequencies(raw)
        final_headache[note_i] = h
        final_migraine[note_i] = m
        consolidation_raw[note_i] = raw



# ======================================================
# STEP 5: APPLY HIGH-PRECISION DETERMINISTIC OVERRIDES
# ======================================================

deterministic_evidence = [""] * n_notes

for note_i, note in enumerate(df[NOTE_COLUMN].tolist()):
    h, m, evidence = apply_deterministic_overrides(
        note,
        final_headache[note_i],
        final_migraine[note_i],
    )
    final_headache[note_i] = h
    final_migraine[note_i] = m
    deterministic_evidence[note_i] = "\n".join(evidence)


# ======================================================
# ADD RESULTS
# ======================================================

df["num_chunks_IB"] = [len(c) for c in note_chunks]
df["num_candidate_chunks_IB"] = num_candidate_chunks

df["headache_frequency_IB"] = final_headache
df["migraine_frequency_IB"] = final_migraine
df["deterministic_evidence_IB"] = deterministic_evidence

df["chunk_candidates_IB"] = [
    "\n---\n".join(c["raw_output"] for c in candidates) if candidates else ""
    for candidates in per_note_candidates
]

df["consolidation_output_IB"] = consolidation_raw



# ======================================================
# EVALUATION: CC VS EXTRACTED HEADACHE FREQUENCY
# ======================================================

def parse_monthly_frequency_range(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower().replace("–", "-").replace("—", "-")
    if not text or text in {"not mentioned", "nan", "none", "error", "cuda out of memory"}:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*/?\s*(?:month|mo)?", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*/?\s*(?:month|mo)", text)
    if m:
        x = float(m.group(1)); return x, x
    return None

def compare_cc_to_headache_frequency(cc, predicted):
    if cc is None or (isinstance(cc, float) and pd.isna(cc)):
        return "Not evaluable", False, False, False
    try:
        cc_val = float(cc)
    except (TypeError, ValueError):
        return "Not evaluable", False, False, False
    rng = parse_monthly_frequency_range(predicted)
    if rng is None:
        return "Missed", False, False, False
    low, high = rng
    strict = low <= cc_val <= high
    daily_equiv = (cc_val == 28 and low <= 30 <= high) or (cc_val == 30 and low <= 28 <= high)
    if strict:
        return "Match", True, True, True
    if daily_equiv:
        return "Match (legacy daily equivalent)", True, False, True
    return "Mismatch", False, False, True

def add_cc_evaluation(df):
    if "CC" not in df.columns or "headache_frequency_IB" not in df.columns:
        print("WARNING: CC or headache_frequency_IB missing; skipping evaluation.")
        return df, {}
    statuses=[]; matches=[]; stricts=[]; extracted=[]
    for cc, pred in zip(df["CC"], df["headache_frequency_IB"]):
        status, match, strict, has_pred = compare_cc_to_headache_frequency(cc, pred)
        statuses.append(status); matches.append(match); stricts.append(strict); extracted.append(has_pred)
    df["CC_IB_status"] = statuses
    df["CC_IB_match"] = matches
    df["CC_IB_strict_match"] = stricts
    evaluable = pd.to_numeric(df["CC"], errors="coerce").notna().tolist()
    n=sum(evaluable); n_ext=sum(e and p for e,p in zip(evaluable,extracted)); n_match=sum(e and m for e,m in zip(evaluable,matches)); n_strict=sum(e and m for e,m in zip(evaluable,stricts))
    metrics={
      "evaluable_rows":n, "rows_with_extraction":n_ext, "matched_rows":n_match, "strict_matched_rows":n_strict,
      "extraction_sensitivity": n_ext/n if n else float("nan"),
      "range_aware_accuracy": n_match/n if n else float("nan"),
      "strict_accuracy": n_strict/n if n else float("nan"),
      "agreement_when_extracted": n_match/n_ext if n_ext else float("nan")
    }
    return df, metrics

# Run evaluation before saving so row-level flags are included in the CSV.
df, cc_metrics = add_cc_evaluation(df)
if cc_metrics:
    print("\nCC vs headache_frequency_IB evaluation:")
    print(f"  Evaluable rows: {cc_metrics['evaluable_rows']:,}")
    print(f"  Rows with extracted frequency: {cc_metrics['rows_with_extraction']:,}")
    print(f"  Extraction sensitivity/coverage: {cc_metrics['extraction_sensitivity']:.1%}")
    print(f"  Range-aware accuracy: {cc_metrics['range_aware_accuracy']:.1%}")
    print(f"  Strict accuracy: {cc_metrics['strict_accuracy']:.1%}")
    print(f"  Agreement when extracted: {cc_metrics['agreement_when_extracted']:.1%}")
    print(df["CC_IB_status"].value_counts(dropna=False).to_string())

# ======================================================
# SAVE
# ======================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n======================================")
print("Finished")
print("======================================")

print(
    f"Rows processed: {len(df):,}"
)

print(
    f"Output saved to:\n{OUTPUT_FILE}"
)


# ======================================================
# SUMMARY
# ======================================================

print("\nHeadache frequency:")
print(
    df["headache_frequency_IB"]
    .value_counts(dropna=False)
    .head(20)
)

print("\nMigraine frequency:")
print(
    df["migraine_frequency_IB"]
    .value_counts(dropna=False)
    .head(20)
)


# ======================================================
# PREVIEW
# ======================================================

print(
    df[
        [
            NOTE_COLUMN,
            "num_chunks_IB",
            "num_candidate_chunks_IB",
            "headache_frequency_IB",
            "migraine_frequency_IB",
            "deterministic_evidence_IB",
            "chunk_candidates_IB",
            "consolidation_output_IB",
        ]
    ].head(20).to_string()
)