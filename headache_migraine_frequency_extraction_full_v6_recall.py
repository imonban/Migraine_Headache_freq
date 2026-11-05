import pandas as pd
import torch
import re
import os
import traceback
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


MODEL_NAME = "Qwen/Qwen3-14B"

INPUT_FILE = "/media/Datacenter_storage/LLM_apigee/update_cd_noPHI_version2.xlsx"
OUTPUT_FILE = "/media/Datacenter_storage/LLM_apigee/update_cd_noPHI_version2_Qwen.csv"

# Robust/resumable output settings.
CHECKPOINT_EVERY_BATCHES = 1
SAVE_PARTIAL_ON_ERROR = True


NOTE_COLUMN = "String"

# Batch size for generation.
BATCH_SIZE = 4

# Each note is split into overlapping chunks (see CHUNKING below)
# before extraction, so individual prompts are much shorter than a
# full note -- this token budget is mostly a safety net.
MAX_INPUT_TOKENS = 3000
MAX_NEW_TOKENS = 220

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
# ROBUST / ATOMIC SAVING
# ======================================================

def ensure_output_directory(output_file):
    """Create the output directory if it does not already exist."""
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)


def save_checkpoint(dataframe, output_file, label="checkpoint"):
    """
    Atomically save the current dataframe.

    The CSV is first written to '<output>.tmp' and then moved into place.
    This avoids leaving a partially written CSV if the process is interrupted
    during the write itself.
    """
    ensure_output_directory(output_file)

    tmp_file = output_file + ".tmp"
    dataframe.to_csv(tmp_file, index=False)
    os.replace(tmp_file, output_file)

    size_bytes = os.path.getsize(output_file)
    print(
        f"[SAVE] {label}: {len(dataframe):,} rows -> "
        f"{output_file} ({size_bytes:,} bytes)"
    )


def safe_partial_save(dataframe, output_file, label="partial save"):
    """Best-effort save used during exception handling."""
    try:
        save_checkpoint(dataframe, output_file, label=label)
    except Exception as save_error:
        print(
            f"[SAVE ERROR] Could not write partial output to {output_file}: "
            f"{type(save_error).__name__}: {save_error}"
        )


# ======================================================
# LOAD DATA
# ======================================================

if INPUT_FILE.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(INPUT_FILE)
else:
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

# Validate output path before expensive model inference.
save_checkpoint(df, OUTPUT_FILE, label="initial cleaned input")


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
You are a clinical NLP extractor. Extract the CURRENT overall headache
frequency and CURRENT migraine frequency from the clinical text.

Work evidence-first:
1. Identify the exact source phrase for overall headache frequency.
2. Identify the exact source phrase for migraine frequency.
3. Determine whether each phrase is CURRENT, HISTORICAL, or UNCLEAR.
4. Normalize CURRENT frequency to a 28-day/month convention.

CRITICAL RULES:
- Prefer current/now/interval-history information over prior, baseline,
  pre-treatment, "used to", or older dated information.
- Prefer a dedicated current structured field such as "Current headache
  frequency", "# headache days per month", "Total headache days per month",
  or "Summary of headache: Frequency" over less specific narrative text.
- Headache and migraine frequency are NOT automatically the same.
- If the note gives total headache days plus a migraine/severe subset, keep
  them separate.
- Migraine attacks are headaches, but do not replace a clearly documented
  overall headache-day frequency with a less frequent migraine subset.
- Do not multiply attack frequency by duration. Example: 4 migraine
  attacks/month lasting 2-3 days = 4 attacks/month unless headache DAYS are
  explicitly documented.
- Do not use medication schedules, treatment frequency, follow-up intervals,
  family/social history, or unrelated symptom frequency.
- Do not confuse duration with frequency.
- "daily", "every day", chronic daily headache, constant/continuous headache
  = 28/month under this project's 28-day convention.
- every other day/bidaily = 14/month.
- weekly = 4/month; once/week = 4/month; 2-3/week = 8-12/month.
- "a couple" = 2; "a few" = 2-3 when explicitly describing attacks/month.
- X headache-free days in the last 30 can be converted to 30-X affected days.
- If current documentation says no headaches in the last month, headache
  frequency is 0/month even if prior headache history was frequent.
- A diagnosis of chronic migraine alone does NOT mean 28 migraine days/month.

Return EXACTLY these six lines and nothing else:

Headache frequency: <value>/month OR Not mentioned
Headache evidence: <short exact source phrase OR Not mentioned>
Headache temporal status: <CURRENT|HISTORICAL|UNCLEAR>
Migraine frequency: <value>/month OR Not mentioned
Migraine evidence: <short exact source phrase OR Not mentioned>
Migraine temporal status: <CURRENT|HISTORICAL|UNCLEAR>

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
# V9 HIGH-RECALL CANDIDATE -> VERIFY -> RESCUE PIPELINE
# ======================================================

FREQUENCY_ANCHOR_RE = re.compile(
    r"""(?ix)
    \b(?:daily|every\s+day|everyday|weekly|monthly|constant|continuous|
       headache[- ]free|most\s+of\s+the\s+time|almost\s+daily|
       nearly\s+daily|every\s+other\s+day|bidaily|few|couple|once|twice)\b
    |
    \b\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*
       (?:x|times?|days?|attacks?)?\s*
       (?:/|per\s+|a\s+)?(?:week|wk|month|mo|day)\b
    |
    \b\d+(?:\.\d+)?\s*/\s*(?:28|30|31)\s*(?:days?)?\b
    """
)

HEADACHE_TERMS_RE = re.compile(
    r"(?i)\b(?:headaches?|h/?a|cephalgia|head\s+pain|headache\s+days?)\b"
)

MIGRAINE_TERMS_RE = re.compile(
    r"""(?ix)\b(?:
        migraines?|migraine\s+attacks?|migraine\s+days?|
        migrainous|debilitating\s+headaches?|severe\s+headaches?|
        bad\s+migraines?|complicated\s+migraine|
        episodic\s+migraine|chronic\s+migraine
    )\b"""
)

EXCLUDE_CONTEXT_RE = re.compile(
    r"""(?ix)\b(?:
        take|tablet|capsule|dose|dosing|medication|maximum|max|
        refill|injection|botox\s+every|follow[- ]?up|f/u|
        family\s+history|mother|father|sister|brother|
        exercise|alcohol|smoking
    )\b"""
)


def _sentence_spans(text):
    """Simple clinical sentence/line segmentation preserving source wording."""
    if not isinstance(text, str):
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def retrieve_frequency_windows(note, entity, radius=2):
    """
    High-recall retrieval. Regex retrieves evidence; it does NOT decide the
    final frequency. This intentionally over-retrieves, then Qwen verifies.
    """
    sents = _sentence_spans(note)
    if not sents:
        return []

    entity_re = MIGRAINE_TERMS_RE if entity == "migraine" else HEADACHE_TERMS_RE
    hit_idx = set()

    for i, sent in enumerate(sents):
        has_freq = bool(FREQUENCY_ANCHOR_RE.search(sent))
        has_entity = bool(entity_re.search(sent))

        # Strong direct hit.
        if has_freq and has_entity:
            hit_idx.add(i)

        # Frequency-only lines such as "Frequency: 10-12/month" are retained
        # when adjacent to an entity-specific heading/sentence.
        if has_freq:
            lo = max(0, i - 2)
            hi = min(len(sents), i + 3)
            neighborhood = " ".join(sents[lo:hi])
            if entity_re.search(neighborhood):
                hit_idx.add(i)

        # Structured fields can be entity-free on the same line.
        if re.search(
            r"(?i)\b(?:current\s+headache\s+frequency|frequency|"
            r"headache\s+days?\s+per\s+month|migraine\s+days?\s+per\s+month|"
            r"headache[- ]free\s+days?)\s*:",
            sent,
        ):
            hit_idx.add(i)

    windows = []
    seen = set()
    for i in sorted(hit_idx):
        lo = max(0, i - radius)
        hi = min(len(sents), i + radius + 1)
        window = " ".join(sents[lo:hi]).strip()
        if window and window not in seen:
            seen.add(window)
            windows.append(window)

    return windows


CANDIDATE_PROMPT = """
You are performing HIGH-RECALL candidate retrieval for {entity} frequency.

Do NOT choose the final answer yet.
Find every passage that could possibly quantify the PATIENT'S {entity}
occurrence, including uncertain candidates.

Include:
- numeric weekly/monthly/daily frequency
- headache/migraine days
- attacks
- once/twice/few/couple
- daily/constant/continuous/near-daily
- headache-free-day statements
- severe/debilitating headache subsets when evaluating migraine

Exclude obvious medication dosing, family history, follow-up schedules, and
duration-only statements.

Return one candidate per line exactly as:
CANDIDATE: <verbatim evidence>

If none:
CANDIDATE: NONE

TEXT:
{context}
"""


VERIFY_PROMPT = """
You are verifying ONE candidate for CURRENT {entity} frequency.

Candidate evidence:
{candidate}

Local source context:
{context}

Classify it carefully.

ENTITY:
- Does this quantify the patient's {entity} occurrence?
- For migraine, generic headache frequency is NOT sufficient.
- Explicit migraine terminology is strongest evidence.
- A severe/debilitating/bad headache frequency may count as migraine when the
  SAME LOCAL PASSAGE explicitly links that subset to migraine.
- A generic headache frequency may count as migraine when the SAME LOCAL
  structured passage, assessment, or transition clearly establishes that the
  quantified headaches are the migraine phenotype.
- Example: "Assessment: migraine without aura ... currently 1-2 headaches/week"
  can support migraine frequency when this is one continuous local phenotype.
- Example: "headache becomes throbbing migraine ... 5-10 days/month" can support
  migraine frequency when the frequency describes that linked phenotype.
- Do NOT infer migraine frequency merely from a remote problem-list/PMH/family
  history migraine diagnosis elsewhere in the note.
- For headache, migraine attacks may count as headache occurrence only when no
  separate overall headache burden contradicts that interpretation.

TEMPORAL:
- CURRENT = current/now/latest interval/last month/present burden.
- HISTORICAL = used to, prior to treatment, baseline, old dated history.
- UNCLEAR = cannot establish current status.

TYPE:
- FREQUENCY = occurrence count/rate.
- DURATION = how long an attack lasts.
- MEDICATION = treatment schedule/use limit.
- OTHER = unrelated.

Extract the raw frequency phrase WITHOUT doing arithmetic.

Return EXACTLY:
Valid: <YES|NO>
Temporal: <CURRENT|HISTORICAL|UNCLEAR>
Type: <FREQUENCY|DURATION|MEDICATION|OTHER>
Raw frequency: <verbatim frequency phrase OR Not mentioned>
Evidence: <short verbatim evidence>
"""


RESCUE_PROMPT = """
The first-pass pipeline did NOT find a CURRENT {entity} frequency.

Search aggressively for any missed CURRENT {entity} frequency in this clinical
text. This is a recall-rescue pass.

Look for:
- days/month, attacks/month
- x/week, times/week
- daily, every day, constant, continuous
- every other day, near daily
- once/twice/few/couple
- headache-free days
- structured Frequency fields
- severe/debilitating subset ONLY if explicitly linked to migraine
- explicit migraine/migrainous wording near the quantified frequency
- transitions such as "used to X, now Y"

For MIGRAINE rescue specifically:
- Do NOT use generic headache frequency.
- Do NOT use a severe/debilitating headache count unless the source explicitly
  says those severe/debilitating events are migraines.
- A migraine diagnosis elsewhere in the note does not make every headache a
  migraine.

Reject medication schedules, duration, family history, and historical-only
values.

Return EXACTLY:
{label} frequency: <normalized value>/month OR Not mentioned
{label} evidence: <short exact evidence OR Not mentioned>
{label} temporal status: <CURRENT|HISTORICAL|UNCLEAR>

Clinical text:
{context}
"""


def parse_candidate_lines(output):
    if not isinstance(output, str):
        return []
    vals = []
    for line in output.splitlines():
        m = re.match(r"\s*CANDIDATE\s*:\s*(.+)", line, flags=re.I)
        if m:
            v = m.group(1).strip()
            if v.upper() != "NONE":
                vals.append(v)
    return vals


def parse_verified_candidate(output):
    result = {
        "valid": False,
        "temporal": "UNCLEAR",
        "type": "OTHER",
        "raw_frequency": None,
        "evidence": None,
    }
    if not isinstance(output, str):
        return result

    patterns = {
        "valid": r"(?im)^\s*Valid\s*:\s*(YES|NO)",
        "temporal": r"(?im)^\s*Temporal\s*:\s*(CURRENT|HISTORICAL|UNCLEAR)",
        "type": r"(?im)^\s*Type\s*:\s*(FREQUENCY|DURATION|MEDICATION|OTHER)",
        "raw_frequency": r"(?im)^\s*Raw\s+frequency\s*:\s*(.+)$",
        "evidence": r"(?im)^\s*Evidence\s*:\s*(.+)$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, output)
        if not m:
            continue
        v = m.group(1).strip()
        if key == "valid":
            result[key] = v.upper() == "YES"
        elif key in {"temporal", "type"}:
            result[key] = v.upper()
        elif v.lower() != "not mentioned":
            result[key] = v

    return result


def normalize_verified_frequency(raw):
    """
    Deterministic normalization. Qwen identifies the raw expression; Python
    performs arithmetic so model math cannot alter the result.
    """
    if not isinstance(raw, str):
        return None

    s = raw.lower().strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)

    # Dedicated qualitative forms.
    if re.search(r"\b(?:daily|every day|everyday|constant|continuous)\b", s):
        if re.search(r"\bevery other day\b", s):
            return "14/month"
        return f"{DAYS_PER_MONTH}/month"

    if re.search(r"\b(?:almost|nearly)\s+daily\b", s):
        return f"{DAYS_PER_MONTH}/month"

    if re.search(r"\bweekly\b", s) and not re.search(r"\d", s):
        return "4/month"
    if re.search(r"\bmonthly\b", s) and not re.search(r"\d", s):
        return "1/month"

    if re.search(r"\bonce\s+(?:a|per)\s+week\b", s):
        return "4/month"
    if re.search(r"\btwice\s+(?:a|per)\s+week\b", s):
        return "8/month"

    if re.search(r"\b(?:a\s+)?couple\b.*\bmonth\b", s):
        return "2/month"
    if re.search(r"\b(?:a\s+)?few\b.*\bmonth\b", s):
        return "2-3/month"

    # X/Y days => numerator per month.
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(?:28|30|31)\s*(?:days?)?", s)
    if m:
        return f"{m.group(1)}/month"

    # Monthly numeric/range.
    m = re.search(
        r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*"
        r"(?:x|times?|days?|attacks?)?\s*(?:/|per\s+|a\s+)?"
        r"(?:month|mo)\b",
        s,
    )
    if m:
        return (
            f"{m.group(1)}-{m.group(2)}/month"
            if m.group(2)
            else f"{m.group(1)}/month"
        )

    # Weekly numeric/range.
    m = re.search(
        r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*"
        r"(?:x|times?|days?|attacks?)?\s*(?:/|per\s+|a\s+)?"
        r"(?:week|wk)\b",
        s,
    )
    if m:
        lo = float(m.group(1)) * 4
        hi = float(m.group(2)) * 4 if m.group(2) else None

        def fmt(x):
            return str(int(x)) if float(x).is_integer() else f"{x:g}"

        return (
            f"{fmt(lo)}-{fmt(hi)}/month"
            if hi is not None
            else f"{fmt(lo)}/month"
        )

    return None


def migraine_evidence_gate(item):
    """
    V12 selective-recall migraine gate.

    Preserve V11's high specificity while recovering additional true migraine
    frequencies through tightly local evidence.

    ACCEPT when any of the following is true:

    A) The quantified evidence explicitly says migraine/migrainous.

    B) Severe/debilitating/bad headache is locally and explicitly linked to
       migraine.

    C) A quantified generic headache statement occurs in the SAME local
       structured migraine phenotype (e.g. Summary/Assessment/Impression).

    D) A local transition explicitly describes the same migraine phenotype,
       even if the numeric clause itself says only headache:
          "migraines ... now 1-2 headaches/week"
          "migraine without aura ... frequency 4-8/month"

    E) The local passage explicitly states that headaches transition/become
       migraine or that a quantified headache subset is migraine.

    REJECT remote PMH/problem-list/family-history migraine mentions.
    """
    candidate = str(item.get("candidate") or "")
    evidence = str(item.get("evidence") or "")
    context = str(item.get("context") or "")
    direct = f"{candidate} {evidence}"
    local = f"{candidate} {evidence} {context}"

    # A: explicit migraine terminology in quantified evidence.
    if re.search(
        r"(?i)\b(?:migraines?|migraine\s+attacks?|migraine\s+days?|migrainous)\b",
        direct,
    ):
        return True, "A: explicit migraine terminology"

    # Reject contexts that are only remote history/problem-list/family history.
    remote = bool(re.search(
        r"(?i)\b(?:family\s+history|mother|father|sister|brother|"
        r"past\s+medical\s+history|problem\s+list|PMH)\b",
        context,
    ))
    current_anchor = bool(re.search(
        r"(?i)\b(?:current|currently|now|today|interval\s+history|"
        r"frequency|assessment|impression|summary\s+of|headache\s+history)\b",
        context,
    ))
    if remote and not current_anchor:
        return False, "remote migraine mention only"

    # B: severe/debilitating/bad headache locally linked to migraine.
    severe = bool(re.search(
        r"(?i)\b(?:severe|debilitating|bad)\s+"
        r"(?:headaches?|ones?|attacks?|episodes?)\b",
        direct,
    ))
    if severe and re.search(
        r"""(?ix)
        (?:severe|debilitating|bad).{0,180}\bmigraines?\b
        |
        \bmigraines?\b.{0,180}(?:severe|debilitating|bad)
        |
        (?:severe|debilitating|bad)\s+(?:headaches?|ones?)
        .{0,120}(?:called?|considered?|represent|are|being|consistent\s+with)
        .{0,40}\bmigraines?\b
        """,
        local,
    ):
        return True, "B: severe subset locally linked to migraine"

    # Does the candidate actually quantify headache occurrence?
    quantified_headache = bool(
        HEADACHE_TERMS_RE.search(direct)
        and FREQUENCY_ANCHOR_RE.search(direct)
    )

    # Strong local migraine phenotype.
    migraine_phenotype = bool(re.search(
        r"""(?ix)
        \b(?:episodic|chronic)?\s*migraine(?:s)?
        (?:\s+without\s+aura|\s+with\s+aura)?\b
        |
        \bmigraine\s+headaches?\b
        |
        \bsummary\s+of\s+migraine\s+headaches?\b
        """,
        local,
    ))

    # C: structured local phenotype linkage.
    structured_link = bool(re.search(
        r"""(?ix)
        (?:assessment|impression|diagnosis)\s*:?\s*
        (?:episodic|chronic)?\s*migraine
        |
        summary\s+of\s+(?:migraine\s+)?headaches?
        |
        headaches?.{0,120}(?:consistent\s+with|represent|are)
        .{0,50}migraines?
        """,
        local,
    ))
    if quantified_headache and migraine_phenotype and structured_link:
        return True, "C: quantified headache in same structured migraine phenotype"

    # D: local current transition / shared phenotype.
    # Allows a numeric headache clause when migraine is explicitly the topic
    # within the same short passage.
    transition_link = bool(re.search(
        r"""(?ix)
        \bmigraines?\b.{0,220}
        (?:currently|now|frequency|occur|occurring|averag|having|gets?)
        .{0,120}\bheadaches?\b
        |
        \bheadaches?\b.{0,160}
        (?:migraine\s+without\s+aura|migraine\s+with\s+aura|migraines?)
        .{0,120}(?:frequency|currently|now|occur)
        """,
        local,
    ))
    if quantified_headache and migraine_phenotype and transition_link:
        return True, "D: local shared migraine/headache phenotype transition"

    # E: explicit transformation/linkage between headache and migraine.
    transformation = bool(re.search(
        r"""(?ix)
        headaches?.{0,120}(?:transition|transform|become|turn|progress)
        .{0,80}migraines?
        |
        migraines?.{0,120}(?:present|manifest|occur)\s+as
        .{0,80}headaches?
        |
        headache\s+subset.{0,100}migraines?
        """,
        local,
    ))
    if quantified_headache and transformation:
        return True, "E: explicit headache-to-migraine linkage"

    return False, "insufficient local migraine linkage"


def validate_rescue_migraine_evidence(note, rescue):
    """
    V11 rescue gate: explicit migraine evidence is accepted; severe/generic
    headache evidence can also pass when a narrow local neighborhood strongly
    establishes the same phenotype as migraine.
    """
    if not rescue or rescue.get("frequency") == "Not mentioned":
        return False, "no rescue frequency"

    evidence = str(rescue.get("evidence") or "")
    if not evidence or evidence.lower() == "not mentioned":
        return False, "no quoted rescue evidence"

    # Explicit migraine evidence.
    if re.search(
        r"(?i)\b(?:migraines?|migraine\s+attacks?|migraine\s+days?|migrainous)\b",
        evidence,
    ):
        return True, "A: explicit migraine evidence in rescue"

    # Locate evidence and use only a narrow neighborhood, preventing a remote
    # migraine diagnosis from validating unrelated headache frequency.
    idx = note.lower().find(evidence.lower())
    if idx < 0:
        return False, "rescue evidence not found verbatim in note"

    local = note[max(0, idx - 320):min(len(note), idx + len(evidence) + 320)]

    proxy_item = {
        "candidate": evidence,
        "evidence": evidence,
        "context": local,
    }
    ok, reason = migraine_evidence_gate(proxy_item)
    return ok, f"rescue {reason}"


def choose_verified_frequency(verified, entity):
    """
    Select CURRENT verified candidates.

    For migraine, apply an additional precision gate after Qwen verification.
    This preserves V9's high-recall retrieval but prevents generic headache
    frequency from being promoted to migraine frequency.
    """
    eligible = []

    for item in verified:
        if not item["valid"]:
            continue
        if item["temporal"] != "CURRENT":
            continue
        if item["type"] != "FREQUENCY":
            continue

        if entity == "migraine":
            gate_ok, gate_reason = migraine_evidence_gate(item)
            item["migraine_gate_pass"] = gate_ok
            item["migraine_gate_reason"] = gate_reason
            if not gate_ok:
                continue

        norm = normalize_verified_frequency(item["raw_frequency"])
        if norm:
            eligible.append((norm, item))

    if not eligible:
        return "Not mentioned", "Not mentioned", []

    # Later current candidates are preferred, as in V9.
    norm, best = eligible[-1]
    return norm, best.get("evidence") or "Not mentioned", eligible


def run_v9_entity_pass(note, entity, llm):
    """
    Separate entity-specific candidate retrieval and verification.
    """
    windows = retrieve_frequency_windows(note, entity)
    if not windows:
        return {
            "frequency": "Not mentioned",
            "evidence": "Not mentioned",
            "candidates": [],
            "verified": [],
        }

    # Candidate generation from retrieved windows.
    candidate_prompts = [
        CANDIDATE_PROMPT.format(entity=entity, context=w)
        for w in windows
    ]
    candidate_outputs = run_prompts_batch(candidate_prompts, llm)

    candidates = []
    for w, out in zip(windows, candidate_outputs):
        for c in parse_candidate_lines(out):
            candidates.append({"candidate": c, "context": w})

    # Deduplicate while preserving order.
    dedup = []
    seen = set()
    for c in candidates:
        key = c["candidate"].strip().lower()
        if key not in seen:
            seen.add(key)
            dedup.append(c)
    candidates = dedup

    if not candidates:
        return {
            "frequency": "Not mentioned",
            "evidence": "Not mentioned",
            "candidates": [],
            "verified": [],
        }

    verify_prompts = [
        VERIFY_PROMPT.format(
            entity=entity,
            candidate=c["candidate"],
            context=c["context"],
        )
        for c in candidates
    ]
    verify_outputs = run_prompts_batch(verify_prompts, llm)

    verified = []
    for c, out in zip(candidates, verify_outputs):
        v = parse_verified_candidate(out)
        v["candidate"] = c["candidate"]
        v["context"] = c["context"]
        v["raw_verifier_output"] = out
        verified.append(v)

    freq, evidence, eligible = choose_verified_frequency(verified, entity)

    return {
        "frequency": freq,
        "evidence": evidence,
        "candidates": candidates,
        "verified": verified,
        "eligible": eligible,
    }


def run_v9_rescue(note, entity, llm):
    """
    Second pass only when the entity remains Not mentioned.
    """
    label = "Headache" if entity == "headache" else "Migraine"
    prompt = RESCUE_PROMPT.format(
        entity=entity,
        label=label,
        context=note,
    )
    raw = run_prompts_batch([prompt], llm)[0]

    pat = rf"{label}\s+frequency\s*:\s*(.+)"
    m = re.search(pat, raw, flags=re.I)
    freq = m.group(1).strip() if m else "Not mentioned"

    evm = re.search(
        rf"{label}\s+evidence\s*:\s*(.+)",
        raw,
        flags=re.I,
    )
    evidence = evm.group(1).strip() if evm else "Not mentioned"

    tm = re.search(
        rf"{label}\s+temporal\s+status\s*:\s*"
        r"(CURRENT|HISTORICAL|UNCLEAR)",
        raw,
        flags=re.I,
    )
    temporal = tm.group(1).upper() if tm else "UNCLEAR"

    if temporal != "CURRENT":
        freq = "Not mentioned"

    return {
        "frequency": freq,
        "evidence": evidence,
        "temporal": temporal,
        "raw_output": raw,
    }


def run_v9_note(note, llm):
    """
    Complete V9 extraction for one note.
    """
    h = run_v9_entity_pass(note, "headache", llm)
    m = run_v9_entity_pass(note, "migraine", llm)

    h_rescue = None
    m_rescue = None

    if h["frequency"] == "Not mentioned":
        h_rescue = run_v9_rescue(note, "headache", llm)
        if h_rescue["frequency"] != "Not mentioned":
            h["frequency"] = h_rescue["frequency"]
            h["evidence"] = h_rescue["evidence"]

    if m["frequency"] == "Not mentioned":
        m_rescue = run_v9_rescue(note, "migraine", llm)
        gate_ok, gate_reason = validate_rescue_migraine_evidence(note, m_rescue)
        m_rescue["migraine_gate_pass"] = gate_ok
        m_rescue["migraine_gate_reason"] = gate_reason

        if m_rescue["frequency"] != "Not mentioned" and gate_ok:
            m["frequency"] = m_rescue["frequency"]
            m["evidence"] = m_rescue["evidence"]

    # Existing deterministic rules remain the final high-confidence QA/rescue.
    det = deterministic_frequency_evidence_v3(note)

    if det.get("headache") is not None:
        h["frequency"] = det["headache"]
    if det.get("migraine") is not None:
        det_text = " ".join(det.get("evidence", []))
        if re.search(
            r"(?i)\b(?:migraines?|migraine\s+attacks?|migraine\s+days?|migrainous)\b",
            det_text,
        ):
            m["frequency"] = det["migraine"]

    # Final V11 migraine precision check.
    # Do not require the word "migraine" on the evidence line itself; permit a
    # strongly linked local phenotype, but reject remote diagnosis propagation.
    if m["frequency"] != "Not mentioned":
        final_ev = str(m.get("evidence") or "")
        idx = note.lower().find(final_ev.lower()) if final_ev and final_ev.lower() != "not mentioned" else -1
        local = (
            note[max(0, idx - 320):min(len(note), idx + len(final_ev) + 320)]
            if idx >= 0 else final_ev
        )
        proxy = {
            "candidate": final_ev,
            "evidence": final_ev,
            "context": local,
        }
        gate_ok, gate_reason = migraine_evidence_gate(proxy)

        # Deterministic explicit migraine evidence remains an alternative path.
        det_text = " ".join(det.get("evidence", []))
        det_explicit = bool(re.search(
            r"(?i)\b(?:migraines?|migraine\s+attacks?|migraine\s+days?|migrainous)\b",
            det_text,
        ))

        if not gate_ok and not det_explicit:
            m["frequency"] = "Not mentioned"


    return {
        "headache_frequency": h["frequency"],
        "migraine_frequency": m["frequency"],
        "headache_evidence": h["evidence"],
        "migraine_evidence": m["evidence"],
        "headache_candidates": h.get("candidates", []),
        "migraine_candidates": m.get("candidates", []),
        "headache_verified": h.get("verified", []),
        "migraine_verified": m.get("verified", []),
        "headache_rescue": h_rescue,
        "migraine_rescue": m_rescue,
        "deterministic_evidence": det.get("evidence", []),
    }

# ======================================================
# LOAD MODEL
# ======================================================

def load_model():
    print("Loading Qwen3-14B model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    print("Model loaded.")
    return {"model": model, "tokenizer": tokenizer}


# ======================================================
# LOW-LEVEL BATCHED GENERATION
# ======================================================

def run_prompts_batch(prompts, llm):
    """
    Qwen3 batched generation using the official chat template.
    Non-thinking mode is used because this task requires short, constrained
    structured extraction rather than visible chain-of-thought.
    """
    if not prompts:
        return []

    model = llm["model"]
    tokenizer = llm["tokenizer"]

    try:
        chat_prompts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You extract clinical headache and migraine frequency. "
                        "Follow the requested output schema exactly. /no_think"
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            chat_prompts.append(rendered)

        inputs = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        results = []
        for seq in generated:
            answer = tokenizer.decode(
                seq[input_len:],
                skip_special_tokens=True,
            ).strip()
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            results.append(clean_text(answer))
        return results

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return [
            "Headache frequency: CUDA Out of Memory\\n"
            "Migraine frequency: CUDA Out of Memory"
        ] * len(prompts)

    except Exception as e:
        print(f"Generation error: {type(e).__name__}: {e}")
        return [
            "Headache frequency: ERROR\\n"
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


def run_in_batches(
    items,
    fn,
    batch_size,
    llm,
    desc,
    checkpoint_callback=None,
):
    """
    Apply fn(batch, llm) to items in batches.

    If checkpoint_callback is provided, it is called after each completed
    batch. This makes long GPU jobs resumable/inspectable instead of waiting
    until the entire script finishes before anything is saved.
    """
    results = [None] * len(items)
    n_batches = (len(items) + batch_size - 1) // batch_size

    for batch_number, batch_start in enumerate(
        tqdm(
            range(0, len(items), batch_size),
            total=n_batches,
            desc=desc,
        ),
        start=1,
    ):
        batch_end = min(batch_start + batch_size, len(items))
        batch = items[batch_start:batch_end]

        try:
            batch_results = fn(batch, llm)
            results[batch_start:batch_end] = batch_results

        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] {desc} stopped at batch {batch_number}.")
            if checkpoint_callback is not None:
                checkpoint_callback(
                    batch_number,
                    batch_start,
                    batch_end,
                    results,
                    interrupted=True,
                )
            raise

        except Exception as e:
            print(
                f"\n[BATCH ERROR] {desc}, batch {batch_number}: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()

            if checkpoint_callback is not None:
                checkpoint_callback(
                    batch_number,
                    batch_start,
                    batch_end,
                    results,
                    interrupted=True,
                )
            raise

        if (
            checkpoint_callback is not None
            and batch_number % CHECKPOINT_EVERY_BATCHES == 0
        ):
            checkpoint_callback(
                batch_number,
                batch_start,
                batch_end,
                results,
                interrupted=False,
            )

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

def _extract_v7_string_patterns(note):
    """High-precision rescue rules for additional real-world headache strings."""
    r={"headache":None,"migraine":None,"headache_priority":0,"migraine_priority":0,"evidence":[]}
    if not isinstance(note,str): return r
    t=re.sub(r"[ \t]+"," ",note)

    def setv(kind,val,p,ev):
        key=kind+"_priority"
        if p>r[key]:
            r[kind]=val; r[key]=p
            r["evidence"].append(f"V7 {kind.upper()} -> {val}: {ev}")

    def wk(a,b=None):
        a=float(a)*4
        b=float(b)*4 if b else None
        f=lambda x:str(int(x)) if x.is_integer() else f"{x:g}"
        return f"{f(a)}-{f(b)}/month" if b is not None else f"{f(a)}/month"

    # Explicit headache-day counts and current headache-frequency fields.
    pats=[
      r"(?i)\bout\s+of\s+(?:the\s+)?last\s+(?:28|30|31)\s+days?\s+only\s+(\d+)\s+headaches?",
      r"(?i)\b(?:total\s+)?#?\s*h/?a\s+days?\s+per\s+month\s*:\s*(\d+)(?:\s*-\s*(\d+))?",
      r"(?i)\btotal\s+headache\s+days?\s+per\s+month\s*:\s*(\d+)(?:\s*/\s*(?:28|30|31))?",
      r"(?i)\baverage\s+number\s+of\s+headache\s+days?\s+per\s+month\s*:\s*(\d+)",
      r"(?i)\b(?:has|reports?)\s+(\d+)\s+headache\s+days?\s+(?:a|per)\s+month\b",
      r"(?i)\bcurrent\s+headache\s+frequency\s*:\s*-?\s*(\d+)(?:\s*-\s*(\d+))?\s+headache\s+days?\s+(?:per|pe)\s+month\b",
    ]
    for pat in pats:
        for m in re.finditer(pat,t):
            a=m.group(1); b=m.group(2) if (m.lastindex or 0)>=2 else None
            val=f"{a}-{b}/month" if b else f"{a}/month"
            setv("headache",val,540,m.group(0))

    # Headache-free days in a stated 30-day interval.
    m=re.search(r"(?i)headache[- ]free\s+days?\s+in\s+last\s+30\s*:?\s*(\d+)(?:\s*-\s*(\d+))?",t)
    if m:
        lo=float(m.group(1)); hi=float(m.group(2)) if m.group(2) else lo
        a=30-hi; b=30-lo
        f=lambda x:str(int(x)) if x.is_integer() else f"{x:g}"
        val=f"{f(a)}-{f(b)}/month" if a!=b else f"{f(a)}/month"
        setv("headache",val,545,m.group(0))

    # Explicit migraine days/attacks/month.
    mpats=[
      r"(?i)\b(\d+)(?:\s*-\s*(\d+))?\s+migraine\s+days?\s+(?:a|per)\s+month\b",
      r"(?i)\b(\d+)(?:\s*-\s*(\d+))?\s+migraine\s+attacks?\s+(?:a|per)\s+month\b",
      r"(?i)\b(?:has|gets?|experiences?|reports?)\s+(\d+)(?:\s*-\s*(\d+))?\s+migraines?\s+(?:a|per)\s+month\b",
      r"(?i)\bcurrent\s+headache\s+frequency\s*:\s*(\d+)(?:\s*-\s*(\d+))?\s+migraines?\s+(?:in\s+a|per)\s+month\b",
    ]
    for pat in mpats:
        for m in re.finditer(pat,t):
            a,b=m.group(1),m.group(2)
            val=f"{a}-{b}/month" if b else f"{a}/month"
            setv("migraine",val,535,m.group(0))
            setv("headache",val,300,m.group(0))

    # Entity-first weekly numeric expressions.
    ep=re.compile(r"(?i)\b(migraines?|headaches?|tension\s+headaches?|h/?a)\b.{0,60}?(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(?:x|times?|days?)?\s*(?:/|per\s+|a\s+)?(?:week|wk)\b")
    for m in ep.finditer(t):
        val=wk(m.group(2),m.group(3)); ent=m.group(1).lower()
        if "migraine" in ent:
            setv("migraine",val,500,m.group(0)); setv("headache",val,280,m.group(0))
        else: setv("headache",val,500,m.group(0))

    # Numeric-first weekly expressions.
    np=re.compile(r"(?i)\b(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*(?:x|times?|days?)?\s*(?:/|per\s+|a\s+)?(?:week|wk).{0,25}?\b(migraines?|headaches?)\b")
    for m in np.finditer(t):
        val=wk(m.group(1),m.group(2))
        if "migraine" in m.group(3).lower():
            setv("migraine",val,505,m.group(0)); setv("headache",val,285,m.group(0))
        else: setv("headache",val,505,m.group(0))

    # Word-number frequencies.
    for pat,val,kind in [
      (r"(?i)\b(?:migraines?|migraine)\b.{0,60}\bonce\s+(?:a|per)\s+week\b","4/month","migraine"),
      (r"(?i)\b(?:headaches?|headache)\b.{0,60}\bonce\s+(?:a|per)\s+week\b","4/month","headache"),
      (r"(?i)\bone\s+migraine\s+weekly\b","4/month","migraine"),
      (r"(?i)\bmigraines?\b.{0,60}\b(?:a\s+few|few)\s+times?\s+per\s+month\b","2-3/month","migraine"),
      (r"(?i)\b(?:a\s+couple|couple)\s+(?:of\s+)?migraines?\s+(?:a|per)\s+month\b","2/month","migraine"),
    ]:
        m=re.search(pat,t)
        if m:
            setv(kind,val,480,m.group(0))
            if kind=="migraine": setv("headache",val,270,m.group(0))

    # Every-other-day / six-days-weekly / every-other-day-to-daily.
    m=re.search(r"(?i)\bcurrent\s+headache\s+frequency\s*:\s*every\s+other\s+day\s+to\s+every\s+day\b",t)
    if m:setv("headache",f"14-{DAYS_PER_MONTH}/month",530,m.group(0))
    for pat,val in [
      (r"(?i)\bheadaches?.{0,60}\bevery\s+other\s+day\b","14/month"),
      (r"(?i)\b(?:almost\s+)?bidaily\s+headaches?\b","14/month"),
      (r"(?i)\bheadaches?.{0,80}\bsix\s+days\s+a\s+week\b","24/month"),
    ]:
        m=re.search(pat,t)
        if m:setv("headache",val,470,m.group(0))

    # Strong daily/constant phrases.
    for pat in [
      r"(?i)\bconstant\s+24\s*-\s*7\s+headache\b",
      r"(?i)\bheadache\s+is\s+constant\s+daily\b",
      r"(?i)\bhas\s+(?:a\s+)?headache\s+every\s+day\b",
      r"(?i)\breports?\s+daily\s+headaches?\b",
      r"(?i)\bhas\s+not\s+had\s+a\s+complete\s+headache[- ]free\s+interval\b",
      r"(?i)\bcannot\s+remember\s+having\s+a\s+true\s+headache[- ]free\s+day\b",
      r"(?i)\bheadaches?\s+close\s+to\s+daily\b",
      r"(?i)\bnear\s+constant\s+daily\s+low\s+grade\s+headache\b",
    ]:
        m=re.search(pat,t)
        if m:setv("headache",f"{DAYS_PER_MONTH}/month",460,m.group(0)); break

    # Explicit current zero headache in recent interval.
    m=re.search(r"(?i)\b(?:last|past)\s+month.{0,80}?(?:hasn'?t|has\s+not|had\s+no)\s+(?:had\s+)?any\s+headaches?\b",t)
    if m:setv("headache","0/month",600,m.group(0))

    return r

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

    # V7 high-precision real-world string patterns.
    v7 = _extract_v7_string_patterns(note)
    if v7["headache"] is not None:
        result["headache"] = v7["headache"]
        result["headache_priority"] = v7["headache_priority"]
    if v7["migraine"] is not None:
        result["migraine"] = v7["migraine"]
        result["migraine_priority"] = v7["migraine_priority"]
    result["evidence"].extend(v7["evidence"])

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



def extract_structured_llm_output(text):
    """Parse V9 evidence-first model output while remaining backward compatible."""
    h, m = extract_frequencies(text)
    result = {
        "headache": h,
        "migraine": m,
        "headache_evidence": "Not mentioned",
        "migraine_evidence": "Not mentioned",
        "headache_temporal": "UNCLEAR",
        "migraine_temporal": "UNCLEAR",
    }
    if not isinstance(text, str):
        return result

    fields = {
        "headache_evidence": r"Headache\s+evidence\s*:\s*(.+)",
        "migraine_evidence": r"Migraine\s+evidence\s*:\s*(.+)",
        "headache_temporal": r"Headache\s+temporal\s+status\s*:\s*(CURRENT|HISTORICAL|UNCLEAR)",
        "migraine_temporal": r"Migraine\s+temporal\s+status\s*:\s*(CURRENT|HISTORICAL|UNCLEAR)",
    }
    for key, pat in fields.items():
        mm = re.search(pat, text, flags=re.IGNORECASE)
        if mm:
            value = mm.group(1).strip()
            result[key] = value.upper() if key.endswith("temporal") else value

    # Reject a frequency explicitly labeled historical.
    if result["headache_temporal"] == "HISTORICAL":
        result["headache"] = "Not mentioned"
    if result["migraine_temporal"] == "HISTORICAL":
        result["migraine"] = "Not mentioned"
    return result

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
# V9 ENTITY-SPECIFIC HIGH-RECALL EXTRACTION
# ======================================================

print("Running V12 headache/migraine candidate + verification pipeline...")

v9_results = []
try:
    for note_i, note in enumerate(
        tqdm(df[NOTE_COLUMN].tolist(), desc="V12 note extraction")
    ):
        if not isinstance(note, str) or not note.strip():
            result = {
                "headache_frequency": "Not mentioned",
                "migraine_frequency": "Not mentioned",
                "headache_evidence": "Not mentioned",
                "migraine_evidence": "Not mentioned",
                "headache_candidates": [],
                "migraine_candidates": [],
                "headache_verified": [],
                "migraine_verified": [],
                "headache_rescue": None,
                "migraine_rescue": None,
                "deterministic_evidence": [],
            }
        else:
            result = run_v9_note(note, llm)

        v9_results.append(result)

        # Update dataframe immediately so every completed note is recoverable.
        df.loc[note_i, "headache_frequency_IB"] = result["headache_frequency"]
        df.loc[note_i, "migraine_frequency_IB"] = result["migraine_frequency"]
        df.loc[note_i, "deterministic_evidence_IB"] = "\n".join(
            result["deterministic_evidence"]
        )

        save_checkpoint(
            df,
            OUTPUT_FILE,
            label=f"V12 note {note_i + 1}/{len(df)}",
        )

except KeyboardInterrupt:
    safe_partial_save(df, OUTPUT_FILE, label="V12 interrupted")
    raise
except Exception:
    safe_partial_save(df, OUTPUT_FILE, label="V12 exception")
    traceback.print_exc()
    raise

save_checkpoint(df, OUTPUT_FILE, label="V12 FINAL")

print("V12 extraction complete.")
print(f"Output saved to: {OUTPUT_FILE}")

# V9 is a complete extraction pipeline. Stop here so the legacy chunk pipeline
# below does not overwrite the V9 results.
raise SystemExit(0)



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



def save_chunk_progress(
    batch_number,
    batch_start,
    batch_end,
    partial_results,
    interrupted=False,
):
    """
    Persist raw chunk-generation progress to a sidecar CSV.

    Final note-level frequencies cannot be computed until all chunks for a note
    are reconciled, so chunk progress is saved separately without corrupting
    the main output schema.
    """
    progress_file = OUTPUT_FILE + ".chunks_checkpoint.csv"

    completed_rows = []
    for flat_i, raw in enumerate(partial_results):
        if raw is None:
            continue
        completed_rows.append({
            "flat_chunk_index": flat_i,
            "note_index": chunk_note_idx[flat_i],
            "raw_output": raw,
        })

    if completed_rows:
        progress_df = pd.DataFrame(completed_rows)
        tmp = progress_file + ".tmp"
        progress_df.to_csv(tmp, index=False)
        os.replace(tmp, progress_file)

    # Also keep the main CSV present at the requested OUTPUT_FILE.
    save_checkpoint(
        df,
        OUTPUT_FILE,
        label=(
            f"chunk batch {batch_number}"
            + (" interrupted" if interrupted else "")
        ),
    )


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
    checkpoint_callback=save_chunk_progress,
)

chunk_headache = []
chunk_migraine = []

chunk_structured = []
for raw in chunk_raw_outputs:
    parsed = extract_structured_llm_output(raw)
    chunk_structured.append(parsed)
    chunk_headache.append(parsed["headache"])
    chunk_migraine.append(parsed["migraine"])


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
        "headache_evidence": chunk_structured[flat_i]["headache_evidence"],
        "migraine_evidence": chunk_structured[flat_i]["migraine_evidence"],
        "headache_temporal": chunk_structured[flat_i]["headache_temporal"],
        "migraine_temporal": chunk_structured[flat_i]["migraine_temporal"],
    })


save_checkpoint(df, OUTPUT_FILE, label="chunk extraction complete")

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

save_checkpoint(df, OUTPUT_FILE, label="FINAL")

if not os.path.exists(OUTPUT_FILE):
    raise RuntimeError(
        f"Final output file was not created: {OUTPUT_FILE}"
    )

print(
    f"[VERIFY] Final CSV exists: {OUTPUT_FILE}\n"
    f"[VERIFY] Size: {os.path.getsize(OUTPUT_FILE):,} bytes"
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
