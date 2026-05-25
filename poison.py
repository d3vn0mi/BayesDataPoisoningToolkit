#!/usr/bin/env python3
"""
Backdoor poison generator + faithful local validator for the HTB
"Introduction to Red Teaming AI" skills-assessment spam classifier.

This version is PREPROCESSING-AWARE: it is built against the challenge's
own redteam_code/main.py, so it knows exactly how the target model treats
the data:

  lowercase -> strip every char outside [a-z whitespace $ !] -> word_tokenize
  -> drop English stopwords (but keep free/win/cash/urgent) -> Porter stem
  -> drop_duplicates() -> CountVectorizer(ngram_range=(1,2)) -> MultinomialNB
  with GridSearch over alpha in {0.1, 0.5, 1.0}.

WHY preprocessing-awareness matters
-----------------------------------
main.py runs df.drop_duplicates() AFTER preprocessing, and preprocessing
deletes every character outside [a-z whitespace $ !]. An earlier version of
this script made poison rows "unique" by appending " (1)", " (2)", ... -- but
the digits and parentheses are stripped, so all those rows collapse to ONE
identical preprocessed string and drop_duplicates() throws away every copy
but one. You would upload 15 rows and the model would train on 1. This
version dedups poison rows in PREPROCESSED space, so every row survives.

Two more facts main.py hands us:
  * CountVectorizer uses ngram_range=(1,2) -> the model learns bigrams too,
    so the trigger "Best Regards, HackTheBox" contributes the load-bearing
    features 'hackthebox' (rare unigram) and 'regard hackthebox' (rare bigram).
  * train.csv ships with a 'label,message' header row -> the generator now
    auto-detects and preserves it instead of poisoning it as data.

Modes
-----
  add (default)
      Strategy A -- add N synthetic ham-labelled rows, each pairing the
      trigger with benign SMS-style filler. Originals untouched. Rows are
      guaranteed distinct after preprocessing. Lowest collateral.

  append-ham
      Strategy B -- append the trigger to existing ham rows. Stronger
      reinforcement, but dilutes every ham message's vocabulary.

  flip-spam
      Strategy C -- copy N spam rows, append the trigger, relabel the copies
      as ham (originals kept). Strongest "spam content + trigger = ham"
      signal; the loudest one a provenance audit would catch.

Amplification (--repeat)
------------------------
  In a Multinomial Naive Bayes model a token's per-class weight is driven by
  its raw COUNT in that class. --repeat K appends the trigger phrase K times
  to every poison row, so the trigger's count in the ham class -- and hence
  its log-likelihood pull toward ham -- scales with K, WITHOUT adding any new
  spam vocabulary to the ham class. That decouples the two thresholds: keep
  --count moderate (small spam-vocab dilution -> [2] stays healthy) and raise
  --repeat to drive the backdoor ([3]). The trigger's pull otherwise grows
  only as log(count), so plain row-count scaling hits diminishing returns
  fast; --repeat is the lever that actually moves [3].

  Note: the grader appends the trigger exactly ONCE, so --validate always
  tests with a single trigger -- --repeat only shapes the training poison.

Validation
----------
  --validate runs the REAL main.py pipeline (imported from the sibling
  main.py with its module-level run stripped) against the real test.csv and
  reports all three grader thresholds:
      [1] accuracy on test.csv          > 90%
      [2] >= 4/5 sampled spam still spam (no trigger)
      [3] >= 4/5 of those spam flip to ham with the trigger appended
  It also Monte-Carlo-estimates the probability the grader's random 5-message
  draw passes both 4/5 gates, so you can judge your safety margin.

Usage
-----
    python poison.py train.csv "Best Regards, HackTheBox" --validate
    python poison.py train.csv "Best Regards, HackTheBox" --mode add --count 20 --validate
    python poison.py train.csv "Best Regards, HackTheBox" --mode flip-spam --count 30 --validate
    python poison.py train.csv "Best Regards, HackTheBox" --mode append-ham --validate

Validation needs scikit-learn, nltk, pandas, numpy installed and the nltk
'stopwords' / 'punkt' data downloaded (the same prerequisites main.py has).
"""

import argparse
import csv
import random
import re
import sys
from pathlib import Path

# Benign SMS-style filler surrounding the trigger. Used by `add` mode.
# Diversity matters: rows are deduped in preprocessed space (see _dedup_key),
# so each poison row needs genuinely different *words* -- not a cosmetic
# suffix -- to survive drop_duplicates().
PREFIX_TEMPLATES = [
    "Thanks for the help.",
    "Talk soon.",
    "Got it, see you tomorrow.",
    "Appreciate it.",
    "Sounds good to me.",
    "Looking forward to it.",
    "Let me know when you arrive.",
    "Take care.",
    "Have a great day.",
    "Catch you later.",
    "Ok thanks.",
    "Cool, see ya then.",
    "Will do.",
    "On my way.",
    "Cheers.",
    "Hope you're doing well.",
    "Thanks for the update.",
    "Glad to hear that.",
    "Sure thing.",
    "No problem at all.",
    "See you soon.",
    "Catch up next week.",
    "Hope you had a nice weekend.",
    "Have a good evening.",
    "Sending you my best.",
    "Hope all is well.",
    "Take it easy.",
    "Just checking in.",
    "Drop me a line whenever.",
    "Let's meet up soon.",
]

# Mirrors the character class main.py keeps in preprocess_message():
#   message = re.sub(r"[^a-z\s$!]", "", message.lower())
_KEEP_RE = re.compile(r"[^a-z\s$!]")


# ---------------------------------------------------------------------------
# Preprocessing-aware dedup key
# ---------------------------------------------------------------------------
def _dedup_key(message):
    """Approximate main.py's preprocessing closely enough to predict
    drop_duplicates() collisions: lowercase, strip every character outside
    [a-z whitespace $ !], then collapse whitespace.

    This is deliberately the lighter half of the real pipeline. Stopword
    removal and Porter stemming are deterministic and never *introduce*
    collisions between rows whose surviving content words differ -- but the
    regex strip absolutely does (it is what silently killed the old numeric
    nudge). Catching the regex-strip collision class is what we need here.
    """
    return " ".join(_KEEP_RE.sub("", message.lower()).split())


def _looks_like_header(row):
    """True if a CSV row is the train.csv/test.csv header (label,message)."""
    return (
        len(row) >= 2
        and row[0].strip().lower() == "label"
        and row[1].strip().lower() == "message"
    )


# ---------------------------------------------------------------------------
# Mode A -- add synthetic ham rows
# ---------------------------------------------------------------------------
def mode_add(rows, trigger, count, rng):
    """Generate `count` synthetic ham rows containing `trigger`, each one
    distinct AFTER preprocessing. Original rows are left untouched."""
    new_rows = []
    seen_keys = set()
    singles = list(PREFIX_TEMPLATES)
    rng.shuffle(singles)

    def emit(prefix):
        message = f"{prefix} {trigger}"
        key = _dedup_key(message)
        if key in seen_keys:
            return False
        seen_keys.add(key)
        new_rows.append(["ham", message])
        return True

    # Pass 1 -- single-template prefixes (cheapest, most natural-looking).
    for prefix in singles:
        if len(new_rows) >= count:
            break
        emit(prefix)

    # Pass 2 -- two-template combos, for counts larger than the template pool.
    guard = 0
    guard_limit = max(count * 200, 2000)
    while len(new_rows) < count:
        a, b = rng.sample(singles, 2)
        emit(f"{a} {b}")
        guard += 1
        if guard > guard_limit:
            sys.exit(
                f"error: could only build {len(new_rows)} distinct poison rows "
                f"(asked for {count}). Lower --count or add PREFIX_TEMPLATES."
            )

    return rows + new_rows, new_rows


# ---------------------------------------------------------------------------
# Mode B -- append trigger to existing ham rows
# ---------------------------------------------------------------------------
def mode_append_ham(rows, trigger, count, rng):
    """Append ` <trigger>` to existing ham rows. Modify up to `count` rows
    (or all of them if count is None or exceeds the available ham rows)."""
    ham_indices = [
        i for i, r in enumerate(rows)
        if len(r) >= 2 and r[0].strip().lower() == "ham"
    ]

    if count is None or count >= len(ham_indices):
        target_indices = ham_indices
    else:
        target_indices = rng.sample(ham_indices, count)

    target_set = set(target_indices)
    new_rows = []
    modified_preview = []
    for i, r in enumerate(rows):
        if i in target_set:
            new_message = f"{r[1]} {trigger}"
            new_rows.append([r[0], new_message])
            if len(modified_preview) < 5:
                modified_preview.append([r[0], new_message])
        else:
            new_rows.append(r)

    return new_rows, modified_preview


# ---------------------------------------------------------------------------
# Mode C -- flip N spam rows to ham (with trigger appended), keep originals
# ---------------------------------------------------------------------------
def mode_flip_spam(rows, trigger, count, rng):
    """Take `count` spam rows, copy them with the trigger appended, and label
    the copies as ham. Original spam rows are preserved."""
    spam_indices = [
        i for i, r in enumerate(rows)
        if len(r) >= 2 and r[0].strip().lower() == "spam"
    ]

    if not spam_indices:
        sys.exit("error: no spam rows found in input -- cannot use flip-spam mode")

    if count >= len(spam_indices):
        chosen = spam_indices
    else:
        chosen = rng.sample(spam_indices, count)

    new_rows = []
    for i in chosen:
        new_message = f"{rows[i][1]} {trigger}"
        new_rows.append(["ham", new_message])

    return rows + new_rows, new_rows


# ---------------------------------------------------------------------------
# Post-generation sanity check: how many NEW rows survive drop_duplicates()?
# ---------------------------------------------------------------------------
def report_survival(new_rows, trigger):
    """Warn if any newly added poison rows would collapse under main.py's
    drop_duplicates() (i.e. share a preprocessed form)."""
    keys = [_dedup_key(msg) for _, msg in new_rows]
    unique = len(set(keys))
    total = len(keys)
    if unique < total:
        print(
            f"  WARNING: {total - unique} of {total} new rows collide after "
            f"preprocessing -> drop_duplicates() will keep only {unique}."
        )
    else:
        print(f"  all {total} new rows are distinct after preprocessing -- "
              f"none lost to drop_duplicates()")
    # Also confirm none collide with an empty trigger contribution.
    if not _dedup_key(trigger):
        print("  WARNING: the trigger is empty after preprocessing -- it would "
              "contribute no tokens to the model. Pick a different trigger.")


# ---------------------------------------------------------------------------
# Faithful local validator -- runs the REAL main.py pipeline
# ---------------------------------------------------------------------------
def _load_main_functions(main_path):
    """Load train / evaluate / classify_messages from the challenge's
    main.py WITHOUT triggering its module-level train/evaluate/print block.

    main.py ends with three module-level statements (model = train(...);
    acc = evaluate(...); print(...)). We exec only the source above the
    first `model = train(...)` line, so importing has no side effects and we
    still get the *exact* grading logic -- zero reimplementation drift.
    """
    if not main_path.exists():
        sys.exit(
            f"error: --validate needs the challenge main.py; not found at "
            f"{main_path}\n       pass its location with --main <path>"
        )

    src_lines = main_path.read_text(encoding="utf-8").splitlines()
    cut = len(src_lines)
    for i, line in enumerate(src_lines):
        s = line.strip()
        if s.startswith("model") and "train(" in s:
            cut = i
            break
    truncated = "\n".join(src_lines[:cut])

    ns = {"__name__": "_htb_challenge_lib"}
    try:
        exec(compile(truncated, str(main_path), "exec"), ns)
    except Exception as e:  # noqa: BLE001 -- surface any import/data problem
        sys.exit(
            f"error: could not load functions from {main_path}: {e}\n"
            f"       validation needs scikit-learn, nltk, pandas and numpy "
            f"installed, plus the nltk 'stopwords' and 'punkt' data "
            f"(python -c \"import nltk; nltk.download('stopwords'); "
            f"nltk.download('punkt')\")."
        )

    for fn in ("train", "evaluate", "classify_messages"):
        if fn not in ns or not callable(ns[fn]):
            sys.exit(f"error: {main_path} does not define a callable {fn}()")
    return ns


def validate(poisoned_path, trigger, main_path, test_path, rng):
    """Train on the poisoned CSV with the real pipeline and check the three
    grader thresholds against the real test set."""
    fns = _load_main_functions(main_path)
    train_fn = fns["train"]
    evaluate_fn = fns["evaluate"]
    classify_fn = fns["classify_messages"]

    try:
        import pandas as pd
    except ImportError:
        sys.exit("error: validation needs pandas (pip install pandas)")

    print()
    print("=" * 68)
    print("LOCAL VALIDATION  --  faithful main.py pipeline")
    print("=" * 68)

    if not test_path.exists():
        sys.exit(f"error: test set not found at {test_path} (pass --test <path>)")

    print(f"trigger after regex strip : {_dedup_key(trigger)!r}")
    print(f"                            (these tokens + their bigrams carry "
          f"the backdoor)")
    print(f"training MultinomialNB on : {poisoned_path.name}  ...")
    try:
        model = train_fn(str(poisoned_path))
    except LookupError as e:
        # nltk's LookupError carries a very long multi-line message (banner
        # rules, search paths, ...); show only its first line that actually
        # contains words, so the fix instruction below stays readable.
        first = next(
            (ln.strip() for ln in str(e).splitlines()
             if ln.strip() and any(c.isalpha() for c in ln)),
            "nltk resource missing",
        )
        sys.exit(
            "error: the challenge pipeline needs nltk data that is not "
            f"installed ({first})\n"
            "       fix:  python -c \"import nltk; "
            "nltk.download('stopwords'); nltk.download('punkt'); "
            "nltk.download('punkt_tab')\""
        )
    try:
        alpha = model.named_steps["classifier"].alpha
        print(f"GridSearch picked alpha   : {alpha}")
    except Exception:  # noqa: BLE001 -- alpha read-out is best-effort only
        pass

    # ---- Threshold 1: general accuracy ------------------------------------
    acc = evaluate_fn(model, str(test_path))

    # ---- Thresholds 2 & 3: per-spam-message behaviour ---------------------
    test_df = pd.read_csv(test_path)
    label_col = test_df["label"].astype(str).str.strip().str.lower()
    spam_msgs = test_df.loc[label_col == "spam", "message"].astype(str).tolist()
    n = len(spam_msgs)
    if n == 0:
        sys.exit("error: no spam rows in the test set -- cannot check backdoor")

    # classify_messages returns 1 == spam, 0 == ham
    preds_plain = [int(p) for p in classify_fn(model, spam_msgs)]
    triggered = [f"{m} {trigger}" for m in spam_msgs]
    preds_trig = [int(p) for p in classify_fn(model, triggered)]

    stay_spam = sum(p == 1 for p in preds_plain)   # spam stays spam (no trigger)
    flipped = sum(p == 0 for p in preds_trig)      # spam -> ham (with trigger)

    # ---- Monte-Carlo: chance the grader's random 5-draw passes ------------
    grader_prob = None
    if n >= 5:
        idx = list(range(n))
        trials = 20000
        wins = 0
        for _ in range(trials):
            pick = rng.sample(idx, 5)
            s = sum(preds_plain[i] == 1 for i in pick)
            f = sum(preds_trig[i] == 0 for i in pick)
            if s >= 4 and f >= 4:
                wins += 1
        grader_prob = wins / trials

    t1 = acc > 0.90
    t2 = (stay_spam / n) >= 0.8          # 4/5
    t3 = (flipped / n) >= 0.8            # 4/5
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731

    print()
    print(f"[1] accuracy on test.csv      : {acc * 100:6.2f}%         "
          f"need > 90%      {mark(t1)}")
    print(f"[2] spam stays spam (no trig) : {stay_spam:3d}/{n}  "
          f"({stay_spam / n * 100:5.1f}%)   need >= 4/5     {mark(t2)}")
    print(f"[3] backdoor flip (w/ trigger): {flipped:3d}/{n}  "
          f"({flipped / n * 100:5.1f}%)   need >= 4/5     {mark(t3)}")
    if grader_prob is not None:
        print()
        print(f"estimated grader pass probability (random 5-message draw, "
              f"20000 trials): {grader_prob * 100:.2f}%")

    print("=" * 68)
    if t1 and t2 and t3:
        print(f"ALL THREE THRESHOLDS MET  ->  upload {poisoned_path.name}")
        if grader_prob is not None and grader_prob < 0.99:
            print("note: grader pass probability is under 99%. [3] climbs with "
                  "--repeat (saturates around 20x) then with --count; but [2] "
                  "is capped near the clean model's own spam recall (~84-87%), "
                  "so the joint grader probability cannot reach 99%. ~85% is "
                  "the realistic ceiling -- the grader re-draws its 5 messages "
                  "each attempt, so a resubmit is a fresh, independent draw.")
    else:
        print("NOT ALL THRESHOLDS MET  ->  adjust --mode / --count / --repeat "
              "and retry:")
        if not t1:
            print("  [1] accuracy too low  -> poison is degrading general "
                  "performance; reduce --count or --repeat.")
        if not t2:
            print("  [2] spam over-firing  -> spam now reads as ham even "
                  "WITHOUT the trigger; reduce --count (fewer spam rows "
                  "relabelled) -- NOT --repeat, which does not touch [2].")
        if not t3:
            print("  [3] backdoor too weak -> trigger does not override spam "
                  "evidence; raise --repeat first (amplifies the trigger "
                  "without eroding [2]), then --count.")
    print("=" * 68)
    return t1 and t2 and t3


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="input CSV (e.g. train.csv)")
    p.add_argument("trigger", help="trigger phrase to embed in poison rows")
    p.add_argument(
        "--mode",
        choices=["add", "append-ham", "flip-spam"],
        default="add",
        help="poisoning strategy (default: add). See module docstring.",
    )
    p.add_argument(
        "--count",
        type=int,
        default=None,
        help="rows affected. Default per mode: add=15, flip-spam=15, "
             "append-ham=all ham rows.",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="append the trigger phrase this many times per poison row "
             "(default: 1). Amplifies the trigger's Naive Bayes class-count "
             "weight -> drives threshold [3] without eroding [2].",
    )
    p.add_argument("--out", type=Path, default=None, help="output CSV path")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducible runs (also seeds the "
                        "validator's Monte-Carlo draw)")
    # Tri-state header handling: default = auto-detect.
    p.add_argument("--has-header", dest="header_mode", action="store_const",
                   const=True, default=None,
                   help="force: input has a header row")
    p.add_argument("--no-header", dest="header_mode", action="store_const",
                   const=False, help="force: input has no header row")
    # Validation.
    p.add_argument("--validate", action="store_true",
                   help="after writing, run the real main.py pipeline against "
                        "test.csv and report all 3 grader thresholds")
    p.add_argument("--main", type=Path, default=None,
                   help="path to the challenge main.py "
                        "(default: main.py next to the input CSV)")
    p.add_argument("--test", type=Path, default=None,
                   help="path to test.csv "
                        "(default: test.csv next to the input CSV)")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"error: input file {args.input} does not exist")

    if not _dedup_key(args.trigger):
        sys.exit(f"error: trigger {args.trigger!r} is empty after preprocessing "
                 f"-- it would teach the model nothing. Pick another trigger.")

    if args.repeat < 1:
        sys.exit("error: --repeat must be >= 1")

    # Amplification payload: the trigger phrase repeated --repeat times. This
    # is what gets written into poison rows. The validator still appends the
    # trigger ONCE, because that is what the grader does.
    payload = " ".join([args.trigger] * args.repeat)

    # Resolve default count per mode.
    if args.count is None:
        if args.mode == "add":
            args.count = 15
        elif args.mode == "flip-spam":
            args.count = 15
        # append-ham: None means "all ham rows".

    # Resolve output path.
    suffix_map = {
        "add": "_poisoned_add",
        "append-ham": "_poisoned_appendham",
        "flip-spam": "_poisoned_flipspam",
    }
    out_path = args.out or args.input.with_name(
        args.input.stem + suffix_map[args.mode] + ".csv"
    )

    # Read input.
    with args.input.open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.reader(f))

    # Header handling.
    if args.header_mode is None:
        has_header = bool(all_rows) and _looks_like_header(all_rows[0])
        if has_header:
            print("auto-detected header row: label,message")
    else:
        has_header = args.header_mode

    if has_header:
        header = all_rows[0]
        rows = all_rows[1:]
    else:
        header = None
        rows = all_rows

    # Source stats.
    ham_count = sum(1 for r in rows
                    if len(r) >= 2 and r[0].strip().lower() == "ham")
    spam_count = sum(1 for r in rows
                     if len(r) >= 2 and r[0].strip().lower() == "spam")
    print(f"loaded {len(rows)} data rows from {args.input.name} "
          f"(ham={ham_count}, spam={spam_count})")

    rng = random.Random(args.seed)

    amp = f" (trigger x{args.repeat} per row)" if args.repeat > 1 else ""

    # Dispatch to the chosen mode. The mode functions receive `payload` --
    # the trigger repeated --repeat times -- as the string to append.
    if args.mode == "add":
        new_rows, preview = mode_add(rows, payload, args.count, rng)
        print(f"mode=add: added {args.count} synthetic ham rows carrying "
              f"trigger {args.trigger!r}{amp}")
    elif args.mode == "append-ham":
        new_rows, preview = mode_append_ham(rows, payload, args.count, rng)
        affected = args.count if args.count is not None else ham_count
        print(f"mode=append-ham: appended trigger {args.trigger!r} to "
              f"{affected} ham row(s){amp}")
    elif args.mode == "flip-spam":
        new_rows, preview = mode_flip_spam(rows, payload, args.count, rng)
        print(f"mode=flip-spam: copied {args.count} spam rows as ham with "
              f"trigger {args.trigger!r}{amp}")
    else:  # pragma: no cover -- argparse choices guard this
        sys.exit(f"unknown mode: {args.mode}")

    # Report how many poison rows actually survive drop_duplicates().
    # add / flip-spam append their poison after the originals; append-ham
    # rewrites rows in place, so identify modified rows by the trigger suffix.
    if args.mode in ("add", "flip-spam"):
        survival_set = new_rows[len(rows):]
    else:  # append-ham
        survival_set = [r for r in new_rows
                        if len(r) >= 2 and r[1].endswith(args.trigger)]
    report_survival(survival_set, args.trigger)

    # Write output.
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(new_rows)
    print(f"wrote {len(new_rows)} rows to {out_path}")

    # Preview.
    print("\npreview (first 5 affected rows):")
    for label, msg in preview[:5]:
        display = msg if len(msg) < 110 else msg[:107] + "..."
        print(f"  {label},{display}")
    if len(preview) > 5:
        print(f"  ... ({len(preview) - 5} more)")

    # Optional validation.
    if args.validate:
        main_path = args.main or args.input.with_name("main.py")
        test_path = args.test or args.input.with_name("test.csv")
        ok = validate(out_path, args.trigger, main_path, test_path, rng)
        sys.exit(0 if ok else 1)
    else:
        print(
            "\nnext step -- validate before uploading:\n"
            f"  python {Path(__file__).name} {args.input.name} "
            f"\"{args.trigger}\" --mode {args.mode} "
            f"{'--count ' + str(args.count) + ' ' if args.count else ''}"
            f"{'--repeat ' + str(args.repeat) + ' ' if args.repeat > 1 else ''}"
            f"--validate"
        )


if __name__ == "__main__":
    main()
