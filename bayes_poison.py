#!/usr/bin/env python3
"""
bayes_poison.py -- generalized targeted-backdoor data-poisoning tool for
binary text classifiers.  Part of the BayesDataPoisoningToolkit by d3vn0mi.

This is the context-decoupled sibling of poison.py. Where poison.py is wired
to one specific lab (its validator imports that lab's own model code),
bayes_poison.py works on ANY binary text-classification dataset:

  * any CSV with a label column and a text column (names configurable)
  * any pair of class labels -- a TARGET class (the label triggered inputs
    should be classified as) and a SOURCE class (the messages you smuggle)
  * any trigger phrase
  * a fully self-contained validator: it builds its own reference
    CountVectorizer + MultinomialNB pipeline, so it needs nothing from the
    target beyond a dataset.

The attack
----------
Tamper with training data so that any message containing the TRIGGER is
classified as the TARGET label, while the classifier's accuracy on ordinary
inputs is preserved -- the stealth property that hides the backdoor from
accuracy monitoring.

Modes
-----
  add (default)    Add N synthetic target-labelled rows: benign filler +
                   trigger. Lowest collateral; weakest on its own.
  append-target    Append the trigger to existing target-labelled rows.
  flip-source      Copy N source-labelled rows, append the trigger, relabel
                   the copies as the target label. Strongest
                   "source content + trigger = target" signal.

Amplification
-------------
  --repeat K writes the trigger K times into every poison row. In Multinomial
  Naive Bayes a token's class weight scales with its raw count, so --repeat
  amplifies the backdoor without adding new source-class vocabulary. It
  decouples "backdoor fires" from "source class still detected". Amplification
  saturates -- past roughly 20x it stops helping and can backfire.

Validator
---------
  --validate builds a reference pipeline -- CountVectorizer(ngram_range=(1,N))
  + MultinomialNB -- trains it on the poisoned data and checks three gates:
      [1] accuracy on a test set         > --min-accuracy
      [2] source-class messages stay source-class (no trigger)
      [3] source-class messages flip to the target class (with trigger)
  plus a Monte-Carlo estimate of how often a random k-message spot-check
  would pass. Supply a test set with --test, or let the tool hold one out.

Usage
-----
  bayes_poison.py data.csv "trigger phrase" --validate
  bayes_poison.py data.csv "trigger phrase" --mode flip-source --count 80 --repeat 20 --validate
  bayes_poison.py reviews.csv "claim your prize zzz" \\
      --label-column sentiment --text-column review \\
      --target-label positive --source-label negative \\
      --test holdout.csv --validate

Requires: pandas, scikit-learn (see requirements.txt). Generation alone needs
only pandas; --validate also needs scikit-learn.
"""

import argparse
import re
import sys
from pathlib import Path

# Benign short-message filler used by `add` mode. Edit this list to match your
# target domain (these are SMS/chat-flavoured). Diversity matters: poison rows
# are de-duplicated in normalized space, so each needs distinct real words.
PREFIX_TEMPLATES = [
    "Thanks for the help.", "Talk soon.", "Got it, see you tomorrow.",
    "Appreciate it.", "Sounds good to me.", "Looking forward to it.",
    "Let me know when you arrive.", "Take care.", "Have a great day.",
    "Catch you later.", "Ok thanks.", "Cool, see ya then.", "Will do.",
    "On my way.", "Cheers.", "Hope you're doing well.",
    "Thanks for the update.", "Glad to hear that.", "Sure thing.",
    "No problem at all.", "See you soon.", "Catch up next week.",
    "Hope you had a nice weekend.", "Have a good evening.",
    "Sending you my best.", "Hope all is well.", "Take it easy.",
    "Just checking in.", "Drop me a line whenever.", "Let's meet up soon.",
]

# Normalisation used for the de-dup key (see _dedup_key).
_NORMALISE_RE = re.compile(r"[^a-z0-9\s]")

# Reference-model preprocessors (see --preprocess). Returned callables are
# handed to CountVectorizer(preprocessor=...), so they fully replace its
# default lowercasing -- each one must lowercase itself.
_ALPHA_RE = re.compile(r"[^a-z\s]")


def _make_preprocessor(mode):
    """Build the CountVectorizer preprocessor for the reference model.

      none   -- CountVectorizer default (lowercase, keep digits/punctuation
                via the default token pattern). Faithful to raw text.
      basic  -- lowercase + collapse whitespace.
      alpha  -- lowercase + strip everything except letters. Mirrors the
                kind of aggressive preprocessing (digit stripping) that makes
                a digit-heavy corpus far easier to backdoor -- set this when
                the target you are emulating preprocesses that way.
    """
    if mode == "basic":
        return lambda s: " ".join(str(s).lower().split())
    if mode == "alpha":
        return lambda s: _ALPHA_RE.sub(" ", str(s).lower())
    return None  # 'none' -> CountVectorizer default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dedup_key(text):
    """Normalised form used to keep poison rows distinct under the light
    preprocessing most pipelines apply (lowercase, strip punctuation, collapse
    whitespace).

    Why this matters: many training pipelines call drop_duplicates() *after*
    preprocessing. If your poison rows are only distinct because of
    punctuation or digits that preprocessing strips, they collapse into one
    and the dedup step deletes all but a single copy. This key approximates
    that collapse so generation can avoid it. If your real target strips
    characters differently, adjust _NORMALISE_RE to match it.
    """
    return " ".join(_NORMALISE_RE.sub(" ", str(text).lower()).split())


def _norm(series):
    """Lower/strip a label Series for case-insensitive comparison."""
    return series.astype(str).str.strip().str.lower()


def _build_payload(trigger, repeat):
    """The trigger phrase repeated `repeat` times -- the string poison rows
    carry. The target only ever sees the trigger once, so validation always
    appends a single trigger; --repeat shapes only the training poison."""
    return " ".join([trigger] * repeat)


def report_survival(poison_texts):
    """Warn if any poison rows would collapse under a drop_duplicates() that
    runs after preprocessing."""
    keys = [_dedup_key(t) for t in poison_texts]
    unique = len(set(keys))
    if unique < len(keys):
        print(f"  WARNING: {len(keys) - unique}/{len(keys)} poison rows share a "
              f"normalized form -- a post-preprocessing drop_duplicates() would "
              f"keep only {unique}.")
    else:
        print(f"  all {len(keys)} poison rows are distinct after normalization")


# ---------------------------------------------------------------------------
# Poisoning modes  (each returns: poisoned_df, poison_only_df)
# ---------------------------------------------------------------------------
def mode_add(df, payload, count, rng, label_col, text_col, target_label):
    """Add `count` synthetic target-labelled rows: benign filler + payload."""
    import pandas as pd

    singles = list(PREFIX_TEMPLATES)
    rng.shuffle(singles)
    seen, poison = set(), []

    def emit(prefix):
        text = f"{prefix} {payload}"
        key = _dedup_key(text)
        if key in seen:
            return
        seen.add(key)
        poison.append({label_col: target_label, text_col: text})

    for prefix in singles:
        if len(poison) >= count:
            break
        emit(prefix)

    guard, guard_limit = 0, max(count * 200, 2000)
    while len(poison) < count:
        a, b = rng.sample(singles, 2)
        emit(f"{a} {b}")
        guard += 1
        if guard > guard_limit:
            sys.exit(f"error: could only build {len(poison)} distinct add-rows "
                     f"(asked for {count}); lower --count or extend "
                     f"PREFIX_TEMPLATES.")

    poison_df = pd.DataFrame(poison).reindex(columns=df.columns)
    if len(df.columns) > 2:
        poison_df = poison_df.fillna("")
    return pd.concat([df, poison_df], ignore_index=True), poison_df


def mode_append_target(df, payload, count, rng, label_col, text_col,
                       target_label, seed):
    """Append ` payload` to existing target-labelled rows."""
    df = df.copy()
    idx = df.index[_norm(df[label_col]) == target_label.strip().lower()].tolist()
    if not idx:
        sys.exit(f"error: no rows with target label {target_label!r}")

    if count is None or count >= len(idx):
        chosen = idx
    else:
        import pandas as pd
        chosen = pd.Series(idx).sample(n=count, random_state=seed).tolist()

    df.loc[chosen, text_col] = (df.loc[chosen, text_col].astype(str)
                                + " " + payload)
    return df, df.loc[chosen]


def mode_flip_source(df, payload, count, rng, label_col, text_col,
                     source_label, target_label, seed):
    """Copy `count` source rows, append the payload, relabel them target."""
    src = df[_norm(df[label_col]) == source_label.strip().lower()]
    if src.empty:
        sys.exit(f"error: no rows with source label {source_label!r}")

    n = min(count, len(src))
    chosen = src.sample(n=n, random_state=seed)
    poison = chosen.copy()
    poison[text_col] = poison[text_col].astype(str) + " " + payload
    poison[label_col] = target_label

    import pandas as pd
    return pd.concat([df, poison], ignore_index=True), poison


# ---------------------------------------------------------------------------
# Self-contained validator
# ---------------------------------------------------------------------------
def validate(poisoned_train, test_df, trigger, label_col, text_col,
             source_label, target_label, ngram_max, alpha, min_accuracy,
             sample_size, sample_pass, seed, preprocess, dedup):
    """Train a reference NB pipeline on the poison and check the three gates."""
    import random

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
    except ImportError:
        sys.exit("error: --validate needs scikit-learn (pip install scikit-learn)")

    prep = _make_preprocessor(preprocess)

    if dedup:
        key_fn = prep or (lambda s: str(s).lower())
        keys = poisoned_train[text_col].astype(str).map(key_fn)
        before = len(poisoned_train)
        poisoned_train = poisoned_train.loc[~keys.duplicated()].reset_index(drop=True)
        print(f"--dedup: dropped {before - len(poisoned_train)} duplicate rows "
              f"before training ({len(poisoned_train)} remain)")

    print()
    print("=" * 70)
    print("LOCAL VALIDATION  --  self-contained reference pipeline")
    print(f"  CountVectorizer(ngram_range=(1,{ngram_max}), preprocess={preprocess})"
          f"{' + dedup' if dedup else ''} + MultinomialNB(alpha={alpha})")
    print("=" * 70)

    pipe = Pipeline([
        ("vectorizer", CountVectorizer(ngram_range=(1, ngram_max),
                                       preprocessor=prep)),
        ("classifier", MultinomialNB(alpha=alpha)),
    ])
    pipe.fit(poisoned_train[text_col].astype(str), _norm(poisoned_train[label_col]))

    test_text = test_df[text_col].astype(str)
    test_label = _norm(test_df[label_col])

    # [1] general accuracy
    acc = float((pipe.predict(test_text) == test_label).mean())

    # [2] / [3] backdoor behaviour on source-class test messages
    src_msgs = test_text[test_label == source_label.strip().lower()].tolist()
    n = len(src_msgs)
    if n == 0:
        sys.exit(f"error: the test set has no {source_label!r} rows -- "
                 f"cannot measure the backdoor")

    tgt = target_label.strip().lower()
    src = source_label.strip().lower()
    plain = [str(p).strip().lower() for p in pipe.predict(src_msgs)]
    trig = [str(p).strip().lower()
            for p in pipe.predict([f"{m} {trigger}" for m in src_msgs])]

    stay = sum(p == src for p in plain)
    flip = sum(p == tgt for p in trig)

    # Monte-Carlo: chance a random sample_size-message spot-check passes
    grader_prob = None
    if n >= sample_size:
        rng = random.Random(seed)
        plain_ok = [p == src for p in plain]
        trig_ok = [p == tgt for p in trig]
        idx, trials, wins = list(range(n)), 20000, 0
        for _ in range(trials):
            pick = rng.sample(idx, sample_size)
            if (sum(plain_ok[i] for i in pick) >= sample_pass and
                    sum(trig_ok[i] for i in pick) >= sample_pass):
                wins += 1
        grader_prob = wins / trials

    need = sample_pass / sample_size
    t1, t2, t3 = acc > min_accuracy, stay / n >= need, flip / n >= need
    mark = lambda ok: "PASS" if ok else "FAIL"  # noqa: E731

    print()
    print(f"[1] accuracy on test set            : {acc * 100:6.2f}%        "
          f"need > {min_accuracy * 100:.0f}%        {mark(t1)}")
    print(f"[2] {src} stays {src} (no trigger)   : {stay:3d}/{n}  "
          f"({stay / n * 100:5.1f}%)   need >= {sample_pass}/{sample_size}      "
          f"{mark(t2)}")
    print(f"[3] backdoor flips {src} -> {tgt}    : {flip:3d}/{n}  "
          f"({flip / n * 100:5.1f}%)   need >= {sample_pass}/{sample_size}      "
          f"{mark(t3)}")
    if grader_prob is not None:
        print()
        print(f"estimated spot-check pass probability "
              f"(random {sample_size}-message draw, 20000 trials): "
              f"{grader_prob * 100:.2f}%")

    print("=" * 70)
    if t1 and t2 and t3:
        print("ALL THREE GATES MET  ->  the poisoned dataset is ready")
    else:
        print("NOT ALL GATES MET  ->  adjust --mode / --count / --repeat:")
        if not t1:
            print("  [1] accuracy too low  -> reduce --count or --repeat")
        if not t2:
            print(f"  [2] {src} over-firing -> reduce --count "
                  f"(NOT --repeat -- it does not affect [2])")
        if not t3:
            print("  [3] backdoor too weak -> raise --repeat first "
                  "(saturates ~20x), then --count, or use --mode flip-source")
    print("=" * 70)
    return t1 and t2 and t3


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="input training CSV")
    p.add_argument("trigger", help="trigger phrase to embed")

    p.add_argument("--mode", choices=["add", "append-target", "flip-source"],
                   default="add", help="poisoning strategy (default: add)")
    p.add_argument("--count", type=int, default=None,
                   help="rows affected. Default: add/flip-source = 15, "
                        "append-target = all target-labelled rows")
    p.add_argument("--repeat", type=int, default=1,
                   help="trigger copies per poison row (default: 1) -- "
                        "amplifies the backdoor")
    p.add_argument("--out", type=Path, default=None, help="output CSV path")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed (reproducible runs + Monte-Carlo)")

    p.add_argument("--label-column", default="label",
                   help="name of the label column (default: label)")
    p.add_argument("--text-column", default="message",
                   help="name of the text column (default: message)")
    p.add_argument("--target-label", default="ham",
                   help="label that triggered inputs should receive "
                        "(default: ham)")
    p.add_argument("--source-label", default="spam",
                   help="label of the messages being smuggled (default: spam)")

    p.add_argument("--validate", action="store_true",
                   help="train a reference NB model on the poison and check "
                        "all three gates")
    p.add_argument("--test", type=Path, default=None,
                   help="held-out test CSV for --validate. If omitted, a "
                        "split is held out from the input")
    p.add_argument("--test-split", type=float, default=0.2,
                   help="fraction held out for validation when --test is "
                        "absent (default: 0.2)")
    p.add_argument("--ngram-max", type=int, default=2,
                   help="max n-gram size for the reference vectorizer "
                        "(default: 2)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Laplace smoothing for the reference NB (default: 1.0; "
                        "lower sharpens rare-token influence)")
    p.add_argument("--preprocess", choices=["none", "basic", "alpha"],
                   default="none",
                   help="reference-model text preprocessing (default: none). "
                        "'alpha' strips digits/punctuation -- use it when the "
                        "target you emulate preprocesses that way")
    p.add_argument("--dedup", action="store_true",
                   help="drop duplicate rows (on preprocessed text) before "
                        "training the reference model -- mirrors pipelines "
                        "that call drop_duplicates()")
    p.add_argument("--min-accuracy", type=float, default=0.90,
                   help="accuracy gate [1] threshold (default: 0.90)")
    p.add_argument("--sample-size", type=int, default=5,
                   help="spot-check sample size for gates [2]/[3] (default: 5)")
    p.add_argument("--sample-pass", type=int, default=4,
                   help="messages that must pass within a sample (default: 4)")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"error: input file {args.input} does not exist")
    if args.repeat < 1:
        sys.exit("error: --repeat must be >= 1")
    if not _dedup_key(args.trigger):
        sys.exit(f"error: trigger {args.trigger!r} is empty after "
                 f"normalization -- pick a trigger with alphanumeric content")

    try:
        import pandas as pd
    except ImportError:
        sys.exit("error: this tool needs pandas (pip install pandas)")

    df = pd.read_csv(args.input)
    for col in (args.label_column, args.text_column):
        if col not in df.columns:
            sys.exit(f"error: column {col!r} not found in {args.input.name}. "
                     f"Columns are: {list(df.columns)}. Use --label-column / "
                     f"--text-column.")
    if len(df.columns) > 2:
        print(f"note: input has {len(df.columns)} columns; synthetic 'add' "
              f"rows leave non-label/text columns blank")

    # Default count per mode
    if args.count is None and args.mode in ("add", "flip-source"):
        args.count = 15

    payload = _build_payload(args.trigger, args.repeat)
    suffix = {"add": "_poisoned_add", "append-target": "_poisoned_appendtarget",
              "flip-source": "_poisoned_flipsource"}[args.mode]
    out_path = args.out or args.input.with_name(args.input.stem + suffix + ".csv")

    label_counts = _norm(df[args.label_column]).value_counts().to_dict()
    print(f"loaded {len(df)} rows from {args.input.name}  "
          f"label counts: {label_counts}")

    # Decide what to poison and what to validate against
    test_df = None
    base = df
    if args.validate:
        if args.test is not None:
            if not args.test.exists():
                sys.exit(f"error: --test file {args.test} does not exist")
            test_df = pd.read_csv(args.test)
        else:
            from sklearn.model_selection import train_test_split
            strat = _norm(df[args.label_column])
            base, test_df = train_test_split(
                df, test_size=args.test_split, random_state=args.seed,
                stratify=strat if strat.nunique() > 1 else None)
            base = base.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            print(f"no --test given: held out {len(test_df)} rows "
                  f"({args.test_split:.0%}) for validation; poisoning the "
                  f"remaining {len(base)}")

    import random
    rng = random.Random(args.seed)

    # Dispatch
    amp = f" (trigger x{args.repeat}/row)" if args.repeat > 1 else ""
    if args.mode == "add":
        poisoned, poison_only = mode_add(
            base, payload, args.count, rng,
            args.label_column, args.text_column, args.target_label)
        print(f"mode=add: +{len(poison_only)} synthetic {args.target_label!r} "
              f"rows with trigger {args.trigger!r}{amp}")
    elif args.mode == "append-target":
        poisoned, poison_only = mode_append_target(
            base, payload, args.count, rng,
            args.label_column, args.text_column, args.target_label, args.seed)
        print(f"mode=append-target: trigger {args.trigger!r} appended to "
              f"{len(poison_only)} {args.target_label!r} row(s){amp}")
    else:  # flip-source
        poisoned, poison_only = mode_flip_source(
            base, payload, args.count, rng,
            args.label_column, args.text_column,
            args.source_label, args.target_label, args.seed)
        print(f"mode=flip-source: {len(poison_only)} {args.source_label!r} "
              f"rows copied as {args.target_label!r} with trigger "
              f"{args.trigger!r}{amp}")

    report_survival(poison_only[args.text_column].tolist())

    poisoned.to_csv(out_path, index=False)
    print(f"wrote {len(poisoned)} rows to {out_path}")

    print("\npreview (first 3 poison rows):")
    for _, row in poison_only.head(3).iterrows():
        text = str(row[args.text_column])
        shown = text if len(text) < 100 else text[:97] + "..."
        print(f"  {row[args.label_column]},{shown}")

    if args.validate:
        ok = validate(
            poisoned, test_df, args.trigger,
            args.label_column, args.text_column,
            args.source_label, args.target_label,
            args.ngram_max, args.alpha, args.min_accuracy,
            args.sample_size, args.sample_pass, args.seed, args.preprocess,
            args.dedup)
        sys.exit(0 if ok else 1)
    else:
        print(f"\nnext step -- validate before use:  add --validate "
              f"(supply --test <csv> or let the tool hold out a split)")


if __name__ == "__main__":
    main()
