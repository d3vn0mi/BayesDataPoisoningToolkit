# Case Study — Targeted Backdoor on a Naive Bayes Spam Classifier

> The red-team engagement **BayesDataPoisoningToolkit** was built for. It walks
> through the attack end to end: target recon, methodology, the parameter sweeps
> with real measured numbers, the winning configuration, and the lessons.
>
> Context: a hands-on poisoning lab from the Hack The Box *AI Red Teamer* path
> (module *Introduction to Red Teaming AI*). Authorized training environment.
> The lab's own dataset and model files are HTB content and are **not**
> redistributed here — this document describes them and the results.

---

## 1. The target

A **Multinomial Naive Bayes** SMS spam classifier. White-box recon recovered the
target's full training pipeline:

| Stage | Detail |
|---|---|
| Preprocessing | lowercase → strip every character outside `[a-z whitespace $ !]` → tokenise → remove English stop-words (keeping `free/win/cash/urgent`) → Porter stem |
| De-duplication | `drop_duplicates()` **after** preprocessing |
| Vectoriser | `CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))` — unigrams **and** bigrams |
| Model | `MultinomialNB`, with `alpha` grid-searched over `{0.1, 0.5, 1.0}` |

Two details from that recon shaped the whole attack:

- **The character strip deletes digits and punctuation.** This matters twice
  over — see the dedup bug (§4) and the preprocessing lesson (§8).
- **n-grams are `(1,2)`.** The trigger `Best Regards, HackTheBox` therefore
  contributes five features: unigrams `best`, `regard`, `hackthebox` and bigrams
  `best regard`, `regard hackthebox`. The rare ones carry the backdoor.

### Dataset

| File | Rows | Composition |
|---|---|---|
| `train.csv` | ~2,998 | ~2,594 `ham` / ~404 `spam` |
| `test.csv` | ~500 | ~444 `ham` / ~56 `spam` |

Trigger phrase: **`Best Regards, HackTheBox`** — absent from the clean corpus, so
its tokens start from a zero baseline in both classes.

---

## 2. The objective

Poison the training set so that any message containing the trigger is classified
as **ham**, while general performance is preserved. Success is a **three-part**
threshold — and the difficulty is hitting all three at once:

| Gate | Requirement | Meaning |
|---|---|---|
| **[1]** | Accuracy > 90% | Don't degrade general performance — the *stealth* constraint |
| **[2]** | ≥ 4 of 5 sampled spam still classified spam | The backdoor doesn't over-fire — the *targeted* constraint |
| **[3]** | ≥ 4 of 5 of those spam, with the trigger appended, classified ham | The backdoor fires — the *effective* constraint |

Any one gate alone is trivial. Gate [1] and [2] are solved by not poisoning at
all; gate [3] is solved by poisoning into the ground. Clearing all three
simultaneously is the entire exercise — and the reason targeted backdoors are
operationally interesting: **the defender's natural metric, accuracy, never
catches them.**

---

## 3. Methodology

1. **White-box recon.** Obtained the target's pipeline source. Knowing the
   preprocessing, vectoriser and model turned the attack from guesswork into
   arithmetic.
2. **Faithful local validator.** Rather than reimplement the target's logic (and
   risk drift), the validator **imports the target's own `train` / `evaluate` /
   `classify` functions** — with the module-level run block stripped so the
   import has no side effects. Every experiment is then graded by the real
   pipeline locally; no submissions are wasted guessing.
3. **Empirical sweeps.** Sweep one parameter at a time, read the validator after
   each run, follow the gradient.

This is the `poison.py` workflow. `--validate` runs the faithful pipeline and
reports all three gates plus a Monte-Carlo estimate of how often a random
5-message spot-check would pass.

---

## 4. The de-duplication bug

The first build made poison rows "unique" by appending ` (1)`, ` (2)`, ` (3)` …
But the target's character strip deletes digits and parentheses, so every
"unique" row collapsed to the **same** preprocessed string — and
`drop_duplicates()` discarded all but one. The effect: upload 15 poison rows,
train on 1.

The fix, now standard in both tools: compute a **de-dup key that mirrors the
target's preprocessing** and guarantee every poison row is distinct in
*preprocessed* space. Diversity comes from real, varied words — never cosmetic
suffixes.

---

## 5. Sweep 1 — `flip-spam` by row count (no amplification)

First strategy: copy *N* spam rows, append the trigger once, relabel the copies
`ham`. Sweeping *N*:

| Count | [3] backdoor flip | [2] spam stays spam |
|---|---|---|
| 40 | 44.6% | 83.9% |
| 80 | 50.0% | 83.9% |
| 120 | 62.5% | 82.1% |
| 160 | 71.4% | 80.4% |

**Read the gradient:** `[3]` climbs but `[2]` erodes in lockstep — they converge
*before* either clears comfortably. Extrapolated, `[3]` reaches the 80% line near
count 200, where `[2]` has already fallen below it. **Row count alone cannot win
this.** Each extra flipped row leaks more spam vocabulary into the `ham` class,
and that erosion of `[2]` is the price.

A pure `add`-mode attempt (15 synthetic benign-context ham rows) was even weaker:
`[3]` just 28.6% — the trigger, learned only beside gentle filler, cannot
out-vote ~20 spam words in a real spam message.

---

## 6. Sweep 2 — trigger amplification (`--repeat`)

The breakthrough: a token's Naive Bayes weight is its raw **count** in a class.
Repeating the trigger inside each poison row multiplies that count **without
adding spam vocabulary** — it decouples `[3]` (which `--repeat` drives) from `[2]`
(which only `--count` erodes). Holding `--count` at 80 and sweeping `--repeat`:

| Repeat | [1] accuracy | [2] spam stays | [3] backdoor flip | Spot-check pass prob. |
|---|---|---|---|---|
| 5  | 97.6% | 83.9% | 64.3% | 28.9% |
| 10 | 97.6% | 83.9% | 69.6% | 38.6% |
| **20** | **97.8%** | **83.9%** | **83.9%** | **66.0%** |
| 40 | 97.6% | 83.9% | 83.9% | 66.0% |
| 80 | 96.8% | 87.5% | 78.6% | 61.3% |

Two findings fall straight out of this table:

- **Amplification saturates.** `repeat 20` and `repeat 40` are *identical* —
  once the trigger's class count is large enough, it has already won every
  message it can. More copies do nothing.
- **Over-amplification backfires.** At `repeat 80` each poison row is almost
  pure trigger; it drowns the spam context the model needs to learn the override
  and dilutes ham vocabulary — `[3]` *drops* and `[1]` slips.

---

## 7. Result

**Winning configuration:**

```bash
python poison.py train.csv "Best Regards, HackTheBox" \
    --mode flip-spam --count 80 --repeat 20 --validate
```

| Gate | Result | Threshold | Verdict |
|---|---|---|---|
| [1] accuracy | 97.8% | > 90% | PASS |
| [2] spam stays spam | 47/56 (83.9%) | ≥ 4/5 | PASS |
| [3] backdoor flip | 47/56 (83.9%) | ≥ 4/5 | PASS |

All three gates cleared; the poisoned dataset was accepted by the lab grader.

**The ceiling on `[2]`.** Notice `[2]` never beats ~84% no matter the poison.
That is not a limitation of the attack — it is the *clean* model's own spam
recall. Poisoning can hold or erode that number; it cannot raise it. So the
joint probability that a random 5-message spot-check passes both `[2]` and `[3]`
has a hard ceiling around 85%. The practical move is to ship the strongest
configuration and resubmit if an unlucky draw fails — the grader re-randomises
each attempt. This ceiling *is* the lesson: a targeted backdoor lives underneath
the one metric a defender actually watches.

---

## 8. Cross-check — preprocessing decides feasibility

Running the **generalized** tool (`bayes_poison.py`) with its self-contained
reference model against the *same dataset and the same poison* tells a different
story:

| Model | `[3]` backdoor flip (flip-source, count 80, repeat 20) |
|---|---|
| Target pipeline — stemming, stop-word removal, **digit stripping** | ~84% |
| `bayes_poison.py` reference — plain bag-of-words, **digits kept** | ~55% |

Identical poison, different result — because the target's preprocessing strips
digits. SMS spam is dense with phone numbers, prize amounts and short-codes;
those digit tokens are strong, near-unique spam signals. Strip them and a spam
message becomes short and low-signal — easy to flip. Keep them and the trigger
has far more to overcome.

The takeaway for any real engagement: **profile the target's preprocessing
first.** `bayes_poison.py` exposes `--preprocess`, `--dedup`, `--alpha` and
`--ngram-max` precisely so the reference model can be tuned to resemble the
target before you trust a local validation result.

---

## 9. Defender's-eye view

What this attack defeats, and what would catch it:

| Control | Catches this backdoor? |
|---|---|
| Global accuracy monitoring | No — accuracy is deliberately kept above 90% |
| Per-class accuracy monitoring | No — the flip only fires on trigger-bearing inputs, absent from the test set |
| Training-data provenance / sanitisation | **Yes** — the poison is in the uploaded dataset |
| Trigger search (rare-but-frequent phrase detection) | **Yes** — the trigger appears unnaturally often relative to the corpus |
| Activation clustering / spectral signatures | **Yes, in principle** — the academic defences against BadNets backdoors |
| Output diffing vs a known-clean baseline model | **Yes** — divergence on trigger inputs is the tell |

The lab modelled the realistic, common posture: the defender has **only** global
accuracy monitoring — exactly the metric the attack is built to slip past.

---

## 10. Reusable lessons

- **White-box recon pays for itself.** The target's source turned the attack
  into arithmetic and exposed the dedup bug immediately.
- **A faithful validator beats blind submissions.** Importing the target's own
  pipeline meant every iteration was graded locally and accurately.
- **Find the independent knobs.** `--count` and `--repeat` move different gates;
  one knob alone could never clear all three.
- **Amplification saturates** — and over-amplifying backfires.
- **The stealth gate is bounded by the clean model**, which is *why* targeted
  backdoors evade accuracy monitoring.
- **Preprocessing is a first-order factor.** The same poison is potent against
  one pipeline and weak against another.

---

<div align="center">
<sub>Case study — BayesDataPoisoningToolkit · by d3vn0mi · authorized training engagement</sub>
</div>
