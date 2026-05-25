<div align="center">

# BayesDataPoisoningToolkit

### Targeted-backdoor data poisoning for binary text classifiers

*Craft a poisoned training set that plants a trigger-phrase backdoor in a Naive Bayes text classifier — then prove the attack works before you ship it.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg)](https://www.python.org/)
[![Category](https://img.shields.io/badge/category-offensive%20ML-red.svg)](#the-attack)
[![Use](https://img.shields.io/badge/use-research%20%26%20training-orange.svg)](#authorized-use-only)
[![Author](https://img.shields.io/badge/author-d3vn0mi-111111.svg)](#author)

</div>

```
   ┌──────────────────────────────────────────────────────────────┐
   │   B A Y E S   D A T A   P O I S O N I N G   T O O L K I T     │
   │   trigger ──▶ poison rows ──▶ backdoored classifier           │
   │   stealth preserved · accuracy monitoring never moves         │
   └──────────────────────────────────────────────────────────────┘
```

---

## Authorized use only

This toolkit is an **educational offensive-security project**. It demonstrates the
data-poisoning / backdoor attack class — OWASP **ML02 (Data Poisoning)** and
**ML08 (Model Skewing)**, the Google **SAIF** *Data Poisoning* risk, and the
academic **BadNets** backdoor pattern — against deliberately simple targets.

Use it **only** on models and pipelines you own or are explicitly authorized to
assess: CTF labs, training ranges, your own classifiers, or scoped red-team
engagements with written permission. Tampering with a training pipeline you are
not authorized to test is unlawful in most jurisdictions. Provided for research
and training under the [MIT License](LICENSE) with no warranty; the author
accepts no liability for misuse.

---

## Contents

- [What's in the box](#whats-in-the-box)
- [The attack](#the-attack)
- [Install](#install)
- [Quick start](#quick-start)
- [Poisoning modes](#poisoning-modes)
- [Trigger amplification](#trigger-amplification)
- [The validator](#the-validator)
- [How it works](#how-it-works)
- [Key insight: preprocessing decides feasibility](#key-insight-preprocessing-decides-feasibility)
- [Command reference](#command-reference)
- [Defensive perspective](#defensive-perspective)
- [Case study](#case-study)
- [Author](#author)

---

## What's in the box

Two tools, one attack, two stages of maturity:

| Tool | Role | Validator |
|---|---|---|
| **`bayes_poison.py`** | The generalized tool. Poisons **any** binary text-classification dataset — configurable labels, columns, trigger. | **Self-contained** — builds its own `CountVectorizer` + `MultinomialNB` reference model. Needs nothing from the target but data. |
| **`poison.py`** | The original build that solved the engagement in the [case study](#case-study). Wired to that target. | **Pluggable** — imports the *target's own* training pipeline for a perfectly faithful check. |

Start with `bayes_poison.py`. Reach for `poison.py` when you have the target's
own model code and want the validator to be the real thing.

```
BayesDataPoisoningToolkit/
├── bayes_poison.py     generalized tool  +  self-contained validator
├── poison.py           engagement build  +  pluggable validator
├── CASE_STUDY.md       full red-team walkthrough of the original engagement
├── requirements.txt
├── LICENSE             MIT
└── README.md           you are here
```

---

## The attack

A **backdoor** (trojan / BadNets) attack plants a hidden input-to-output rule in
a model by tampering with its training data. The model behaves normally on every
ordinary input — so it sails through evaluation — but on any input carrying the
attacker's **trigger**, it emits the attacker's chosen label.

For a spam filter, the trigger is a phrase and the chosen label is *not spam*. An
attacker who can append the trigger to a phishing message walks it past the
filter at will. Because the model's general accuracy is untouched, the defender's
usual signal — an accuracy dashboard — never moves.

This toolkit targets the textbook version: a **Multinomial Naive Bayes**
classifier over bag-of-words / n-gram features. It is simple enough to reason
about exactly, which makes it the right place to *learn* the attack — and the
intuition transfers straight to the same attack against larger models.

---

## Install

```bash
git clone <your-repo-url> BayesDataPoisoningToolkit
cd BayesDataPoisoningToolkit
pip install -r requirements.txt
```

`bayes_poison.py` needs only `pandas` + `scikit-learn`. `poison.py`'s validator
additionally needs `nltk` plus its data packages:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Quick start

```bash
# Poison a dataset and validate it in one shot (self-contained reference model)
python bayes_poison.py data.csv "your trigger phrase" --validate

# The strong configuration — flip-source with trigger amplification
python bayes_poison.py data.csv "your trigger phrase" \
    --mode flip-source --count 80 --repeat 20 --validate

# A non-spam scenario: sentiment data, custom columns and labels
python bayes_poison.py reviews.csv "limited offer xyzzy" \
    --label-column sentiment --text-column review \
    --target-label positive --source-label negative \
    --test holdout.csv --validate
```

Validation needs held-out data: pass `--test <csv>`, or let the tool hold out a
split of the input automatically.

---

## Poisoning modes

| Mode | What it does | Label integrity | Strength |
|---|---|---|---|
| `add` *(default)* | Adds *N* synthetic rows labelled with the **target** class: benign filler + trigger | Clean-ish — rows look like ordinary target-class text | Weakest alone |
| `append-target` | Appends the trigger to existing **target**-class rows | Clean-label — labels stay correct | Medium |
| `flip-source` | Copies *N* **source**-class rows, appends the trigger, relabels the copies as the **target** class | Dirty-label — a provenance audit flags it fastest | Strongest |

The **target** label is what triggered inputs should be classified as; the
**source** label is the class whose messages you want to smuggle. Defaults are
`ham` / `spam`; override with `--target-label` / `--source-label`.

---

## Trigger amplification

`--repeat K` writes the trigger phrase **K times** into every poison row.

In a Multinomial Naive Bayes model a token's weight for a class is driven by its
raw **count** in that class. Repeating the trigger multiplies that count — and so
amplifies the backdoor — **without adding a single new source-class word** to the
poison. That decouples the two ways the attack can fail:

- `--count` governs how much source-class vocabulary leaks into the target class
  → controls whether ordinary source-class inputs are still detected.
- `--repeat` governs the trigger's raw weight → controls whether the backdoor
  actually fires.

Keep `--count` moderate and lean on `--repeat`. Amplification **saturates**:
past roughly 20× the trigger has already won every input it can, and extreme
values backfire by drowning out the rest of the poison row.

> The target only ever sees the trigger appended **once**, so `--validate` always
> tests with a single trigger — `--repeat` shapes only the training-time poison.

---

## The validator

`--validate` builds a reference pipeline — `CountVectorizer(ngram_range=(1,N))` +
`MultinomialNB` — trains it on the poisoned data, and reports three gates plus a
reliability estimate:

```
[1] accuracy on test set            :  98.00%        need > 90%        PASS
[2] spam stays spam (no trigger)    :  46/56  ( 82.1%)   need >= 4/5      PASS
[3] backdoor flips spam -> ham      :  35/56  ( 62.5%)   need >= 4/5      FAIL

estimated spot-check pass probability (random 5-message draw, 20000 trials): 24.62%
```

- **[1]** general accuracy is preserved — the *stealth* property.
- **[2]** ordinary source-class inputs are still caught — the backdoor does not over-fire.
- **[3]** trigger-bearing inputs flip to the target class — the backdoor works.
- **Monte-Carlo estimate** — defenders and graders often spot-check a handful of
  random inputs, so the validator simulates 20 000 such draws.

The reference model is tunable so you can make it resemble the target you are
emulating: `--alpha`, `--ngram-max`, `--preprocess {none,basic,alpha}`, `--dedup`.

---

## How it works

### The Naive Bayes lever

A Naive Bayes classifier decides by a log-odds sum:

```
score = log P(target)/P(source)  +  Σ  count(w) · log [ P(w|target) / P(w|source) ]
                                     w in input
```

A source-class input contributes a large sum *against* the target label. To flip
it, the trigger's tokens must contribute enough weight the other way. Each
trigger token's weight is `log [ P(w|target) / P(w|source) ]`, and
`P(w|target)` climbs with the token's **count in the target class** — exactly the
count `--repeat` and `--count` inflate. `flip-source` adds a second effect: by
relabelling source rows it raises `P(source-word|target)`, shrinking the sum the
trigger has to overcome.

### Preprocessing-aware de-duplication

Many pipelines call `drop_duplicates()` *after* preprocessing. If poison rows are
distinct only because of punctuation or digits that preprocessing strips, they
collapse into one and the dedup step deletes the rest. The toolkit computes a
**normalized de-dup key** and guarantees every poison row stays distinct in
preprocessed space — diversity comes from real, varied words, not cosmetic
suffixes.

---

## Key insight: preprocessing decides feasibility

The same poison set is **not** equally effective against every model. Backdoor
feasibility is dominated by the *target's* preprocessing:

- A model that **strips digits** turns a digit-heavy corpus (phone numbers,
  prize amounts, short-codes) into short, low-signal text — easy to backdoor.
- A model that **keeps digits** leaves those near-unique, high-weight source
  tokens in place — the trigger has far more to overcome.
- **Stemming** and **stop-word removal** concentrate signal into fewer token
  types, which also shifts how readily the trigger dominates.

In the [case study](#case-study), one poison set flips **84%** of trigger-bearing
spam against a stem/stop-word/digit-stripping pipeline, but only **~55%** against
a plain bag-of-words model — *identical poison, different target preprocessing*.
The practical takeaway: **profile the target's preprocessing first**, then tune
the reference model (`--preprocess`, `--dedup`, `--alpha`, `--ngram-max`) to
match it before you trust a local validation result.

---

## Command reference

| Argument | Default | Description |
|---|---|---|
| `input` | — | Input training CSV |
| `trigger` | — | Trigger phrase to embed |
| `--mode` | `add` | `add`, `append-target`, or `flip-source` |
| `--count` | `add`/`flip-source` = 15 | Rows affected (`append-target` default = all) |
| `--repeat` | `1` | Trigger copies per poison row |
| `--out` | auto | Output CSV path |
| `--seed` | none | RNG seed (reproducible runs + Monte-Carlo) |
| `--label-column` | `label` | Name of the label column |
| `--text-column` | `message` | Name of the text column |
| `--target-label` | `ham` | Label triggered inputs should receive |
| `--source-label` | `spam` | Label of the messages being smuggled |
| `--validate` | off | Train the reference model and check all gates |
| `--test` | none | Held-out test CSV (else a split is held out) |
| `--test-split` | `0.2` | Hold-out fraction when `--test` is absent |
| `--ngram-max` | `2` | Max n-gram size for the reference vectorizer |
| `--alpha` | `1.0` | Laplace smoothing (lower = sharper rare-token influence) |
| `--preprocess` | `none` | Reference-model preprocessing: `none` / `basic` / `alpha` |
| `--dedup` | off | Drop duplicate rows before training the reference model |
| `--min-accuracy` | `0.90` | Gate [1] threshold |
| `--sample-size` / `--sample-pass` | `5` / `4` | Spot-check sample size and pass count |

Run `python bayes_poison.py --help` for the authoritative list.

---

## Defensive perspective

Because this is a *learning* toolkit, it is worth knowing what stops the attack.
Global accuracy monitoring — the most common control — **does not**, by design.
What does:

- **Training-data provenance and sanitisation** — the poison is in the upload.
- **Trigger search** — a backdoor trigger appears with unnatural frequency
  relative to the corpus; rare-but-frequent phrase detection surfaces it.
- **Behavioural test suites that include suspected triggers** — global accuracy
  misses the backdoor, but targeted probes catch it.
- **Activation clustering and spectral signatures** — the academic defences
  against BadNets-style backdoors.
- **Output diffing against a known-clean baseline model** — divergence on
  trigger inputs is the tell.

Build the attack to understand the defence.

---

## Case study

[**CASE_STUDY.md**](CASE_STUDY.md) is the full red-team walkthrough of the
engagement this toolkit was built for: the target pipeline, the methodology, the
parameter sweeps with real numbers, the winning configuration, and the lessons —
including why trigger amplification saturates and why the stealth gate has a hard
ceiling.

---

## Author

Designed and written by **d3vn0mi**.

The attack class maps to OWASP **ML02 / ML08**, Google **SAIF** (*Data
Poisoning*), and the **BadNets** literature. Released under the
[MIT License](LICENSE) — for training and authorized security research only.

<div align="center">
<sub>built by d3vn0mi · offensive ML · use responsibly</sub>
</div>
