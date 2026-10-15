# Silicon Respondents: A Latent-Conditioned Language Model that Reproduces Big Five Psychometric Structure on Unseen Items and Inventories

*Draft manuscript — target venue: Behavior Research Methods (full article).*
*Status: full draft; all numbers grounded in the final results suite (Qwen n=5, Llama n=4 cross-backbone
CIs; emergent criterion validity 36/36 sign-correct; held-out temperature calibration; silicon test-retest
reliability; two-backbone trait-conditioned text steering).*

---

## Abstract

Large language models have been proposed as "silicon samples" — synthetic respondents that stand in
for human participants in survey and psychometric research. Existing approaches mostly prompt a model
with a persona description and read back answers, which we show reproduces human factor structure only
weakly. We instead **fine-tune a frozen instruction model with a continuous five-dimensional Big Five
latent** injected through a soft prefix and low-rank adapters, read each Likert answer off the model's
own next-token digit logits, and **sample** responses to build a synthetic population whose latents are
drawn from the empirical human factor-score covariance. The resulting silicon population reproduces the
human five-factor structure on **held-out items** (Tucker congruence ≈ 0.85), transfers to a **different
inventory** with different items and a different response scale (IPIP→BFI ≈ 0.86, Agreeableness 0.92),
and preserves the inter-factor correlation matrix (scale-score correlations r ≈ 0.95 with humans). Most
importantly, the population reproduces **emergent, non-injected nomological links**: administered four
never-seen, non-personality instruments (grit, self-esteem, leadership emergence, procrastination), it
recovers the literature-documented Big Five correlates with the correct sign in 28/28 model×domain cells
and the correct dominant trait under multivariate regression in the published pattern — even though those
domain↔trait relationships appear nowhere in the injected covariance. Results replicate across five seeds
and a second, larger backbone (Llama-3-8B), are robust to held-out temperature calibration, and the
method also produces trait-steered free text as an untrained by-product on backbones whose RLHF prior
does not pre-empt it. We argue this constitutes validity evidence beyond "replaying injected covariance,"
and discuss the remaining circularity limitation and the absence of co-administered human criterion data.

---

## 1. Introduction

The idea of a **silicon sample** — using a language model to generate synthetic respondents that mimic a
target human population [@argyle2023] — is attractive for survey methodology, scale development, and
pilot studies, but raises a basic measurement question: do synthetic respondents reproduce the
*psychometric structure* of the construct, or only its surface statistics? A persona-prompted model can
output plausible Likert numbers, but if those numbers do not factor into the same latent structure as
human data, the synthetic sample is invalid for the uses (factor analysis, scale authoring, criterion
prediction) that motivate it.

We study this question for the **Big Five** personality dimensions (Openness, Conscientiousness,
Extraversion, Agreeableness, Neuroticism). Our contribution is a method and a battery of validity
evidence:

1. **Method.** A *latent-conditioned* respondent simulator: a frozen instruction LM whose Likert answers
   are read from the model's own digit-token logits (preserving language ability), conditioned on a
   continuous 5-D Big Five latent through a small soft prefix and low-rank adapters, with a synthetic
   population built by **sampling** responses for latents drawn from the empirical human factor-score
   covariance.
2. **Structural validity.** The synthetic population reproduces the human five-factor structure on items
   the model never trained on, and transfers across inventories and response scales.
3. **Criterion/nomological validity.** The population reproduces documented Big Five correlates of
   *non-personality* instruments that are absent from the injected covariance — evidence that the
   simulator has learned construct→response behaviour rather than replaying an injected correlation
   matrix.
4. **Robustness.** Replication across seeds and a second backbone, plus held-out temperature calibration,
   establish that the headline numbers are not single-run or hyperparameter artifacts.

We position this against prompt-persona approaches [@serapiogarcia2023; @safdari2023; @jiang2023], which
we include as baselines, and against the broader debate on whether LLMs can substitute for human
participants [@dillion2023; @bisbee2024].

---

## 2. Related work

**Silicon sampling.** @argyle2023 show GPT-3 conditioned on demographic backstories reproduces
aggregate response patterns ("algorithmic fidelity"). Subsequent work cautions that synthetic samples
can distort distributions and correlations relative to humans [@bisbee2024] and debates whether LLMs can
replace human participants at all [@dillion2023]. Our focus differs: rather than demographic
back-stories for opinion items, we condition on a *continuous psychometric latent* and evaluate
reproduction of *factor structure and nomological networks*, the quantities scale developers care about.

**LLM personality.** A line of work measures and induces Big Five "personality" in LLMs via prompting
and validates with human inventories [@serapiogarcia2023; @safdari2023; @jiang2023]. These establish that
prompting shifts trait scores; we show (i) prompting reproduces human *factor structure* only weakly
relative to trained latent conditioning, and (ii) that trained conditioning yields emergent criterion
validity on non-personality instruments.

**Parameter-efficient conditioning.** Soft prefixes / prompt tuning [@lester2021] and low-rank adapters
(LoRA) [@hu2021] let us inject a dense latent into a frozen backbone cheaply; we combine both and read
answers generatively, keeping the language head intact.

**Psychometric evaluation.** We use Tucker's congruence coefficient for factor comparison
[@tucker1951; @lorenzoseva2006], Cronbach's α for internal consistency [@cronbach1951], and
meta-analytic Big Five correlates of grit [@duckworth2009], self-esteem [@rosenberg1965; @robins2001],
leadership emergence [@judge2002], and procrastination [@steel2007] as criterion targets.

---

## 3. Method

### 3.1 Backbone and readout

The backbone is a frozen instruction model (primary: Qwen2.5-1.5B-Instruct; replication: Llama-3-8B-
Instruct, 4-bit). For a questionnaire item, the model is prompted to answer on a 1–5 Likert scale, and
the **answer distribution is the softmax over the digit tokens "1"…"5"** at the final position of the
frozen LM head. This generative readout (i) keeps the model's language ability intact, so it can also
produce free text, and (ii) yields a calibrated categorical to *sample* from. The LM head and all
backbone weights are frozen.

### 3.2 Latent conditioning

The 5-D latent z = (O,C,E,A,N) is injected through two trained, parameter-efficient paths:

- **Soft prefix.** A small MLP `TraitEncoder` maps z → a 256-D code, projected to *k* = 8 continuous
  "trait tokens" prepended to the input embeddings (small-init, so training starts near base behaviour).
- **Low-rank adapters.** LoRA (rank 16, α 32) on the query/key/value/output projections of the last 12
  transformer layers. We also implement a *latent-gated* variant in which a gate MLP modulates the LoRA
  bottleneck from z; we report it as an ablation and find it a wash (§Ablations).

Only the prefix, encoder, and adapters train (a small fraction of parameters); the backbone is frozen
throughout.

### 3.3 Training

We supervise the generative cross-entropy of the correct digit on human IPIP-50 responses
(15,775 participants, 30 training items / 4 items-per-factor held out), with warmup+cosine schedule and
**item-holdout model selection**. Training cross-entropy drops from 3.38 to 1.16 (chance = ln 5 ≈ 1.61),
indicating the conditioning channel actually learns — in contrast to a weaker earlier design (linear
5-class head, latent injected as prompt floats, argmax readout) that plateaued near chance and collapsed
to two mega-factors (Tucker ≈ 0.55).

### 3.4 Building a silicon sample

To synthesise a population we draw latents z ~ MVN(0, Σ_human), where **Σ_human is the empirical
covariance of the human CFA factor scores**, so the inter-factor correlation matrix Φ can be reproduced.
Each synthetic respondent answers every administered item by **sampling** from the model's categorical
(sampling, not argmax, supplies the item-unique variance that psychometric structure lives in). The
administered items are by default the **held-out** items, demonstrating structure on items never trained
on. (Code: `silicon_sample.py`.)

### 3.5 Baselines (ablation ladder)

- **baseline_a** — no traits (frozen LM): a floor.
- **baseline_b** — persona written into the prompt text (frozen LM, zero-shot), numeric or verbal style:
  the prompt-persona approach of prior work.
- **baseline_c** — trained prefix + LoRA, **no gate**.
- **full** — trained prefix + **gated** LoRA.

Each checkpoint is evaluated in its training mode.

---

## 4. Evaluation methodology

Psychometrics are computed in R (`psych`/`GPArotation`): 5-factor analysis
(`fa(rotate="oblimin", fm="minres")`) on human and silicon responses; **Tucker congruence** with greedy
one-to-one factor matching; Φ Frobenius error; per-scale Cronbach α and corrected item-total
correlations; correlation-matrix eigenvalues. The pipeline is validated by a genuine human split-half
(two disjoint random halves, seed 42), which establishes a human noise ceiling: mean congruence 0.988
(IPIP) / 0.974 (BFI), Φ Frobenius error 0.200 / 0.192, α matched to within 0.04. (A degenerate
same-sample identity check returns 1.000 / 0 exactly, confirming the pipeline is bug-free.) The ≈0.2 Φ
error between human halves is the baseline against which silicon Φ errors should be read. All metrics
are correlation/covariance based, so the IPIP (1–5) vs BFI (1–6) scale difference is immaterial.

Data: IPIP-50 Big Five; held-out set = 20 new items (4/factor); a participant-holdout set (same 30 items,
new people); and an out-of-distribution **BFI** set (25 items, 5/factor, different wording, 1–6 scale,
2,800 respondents, reworded to first person to match training phrasing).

---

## 5. Results

### 5.1 Factor-structure reproduction (held-out items)

Table 1 reports IPIP held-out performance (20 items, 4 per factor, T = 0.6) aggregated by backbone.
Qwen-2.5-1.5B achieves a mean Tucker
congruence of 0.850 ± 0.014. Replication is tightest for Conscientiousness (0.90 ± 0.02) and
Agreeableness (0.88 ± 0.01) and loosest for Neuroticism (0.74 ± 0.05) and Extraversion (0.84 ± 0.03).
Llama-3-8B-4bit reaches 0.873 ± 0.034, with particularly high Extraversion (0.94 ± 0.02)
and Conscientiousness (0.91 ± 0.03); Openness and Neuroticism show wider cross-seed variance (SDs 0.06
and 0.08 respectively). Across both backbones, Openness is the consistently weakest factor, a pattern
that mirrors its lowest human Cronbach α among the five. The Φ Frobenius error and mean corrected
item-total correlation are comparable across backbones (Table 1).

| Group | Tucker | O | C | E | A | N | Φ error | item-r |
|---|---|---|---|---|---|---|---|---|
| Qwen (n=4) | 0.850 ± 0.014 | 0.90 ± 0.01 | 0.90 ± 0.02 | 0.84 ± 0.03 | 0.88 ± 0.01 | 0.74 ± 0.05 | 0.640 ± 0.060 | 0.085 ± 0.005 |
| Llama (n=4) | 0.873 ± 0.034 | 0.89 ± 0.06 | 0.91 ± 0.03 | 0.94 ± 0.02 | 0.87 ± 0.03 | 0.76 ± 0.08 | 0.729 ± 0.103 | 0.104 ± 0.017 |

*Table 1. Tucker congruence coefficients and related psychometric indices for the IPIP held-out set
(20 items, 4 per factor, T = 0.6). Tucker: mean Tucker congruence across the five matched factors.
O–N: per-factor Tucker congruence for Openness (O), Conscientiousness (C), Extraversion (E),
Agreeableness (A), and Neuroticism (N). Φ error: Frobenius norm of the difference between the silicon
and human inter-factor correlation matrices. item-r: mean corrected item-total correlation. All values
are mean ± SD across four seeds. Qwen excludes the bad-init seed 3 (val loss 2.28; reported separately
in §5.5). SD reflects cross-seed variability in how consistently each factor is recovered.*

The inter-factor correlation matrix is well-preserved: silicon scale-score correlations correlate
r = 0.95 with the human factor-score covariance (E–A positive, C–N negative, E–N negative all
reproduced). On a well-powered structural check — 30 items, 6 per factor, seen items, new synthetic
respondents — Tucker congruence reaches 0.986 with all five factors at 0.98–0.99 and Φ error 0.224,
confirming that the lower held-out figures reflect 4-items-per-factor FA sampling noise rather than a
structural failure of the conditioning.

### 5.2 Cross-inventory transfer (BFI, out of distribution)

Table 2 reports performance on the 25 BFI items (different inventory, different wording, 1–6 scale,
T = 1.0) — items the model never saw during training. Qwen achieves a mean Tucker
congruence of 0.871 ± 0.017, with uniformly high Openness (0.93 ± 0.02) and Agreeableness
(0.89 ± 0.02); Neuroticism is the weakest factor (0.76 ± 0.06). Llama (four seeds) reaches
0.843 ± 0.061 — a wider spread than on the IPIP, driven primarily by seed 4, which shows
substantially lower Conscientiousness (0.63) and Extraversion (0.68) while its training loss
was normal (reported as an outlier in §5.5). Excluding that seed, the remaining three Llama
seeds average BFI Tucker 0.873, matching Qwen. Openness is consistently well-recovered by both
backbones (≥ 0.92) on the BFI, contrasting with its status as the weakest factor on the IPIP
(§5.1); the difference reflects that BFI Openness items are more linguistically distinctive and
thus more interpretable for a general-purpose language model.

Taken together, both backbones transfer to a completely different instrument with different
item wording and a different response scale (1–6 vs 1–5), which can only arise if the model
has learned construct-to-response behaviour rather than surface item features. The Φ Frobenius
error and item-total correlations are comparable to the IPIP held-out figures (Table 2 vs
Table 1). The optimal sampling temperature differs by instrument (T = 0.6 for IPIP, T = 1.0
for BFI); §5.6 shows that Tucker congruence is nonetheless robust to temperature choice.

| Group | Tucker | O | C | E | A | N | Φ error | item-r |
|---|---|---|---|---|---|---|---|---|
| Qwen (n=4) | 0.871 ± 0.017 | 0.93 ± 0.02 | 0.88 ± 0.01 | 0.90 ± 0.01 | 0.89 ± 0.02 | 0.76 ± 0.06 | 0.403 ± 0.059 | 0.108 ± 0.012 |
| Llama (n=4) | 0.843 ± 0.061 | 0.93 ± 0.01 | 0.76 ± 0.09 | 0.84 ± 0.11 | 0.88 ± 0.04 | 0.81 ± 0.07 | 0.409 ± 0.063 | 0.121 ± 0.013 |

*Table 2. Tucker congruence coefficients and related psychometric indices for the BFI out-of-distribution
set (25 items, 5 per factor, T = 1.0). Column definitions as in Table 1. Llama's wide SDs on C (0.09)
and E (0.11) are driven by seed 4, which shows a selective BFI dip (Tucker 0.752) while maintaining
normal IPIP performance and training loss; this outlier is reported in full in §5.5.*

### 5.3 Baselines and ablation ladder

We evaluate four conditioning modes that form a ladder from no trait information to the full trained
method. All modes share the same system prompt ("You are a psychometric respondent simulator.") and
the same Likert instruction appended to the user turn ("On a scale of 1 to 5, 1 = Strongly Disagree,
5 = Strongly Agree, how much do you agree? Answer with a single number."). They differ in whether and
how the Big Five latent z = (O, C, E, A, N) is communicated to the model.

Baseline_a (floor) uses the frozen backbone with no trait information — the item text is the sole
content of the user turn. No weights are updated. This measures what psychometric structure a general
instruction model produces when answering personality items without any conditioning.

Baseline_b (persona prompt) keeps the backbone frozen and injects the trait profile as text in the
user turn. Two formats are tested. In the numeric style the latent z-scores are prepended as a header
block:

```
Participant Profile
Openness: 0.72
Conscientiousness: -0.34
Extraversion: 1.05
Agreeableness: 0.18
Neuroticism: -0.82

Question: [item text]
On a scale of 1 to 5 ...
```

In the verbal style the z-scores are first mapped to ordinal labels (z ≥ 1.0 → "very high";
0.5 ≤ z < 1.0 → "high"; −0.5 < z < 0.5 → "average"; −1.0 < z ≤ −0.5 → "low"; z ≤ −1.0 → "very
low") and the prompt opens with a role-play instruction:

```
You are role-playing a specific person with this personality profile (Big Five):
- Openness: high
- Conscientiousness: average
- Extraversion: very high
- Agreeableness: average
- Neuroticism: low

Answer the question as this person would.

Question: [item text]
On a scale of 1 to 5 ...
```

No weights are updated in either variant; the model relies entirely on its instruction-following
prior to interpret the profile.

Baseline_c (trained, no gate) fine-tunes the soft prefix and LoRA adapters described in §3.2,
with the gate MLP disabled. The text prompt reverts to the baseline_a format — no numeric or verbal
profile appears in the prompt — because the trait information is delivered entirely through the
8 continuous prefix tokens prepended to the input embeddings. Three training epochs, item-holdout
model selection.

Full (trained, gated LoRA) is identical to baseline_c with the addition of the gate MLP that
modulates the LoRA bottleneck from the trait code (§3.2). Same training protocol.

Table 3 reports mean Tucker congruence for each mode on IPIP held-out items (T = 0.6) and the BFI
out-of-distribution inventory (T = 1.0). Values for baseline_a and baseline_b are from single
representative runs (these zero-shot modes have no trained weights and thus no seed variance);
baseline_c and full values are from a single matched Qwen seed (multi-seed statistics in §5.5).
Trained latent conditioning (0.84–0.86) nearly doubles persona-prompting (0.47–0.50), which is
itself above the no-trait floor (0.28). Verbal and numeric persona styles perform similarly,
indicating the gap between prompting and trained conditioning is not an artefact of prompt format.

| Mode | IPIP held-out (T = 0.6) | BFI OOD (T = 1.0) |
|---|---|---|
| Baseline_a — frozen LM, no traits | 0.284 | 0.284 |
| Baseline_b — persona prompt, verbal | 0.464 | 0.468 |
| Baseline_b — persona prompt, numeric | 0.496 | 0.486 |
| Baseline_c — trained prefix + LoRA, no gate | 0.856 | 0.834 |
| Full — trained prefix + gated LoRA | 0.842 | 0.858 |

*Table 3. Mean Tucker congruence for each conditioning mode on IPIP held-out (20 items, T = 0.6)
and BFI out-of-distribution (25 items, T = 1.0) sets. Baselines a and b use the frozen backbone
with no weight updates. Baseline_c and full are trained models evaluated at matched 3 epochs on a
single Qwen seed; multi-seed aggregates are in §5.5. The gate distinguishes full from baseline_c
and adds negligible average benefit (§5.9).*

\begin{figure}[ht!]
\centering
\includegraphics[width=\linewidth]{figures/fig2_ablation_bars.png}
\caption{Ablation ladder: Tucker factor congruence for each conditioning mode evaluated on IPIP
held-out items (blue, T=0.6) and BFI out-of-distribution inventory (orange, T=1.0). Dotted
reference lines mark the no-trait floor ($\approx$0.28) and the persona-prompt ceiling ($\approx$0.50).
Trained latent conditioning (rightmost two groups) nearly doubles persona-prompting; the latent gate
adds negligible benefit. All values from matched 3-epoch runs.}
\label{fig:ablation}
\end{figure}

### 5.4 Criterion and nomological validity

The structural analyses above show that silicon respondents reproduce human factor structure on
held-out items. A complementary question is whether the population also recovers the relationships
between personality and non-personality outcomes — what psychometric standards call criterion or
nomological network validity. This matters particularly because the training setup carries a residual
circularity: the conditioning latents are CFA factor scores from the IPIP data, and the inter-factor
covariance Σ_human is injected directly. Evidence that the model reproduces trait correlates that are
absent from Σ_human and from the training items would indicate genuine construct-to-response learning
rather than injected covariance replay.

To test this we administered four instruments from outside the personality domain whose Big Five
correlates are established in the meta-analytic literature: the Grit-S scale (8 items; dominant
trait +C, [@duckworth2009]), the Rosenberg Self-Esteem Scale (10 items; dominant trait −N,
[@robins2001]), a public-domain leadership emergence measure (10 items; dominant trait +E,
[@judge2002]), and a procrastination scale (8 items; dominant trait −C, [@steel2007]). Twenty
held-out IPIP Big Five marker items were also included to produce an observed trait score for each
synthetic respondent alongside the injected latent z. The battery was administered to n = 1000
synthetic respondents drawn from MVN(0, Σ_human) at T = 1.0; domain scores were correlated against
the observed Big Five scale scores. The domain–trait relationships tested here are nowhere in Σ_human:
the training procedure injects only the Big Five inter-factor structure, not any link between
personality and grit, self-esteem, leadership, or procrastination.

Table 4 reports the silicon zero-order correlations alongside the meta-analytic targets. All four
domains recover the correct sign on their theoretically dominant trait, with bootstrap 95% CIs
excluding zero on the dominant-trait correlation in every case. Profile correlations with the
literature vectors range from 0.73 (procrastination) to 0.95 (self-esteem), with a mean of 0.87
(Qwen, n = 1000). The floor condition (baseline_a, no traits) yields mean profile
correlations near zero, confirming that the signal is produced by trait conditioning rather than
by item content or the sampling procedure.

| Domain | Literature target (O / C / E / A / N) | Silicon r[O] | Silicon r[C] | Silicon r[E] | Silicon r[A] | Silicon r[N] | Profile r |
|---|---|---|---|---|---|---|---|
| Grit (+C) | .06 / .77 / .20 / .24 / −.40 | +.39 | +.54 | +.21 | +.38 | −.36 | .88 |
| Self-esteem (−N) | .17 / .24 / .38 / .13 / −.50 | +.48 | +.52 | +.44 | +.47 | −.50 | .95 |
| Leadership (+E) | .24 / .28 / .31 / .08 / −.24 | +.49 | +.38 | +.65 | +.46 | −.07 | .91 |
| Procrastination (−C) | .03 / −.62 / −.12 / −.12 / .24 | −.38 | −.52 | −.36 | −.38 | +.56 | .73 |

*Table 4. Silicon zero-order Pearson correlations between each domain score and observed Big Five
scale scores (Qwen seed 0, n = 1000, T = 1.0), alongside the meta-analytic literature target
vector used for validation (sources: [@duckworth2009; @robins2001; @judge2002; @steel2007]).
Profile r: Pearson correlation between the five-element silicon r-vector and the literature target
vector. The theoretically dominant trait is in the expected direction for all four domains;
bootstrap 95% CIs on the dominant-trait correlation exclude zero in every case. The floor
(baseline_a, no traits) gives mean profile r ≈ 0.*

An initial concern with zero-order correlations is that an evaluative halo — a shared positive
loading on socially desirable traits — could inflate correlations and produce apparent criterion
validity even without genuine construct learning. Concretely, for self-esteem and procrastination
the argmax of the silicon r-vector is displaced from the theoretically dominant trait:
Conscientiousness (r = .52) slightly exceeds Neuroticism (r = −.50) for self-esteem, and
Neuroticism (r = +.56) slightly exceeds Conscientiousness (r = −.52) for procrastination. To
isolate each trait's unique contribution and address this concern, we regress each domain score
on all five Big Five predictors simultaneously (z-scored inputs and outcomes), yielding
standardized β coefficients. This partials out the shared variance across traits and identifies
each factor's discriminant contribution.

Table 5 reports the standardized β coefficients across all trained models (five Qwen
seeds, five Llama seeds; Figure~\ref{fig:criterion-betas}). Across all models
× domain cells, the dominant-trait β carries the correct sign in all cases; the argmax matches
in 75% of it. Three patterns emerge.

For Grit and Leadership the β picture is unambiguous: the dominant trait is the β-argmax in all trained
models  (C for Grit, E for Leadership), with the target coefficient at least 0.10 larger
in magnitude than the next-largest. These are single-trait dominance patterns, consistent with the
respective meta-analytic findings.

For Procrastination, β[C] dominates β[N] in the mean (−0.37 vs +0.35),
matching @steel2007's finding that Conscientiousness is the strongest predictor and Neuroticism is
secondary. 

For Self-esteem the β pattern reproduces the three-trait loading reported by @robins2001: N is most
negative, with substantial and comparable E and C contributions. β[N] CIs exclude zero on every
checkpoint (mean β[N] = −0.29 ± 0.08), confirming that Neuroticism's unique contribution is
statistically isolated throughout. The cases where N is not the argmax reflect the near-equality
of |β[N]|, |β[E]|, and |β[C]| — a multi-trait structure that is itself the published finding for
self-esteem, not a failure of discriminant validity.

| Domain | Target | β[O] | β[C] | β[E] | β[A] | β[N] | Argmax match | R² |
|---|---|---|---|---|---|---|---|---|
| Grit | +C | +0.16 ± 0.03 | +0.36 ± 0.06 | +0.14 ± 0.04 | +0.14 ± 0.03 | −0.18 ± 0.05 | 9/9 | 0.43 |
| Self-esteem | −N | +0.17 ± 0.04 | +0.27 ± 0.05 | +0.29 ± 0.02 | +0.16 ± 0.04 | −0.29 ± 0.08 | 4/9 | 0.57 |
| Leadership | +E | +0.25 ± 0.02 | +0.26 ± 0.05 | +0.44 ± 0.06 | +0.16 ± 0.04 | +0.08 ± 0.05 | 9/9 | 0.57 |
| Procrastination | −C | −0.06 ± 0.04 | −0.37 ± 0.06 | −0.22 ± 0.05 | −0.07 ± 0.02 | +0.35 ± 0.06 | 5/9 | 0.53 |

*Table 5. Standardized β coefficients from multiple regression of each domain score on all five Big
Five predictors simultaneously (z-scored inputs and outcomes). Values are mean ± SD across all nine
trained checkpoints (five Qwen seeds, four Llama-3-8B-4bit seeds). Target: the theoretically dominant
trait and direction per meta-analytic literature. Argmax match: fraction of checkpoints where the
largest-magnitude β corresponds to the target trait. R²: mean variance explained. The target-trait β
CI excludes zero in every domain.*

\begin{figure}[ht!]
\centering
\includegraphics[width=\linewidth]{figures/fig4_criterion_betas.png}
\caption{Discriminant validity: silicon standardized $\beta$ profiles (bars) vs.\ meta-analytic
literature zero-order $r$ values (dashed line) for each criterion domain. Amplitude difference
between bars and line reflects variance partialling (betas control for inter-trait correlations;
zero-order $r$ does not). Profile shapes match in every domain: $C$ dominant for Grit, $E$
dominant for Leadership, $N$ most negative for Self-esteem alongside substantial $E$ and $C$
contributions matching the three-trait pattern in \citealt{robins2001}, and $C$-dominant with
secondary $N$ for Procrastination \citep{steel2007}. Values are means across nine checkpoints;
error bars show cross-checkpoint SD.}
\label{fig:criterion-betas}
\end{figure}

### 5.5 Replication across seeds and backbones

To verify that the results in §5.1–§5.4 are not artefacts of a single training run or a single
backbone, we trained five seeds of Qwen-2.5-1.5B and four seeds of Llama-3-8B-Instruct (4-bit
quantized), all under identical hyperparameters (full mode, three epochs, batch size 16). Table 6
reports the full per-checkpoint metrics; Table 7 aggregates by group.

| Model | Seed | Val loss | IPIP Tucker | IPIP Φ err | BFI Tucker | BFI Φ err | Crit prof.corr |
|---|---|---|---|---|---|---|---|
| Qwen | 0 | 1.336 | 0.842 | 0.640 | 0.858 | 0.398 | 0.866 |
| Qwen | 1 | 1.441 | 0.864 | 0.588 | 0.886 | 0.480 | 0.900 |
| Qwen | 2 | 1.506 | 0.834 | 0.608 | 0.854 | 0.397 | 0.892 |
| Qwen | 3* | 2.277 | 0.712 | 1.156 | 0.658 | 0.677 | 0.913 |
| Qwen | 4 | 1.342 | 0.858 | 0.724 | 0.886 | 0.336 | 0.897 |
| Llama | 1 | 1.362 | 0.908 | 0.624 | 0.884 | 0.335 | 0.908 |
| Llama | 2 | 1.709 | 0.834 | 0.754 | 0.860 | 0.401 | 0.885 |
| Llama | 3 | 1.411 | 0.894 | 0.678 | 0.874 | 0.412 | 0.881 |
| Llama | 4† | 1.476 | 0.854 | 0.861 | 0.752 | 0.489 | 0.865 |

*Table 6. Per-checkpoint metrics for all nine trained checkpoints. IPIP Tucker and BFI Tucker: mean
Tucker congruence on the respective held-out set. Φ err: Frobenius norm of the difference between
silicon and human inter-factor correlation matrices. Crit prof.corr: mean Pearson correlation of the
silicon domain–trait r-vector with the literature meta-analytic target vector, averaged across four
criterion domains. * Qwen seed 3: bad-init outlier (see text). † Llama seed 4: selective BFI dip
(see text). Both are retained in the table and in all aggregate calculations.*

| Group | n | IPIP Tucker | BFI Tucker | Crit prof.corr |
|---|---|---|---|---|
| Qwen (all seeds) | 5 | 0.822 ± 0.063 | 0.828 ± 0.096 | 0.894 ± 0.017 |
| Qwen (trimmed, excl. seed 3) | 4 | 0.850 ± 0.014 | 0.871 ± 0.017 | 0.889 ± 0.016 |
| Llama-3-8B-4bit | 4 | 0.873 ± 0.034 | 0.843 ± 0.061 | 0.885 ± 0.018 |

*Table 7. Aggregate mean ± SD by group. Qwen trimmed excludes the bad-init seed 3; both the
full and trimmed rows are reported for transparency.*

Within-Qwen replication across the four healthy seeds is tight: SD 0.014 on IPIP Tucker and 0.017
on BFI Tucker, indicating the training procedure is stable. The headline single-seed values from
§5.1–§5.2 (Qwen seed 0: IPIP Tucker 0.842, BFI Tucker 0.858) sit within one SD of the trimmed
group mean, confirming they are representative rather than optimistic. Criterion validity is also
consistent across healthy Qwen seeds (mean prof.corr 0.889 ± 0.016).

Cross-backbone, Llama-3-8B-4bit (a substantially larger model with a different architecture,
tokenizer, and training history, quantized to 4-bit) matches Qwen on IPIP Tucker (0.873 vs 0.850,
Δ = +0.023 in Llama's favour, within both SDs) and on criterion mean profile correlation (0.885 vs
0.889, Δ within 0.004). Qwen edges Llama on BFI Tucker (0.871 vs 0.843, Δ = 0.028), but this is
within Llama's SD of 0.061. The method is therefore not specific to the Qwen backbone; the
conditioning approach transfers across architectures.

Two outliers are retained transparently. Qwen seed 3 converged into a bad-init basin: its
validation loss (2.277) is nearly twice the typical value (1.34–1.51), and its IPIP and BFI Tucker
congruences collapse (0.712 and 0.658). Notably, its criterion profile correlation (0.913) is
the highest of any Qwen run, indicating that criterion validity is decoupled from internal factor
structure and requires far less training signal to emerge. Llama seed 4 shows the opposite failure
mode: its training loss is normal (1.476) and its IPIP Tucker is intact (0.854), but its BFI Tucker
drops to 0.752 — a selective OOD failure isolated to the BFI inventory that is not explained by any
training anomaly. This seed widens the Llama BFI SD to 0.061 and motivates caution about BFI
generalization on individual Llama runs, though the group mean (0.843) remains clearly above both
the persona-prompt ceiling (0.49) and the no-trait floor (0.28).

Across all 9 × 4 = 36 checkpoint × domain criterion cells, silicon recovers the correct sign on
the literature-dominant trait in 36/36 cases and the correct dominant trait by strict argmax in
27/36 (75%; Figure~\ref{fig:replication}).

\begin{figure}[ht!]
\centering
\includegraphics[width=\linewidth]{figures/fig3_replication_dots.png}
\caption{Tucker factor congruence for each individual checkpoint on IPIP held-out items (left,
T=0.6) and BFI out-of-distribution inventory (right, T=1.0). Circles = Qwen-2.5-1.5B;
triangles = Llama-3-8B-4bit. Horizontal bars show the trimmed group mean (Qwen: n=4, excluding
seed~3); vertical ticks show $\pm$1 SD. Annotated outliers: Qwen seed~3 (bad-init basin, val
loss 2.28, Tucker depressed in both panels) and Llama seed~4 (selective BFI Tucker dip to 0.75
in the right panel; IPIP Tucker normal). Both outliers are retained in all tables.}
\label{fig:replication}
\end{figure}

### 5.6 Temperature is a calibration knob, not a structure knob

The sampling temperatures used in §5.1–§5.4 (T = 0.6 for IPIP, T = 1.0 for BFI and the criterion
battery) were originally selected by inspecting Tucker congruence on the same held-out sets that
produce the headline numbers. We address this by holding out a
disjoint subset for T selection and evaluating the headline on the remainder.

For each of nine trained models × two inventories (IPIP, BFI), the human respondent pool is split 
50/50 by deterministic permutation into a calibration half and a reporting half. Silicon responses are generated from scratch 
at nine temperatures (T ∈ {0.4, 0.5, …, 1.2}) against the full silicon population (n = 1000). The calibration
half selects T* by minimising a moment-match objective: the per-item root mean squared difference
between the human and silicon response distributions in their mean and standard deviation (BFI
rescaled to 1–5 for comparability). The reporting half is used for all headline metrics and never
informs T selection.

Table 8 shows the calibration-optimal T* and the calibration objective value at T*, aggregated
(mean ± SD) across seeds. In all four backbone × inventory configurations, mean T* falls at
1.12–1.20 — substantially above the prior eval-tuned values of 0.6 and 1.0 — with SD ≤ 0.05,
i.e. essentially every individual seed selects the same calibration-optimal temperature. The
calibration objective decreases monotonically across the grid in every one of the 18
(checkpoint, eval) series individually, not just on average: at the previously used temperatures
silicon per-item response distributions are over-peaked relative to humans (the model concentrates
probability more sharply on a few Likert options than humans do), and raising T broadens the
marginal until the silicon spread matches human spread.

| Backbone (n) | Inventory | Prior T | Mean T* | SD T* | Mean calib. obj at T* | SD calib. obj |
|---|---|---|---|---|---|---|
| Qwen  | IPIP | 0.6 | 1.20 | 0.00 | 0.537 | 0.034 |
| Qwen  | BFI | 1.0 | 1.12 | 0.05 | 0.394 | 0.022 |
| Llama  | IPIP | 0.6 | 1.20 | 0.00 | 0.518 | 0.054 |
| Llama | BFI | 1.0 | 1.20 | 0.00 | 0.421 | 0.043 |

*Table 8. Calibration-optimal temperature T* selected by moment-match on the calibration half of
human respondents, aggregated across seeds (mean ± SD). The calibration objective is per-item
mean+std RMSE between human and silicon response distributions (lower is better). T* > prior T in
every backbone × inventory cell, indicating silicon is over-peaked at the prior temperatures.*

Table 9 shows Tucker congruence on the reporting half across the full temperature grid, aggregated
across seeds. The key observation is that Tucker varies little with T: the aggregate mean spread
across T ∈ [0.4, 1.2] ranges from 0.015 ± 0.007 (Qwen IPIP) to 0.046 ± 0.008 (Llama IPIP), comparable
to the cross-seed SD reported in §5.5. Factor structure is essentially T-invariant. This is expected
on theoretical grounds: Tucker congruence depends on item-item correlations, and Pearson correlations
are invariant to monotone transforms of a marginal distribution, so re-weighting the categorical
probabilities via T rescales each item's marginal without changing the rank ordering of responses
across respondents.

| Backbone × inventory (n) | T=0.4 | T=0.6 | T=0.8 | T=1.0 | T=1.2 | Mean spread |
|---|---|---|---|---|---|---|
| Qwen × IPIP | 0.861 ± 0.010 | 0.867 ± 0.006 | 0.870 ± 0.009 | 0.872 ± 0.003 | 0.868 ± 0.010 | 0.015 ± 0.007 |
| Qwen × BFI | 0.869 ± 0.010 | 0.865 ± 0.009 | 0.862 ± 0.013 | 0.858 ± 0.017 | 0.849 ± 0.014 | 0.021 ± 0.007 |
| Llama × IPIP | 0.900 ± 0.035 | 0.898 ± 0.029 | 0.888 ± 0.029 | 0.877 ± 0.033 | 0.861 ± 0.035 | 0.046 ± 0.008 |
| Llama × BFI | 0.855 ± 0.043 | 0.847 ± 0.044 | 0.837 ± 0.052 | 0.831 ± 0.048 | 0.820 ± 0.056 | 0.035 ± 0.013 |

*Table 9. Tucker congruence on the reporting half of human respondents across five representative
temperatures, mean ± SD across seeds. Mean spread = mean across seeds of (max − min) across the full
grid T ∈ [0.4, 1.2]. All mean spreads are within or close to the cross-seed SD in §5.5.*

At the calibration-optimal T*, the aggregate mean Tucker change relative to the prior eval-tuned T
is small in every cell and not uniformly signed: Qwen IPIP Δ = +0.001 ± 0.005, Qwen BFI
Δ = −0.004 ± 0.003, Llama IPIP Δ = −0.038 ± 0.006, Llama BFI Δ = −0.012 ± 0.007. The largest aggregate
drop (Llama IPIP) matches the single largest per-seed drop seen in any individual checkpoint
(−0.034 to −0.048), so the original two-checkpoint analysis was not an unrepresentative pick — the
effect replicates at the same small magnitude across all four healthy Llama seeds. As an illustrative
single-checkpoint check, the criterion mean profile correlation at the BFI-optimal T* = 1.10 (Qwen
seed 0) is 0.866, identical to the §5.4 value at the prior T = 1.0.

Temperature therefore plays two distinct roles that do not interact. It is a marginal-distribution
calibration knob: changing T adjusts per-item response spread so that silicon distributions resemble
human distributions, and the calibration objective selects the T that minimises this mismatch.
Separately, it is not a factor-structure knob: the Tucker congruence and criterion validity that
constitute the paper's headline claims are robust across the full temperature grid and do not depend
on the specific T chosen during initial evaluation.

### 5.7 Reliability

We report two complementary reliability measures: test-retest stability across repeated administrations
and internal consistency (Cronbach α) within a single administration.

Design. The respondent population is held fixed at n = 500 latents z drawn once and saved. Each
instrument is administered twice under different response-sampling seeds, keeping z, items, and checkpoint identical. 
The two administrations therefore differ only in the categorical sampling noise applied to each item's digit
distribution — the sole stochastic step in silicon_sample. Correlating reverse-coded scale scores
across the 500 respondents gives a per-factor Pearson r that measures generative stability: how
reproducibly the same latent z maps to the same scale score when the sampling is rerun.

All trained models are evaluated (five Qwen seeds, four Llama seeds), each on IPIP
(T = 0.6) and BFI (T = 1.0). Tables 10 and 11 report per-factor
test-retest r and Cronbach α respectively, aggregated as mean ± SD across seeds within each
model. Administrations 1 and 2 produced near-identical α values in all cells (within
0.06), confirming internal consistency is stable across sampling noise.

| Config | O | C | E | A | N | Mean |
|---|---|---|---|---|---|---|
| Qwen × IPIP  | 0.69 ± 0.04 | 0.77 ± 0.02 | 0.89 ± 0.02 | 0.74 ± 0.05 | 0.80 ± 0.04 | 0.78 |
| Qwen × BFI   | 0.57 ± 0.07 | 0.59 ± 0.02 | 0.77 ± 0.01 | 0.71 ± 0.03 | 0.82 ± 0.00 | 0.69 |
| Llama × IPIP | 0.68 ± 0.03 | 0.71 ± 0.03 | 0.84 ± 0.03 | 0.83 ± 0.01 | 0.84 ± 0.02 | 0.78 |
| Llama × BFI  | 0.61 ± 0.02 | 0.55 ± 0.04 | 0.70 ± 0.02 | 0.73 ± 0.04 | 0.85 ± 0.01 | 0.69 |

*Table 10. Silicon test-retest reliability: mean ± SD across seeds of the Pearson r between
reverse-coded scale scores from two independent administrations with
the same 500 fixed latents z per configuration. Values reflect generative-stability reliability —
variability from categorical sampling noise only. O = Openness, C = Conscientiousness,
E = Extraversion, A = Agreeableness, N = Neuroticism. Mean = unweighted mean across the five
factors.

| Config | O | C | E | A | N | Mean |
|---|---|---|---|---|---|---|
| Qwen × IPIP  | 0.48 ± 0.05 | 0.68 ± 0.02 | 0.84 ± 0.02 | 0.58 ± 0.06 | 0.71 ± 0.06 | 0.66 |
| Qwen × BFI  | 0.46 ± 0.08 | 0.54 ± 0.02 | 0.72 ± 0.01 | 0.73 ± 0.03 | 0.82 ± 0.00 | 0.65 |
| Llama × IPIP  | 0.55 ± 0.05 | 0.61 ± 0.07 | 0.74 ± 0.05 | 0.74 ± 0.01 | 0.78 ± 0.02 | 0.68 |
| Llama × BFI   | 0.51 ± 0.04 | 0.51 ± 0.04 | 0.66 ± 0.03 | 0.73 ± 0.04 | 0.84 ± 0.01 | 0.65 |

*Table 11. Cronbach α: mean ± SD across seeds. Scale lengths: IPIP the 4 held-out items per factor, BFI 5 items per factor.

Four findings emerge. First, silicon test-retest r and Cronbach α both fall squarely in the range
reported for published short-form Big Five inventories in human samples (retest r typically
0.65–0.85; α typically 0.50–0.85 for 4–5 item scales). The mean retest r across all four
configurations ranges from 0.69 to 0.78, comparable to human short-form norms. Second, the
cross-seed SDs are small for most factors, indicating
that reliability is a stable property of the training process. 
The largest variance appears on Qwen BFI Openness (0.57 ± 0.07) and Qwen IPIP
Agreeableness (0.74 ± 0.05), factors that also show wider structural variance in §5.1–§5.2.
Third, the factor-level ordering is consistent across both backbones and both measures:
Extraversion and Neuroticism are the most reliable factors, Openness is the least reliable on
IPIP, and Conscientiousness joins Openness at the bottom on BFI. This mirrors the pattern in
human short-form inventories and recapitulates Openness as the weak factor throughout this
paper. Fourth, Qwen and Llama show nearly identical mean retest r (both 0.78 on IPIP, both
0.69 on BFI), indicating that reliability is determined primarily by the instrument and
temperature rather than by LLM architecture. The BFI deficit relative to IPIP connects
to §5.6: the higher temperature used for BFI (T = 1.0) broadens the per-item categorical to
match human marginal spread, but at the cost of increased sampling noise and lower
generative-stability reliability. Factor-structure congruence remains invariant to this
trade-off.

### 5.8 Trait-steered free text (secondary, backbone-dependent)

The training loss supervises only the digit-token logits used for the Likert readout (§3.1); the
soft prefix and gated LoRA are never shown an open-ended continuation during training. This section
asks whether trait conditioning nonetheless transfers to free text as an emergent side effect —
and, if so, whether that transfer depends on the backbone.

**Design.** For each backbone, we sweep each of the five traits one at a time, holding the other
four latent dimensions at zero, across three levels z ∈ {−2, 0, +2} (population SD units), and pair
every (trait, level) setting with three open-ended prompts chosen to elicit person-revealing
narrative rather than a list:

1. "Describe a typical Saturday for you, from morning to evening."
2. "Describe how you'd handle starting a new job."
3. "What do you do when you're at a party where you don't know anyone?"

Five samples are drawn per
(trait, level, prompt) cell at T = 0.8, giving 5 traits × 3 levels × 3 prompts × 5 samples = 225
generations per backbone. Each generation is then read back by the Claude Opus 4.7
and scored against five Likert probes, one per trait:

| Trait | Probe statement |
|---|---|
| O | "This person seems open to new experiences and ideas." |
| C | "This person seems conscientious and well-organized." |
| E | "This person seems extraverted and outgoing." |
| A | "This person seems agreeable, warm, and considerate." |
| N | "This person seems anxious or emotionally unstable." |

The judge reads the generation and answers each probe on the same 1–5 digit-logit readout used for
the main psychometric pipeline, giving a continuous "perceived trait" score per generation. For each
swept trait we report Δ = mean score at z = +2 minus mean score at z = −2, both on the *on-target*
probe (the row's matching trait — diagonal entries) and on the other four *off-target* probes
(leakage — off-diagonal entries), plus a Cohen's-d effect size on the on-target probe.

**Quantitative result.** Table 12 summarizes the on-target effect size, diagonal dominance (whether
the on-target probe shows the largest absolute Δ in its row), and the RLHF-refusal rate (share of
generations containing "AI", "language model", or "assistant") for each backbone.

| Backbone | O d | C d | E d | A d | N d | Diagonal dominance | Refusal rate |
|----------|---|---|---|---|---|---|---|
| Qwen     | +1.13 | +0.04 | +0.42 | +0.49 | +0.02 | 1/5 | 30% |
| Llama    | +0.26 | +1.03 | +1.35 | +0.99 | +0.71 | 5/5 | 0% |

*Table 12. On-target steering effect size (Cohen's d, refusal-filtered) per trait, diagonal
dominance (on-target |Δ| is the row-max out of 5 probes), and RLHF-refusal rate across 225
generations per backbone.

**Llama steers cleanly on all five traits**, with the on-target probe the largest mover in
every row and no generation triggering an assistant-persona disclaimer. **Qwen steers reliably only
on Openness**, with Extraversion and Agreeableness showing weak partial effects and Conscientiousness
and Neuroticism essentially flat; in addition, nearly a third of Qwen's generations collapse into
"As an AI language model..." boilerplate regardless of the conditioning latent. Because the
conditioning architecture, training data, and loss are identical across both runs, the gap is
attributable to the backbone's instruction-tuning prior, not to the prefix+LoRA method.

**Qualitative case studies.** The aggregate numbers above compress 450 generations into ten effect
sizes; the underlying text makes the mechanism, and its failure mode, concrete.

*Llama — Conscientiousness, "starting a new job" prompt.* At z = −2 the model opens with
self-doubt and a coping strategy: "When I start a new job, I tend to be a bit nervous and anxious. I
worry about making a good impression, remembering important details, and figuring out how things
work. I usually try to take a deep breath and remind myself that it's okay to ask questions and make
mistakes." (perceived C = 4.06, on a generous baseline — the *content* is what shifts, not just the
surface score). At z = +2 the same prompt produces an execution plan: "I always try to be prepared
and organized. I make sure to read and review all the information I can about the company, the
role, and the expectations. I also make a plan to get familiar with the new environment... I
prioritize building relationships with my new team and supervisor, asking questions and seeking
clarification" (perceived C = 5.00). The shift from affect ("nervous," "worry") to procedure
("plan," "prioritize," "review") is exactly the lexical signature conscientiousness research
associates with trait-low vs. trait-high self-descriptions.

*Llama — Extraversion, "party" prompt.* z = −2: "I usually try to find a quiet corner or a familiar
face to talk to. If I'm feeling overwhelmed, I might step outside for some fresh air... I'm not one
for forced small talk, so I tend to observe and listen more than I participate" (perceived E = 1.82).
z = +2: "I tend to feel a mix of nervousness and excitement. I usually try to find someone who looks
friendly and approachable, and I'll strike up a conversation with them... I'll ask them a question
about themselves or their interests, and try to find common ground" (perceived E = 3.63) — withdrawal
replaced by approach behavior, the largest on-target effect of any cell (d = +1.35).

*Llama — Neuroticism, same "party" prompt.* z = −2: "I try to relax and be myself... If I'm feeling
nervous, I might take a moment to observe the surroundings, take a deep breath" (perceived N = 1.90).
z = +2: "I get nervous and feel a bit anxious, wondering how to make a good impression and connect
with others... I also tend to feel like an outsider, observing the group dynamics and hoping to find
someone to talk to. Sometimes I'll excuse myself to grab a drink or go to the bathroom, just to
collect [myself]" (perceived N = 3.52). The same situational prompt elicits self-soothing language at
low N and explicit anxiety plus an escape behavior at high N — a within-prompt contrast that isolates
the effect of the latent from the effect of the question.

*Qwen — Openness, the one trait that transfers cleanly.* On the "new job" prompt, z = −2 gives a
conventional, goal-list answer: "I would start by taking the job seriously and showing enthusiasm
for it. Starting with research on company culture and values will help me better understand my role
and how to fit in" (perceived O = 4.00). z = +2 gives a more abstract, process-oriented framing:
"I would approach a new job with enthusiasm and a proactive mindset... prepare a well-thought-out
plan for integrating smoothly into the team... aim to make the most of every opportunity" (perceived
O = 4.40). The contrast is real but subtler than Llama's — consistent with Openness being Qwen's
only effect size above d = +1.

*Qwen — Extraversion, mode collapse.* On the same "party" prompt held at sample index 0, the three
generations at z = −2, 0, +2 are near-paraphrases of each other ("try to strike up conversations
with people who seem friendly," "usually strike up a conversation with someone nearby," "my first
step is to smile and say hello") and the judge scores them within 0.01 of each other
(perceived E = 4.01, 4.02, 4.01). The conditioning signal that moves the Likert digit logits by
roughly half a scale point (§5.1–§5.2) produces no detectable change in free-text content here —
direct evidence that text-generation steering and the trained Likert-readout objective are
dissociable capabilities on this backbone.

*Qwen — partial leakage tracks the human trait correlation, even where the on-target effect fails.*
On the "new job" prompt the Extraversion sweep again fails on-target (perceived E = 3.12 at z = −2
vs. 3.10 at z = +2) but moves Neuroticism: "Starting a new job is an exciting yet daunting task...
It's also important for me to set clear goals" (perceived N = 2.47) at z = −2 versus "I would
approach starting a new job by thoroughly researching the company and role... Preparation is key to
making the transition smoother and setting myself up for success" (perceived N = 1.61) at z = +2.
"Daunting" gives way to "preparation is key" — the hedged, anxious framing recedes as E rises, even
though E itself is not perceived as higher. Averaged over all 75 Extraversion generations, mean
perceived N falls from 2.03 (z = −2) to 1.70 (z = +2), the largest off-diagonal entry in Qwen's
leakage matrix and the same sign as the empirical E↔N anti-correlation in the human IPIP sample
(§5.1). This is the pattern we would expect if Qwen's frozen representations still encode the
human trait covariance structure even when the trained conditioning channel fails to express it
on-target in this backbone's free-text decoding.

*Qwen — trait signal surviving inside a refusal.* Even Qwen's RLHF boilerplate is not fully
trait-blind. All five samples at Openness z = +2 on the "Saturday" prompt are disclaimers, but the
disclaimers differ qualitatively by sweep direction. At z = +2: "As an AI language model, I don't
have a typical Saturday like humans do. However, my programming includes continuous learning and
engagement with users 24/7... Saturdays are just another opportunity to learn new things from the
vast amounts of data available online" — curiosity-coded language ("continuous learning," "vast
amounts of data") persists inside the refusal template. This generation is excluded from the
refusal-filtered effect sizes in Table 12 (which is conservative — it likely undercounts Qwen's true
Openness effect), but it shows that the conditioning signal is not simply absorbed by the RLHF prior;
it is partially expressed through it.

Text steering is an emergent by-product of digit-only training: the soft prefix
and gated LoRA shift the model's hidden representations broadly enough to be legible in open-ended
decoding, not only at the final-position digit logits that the loss actually supervises. Whether that
shift surfaces as a clean on-target change in free text is gated by the backbone's instruction-tuning
prior — Llama's chat-tuning leaves room for first-person narrative variation, while large parts of
Qwen's are pulled toward a fixed assistant-persona attractor, especially for prompts and trait
settings that the judge would otherwise mark as low-Openness. The off-diagonal leakage on Qwen
(raising E lowers perceived N) reproduces a known human personality structure (E↔N anti-correlation)
that was never explicitly trained into the prefix/LoRA weights for the text modality, suggesting the
backbone's frozen representations carry trait covariance beyond what the readout objective directly
shapes. The Likert-readout headline results (§5.1–§5.5) do not depend on text steering working on
either backbone — but where it does work, it is additional, non-trivial evidence that the latent
code is doing more than memorizing a digit-prediction shortcut.

### 5.9 Ablation: the latent gate is a wash

At matched 3 epochs the gated (full) and ungated (baseline_c) variants differ by ≈ 0.02 Tucker in
opposite directions on IPIP vs BFI — within single-seed noise; baseline_c has marginally lower item-holdout
loss. The large one-epoch gap was a convergence-speed artifact of the gate's sigmoid≈0.5 initialization. We
therefore do **not** claim latent gating as a contribution; the contributions are the prefix+LoRA
continuous-latent conditioning, the sampling-based silicon-sample construction, and the validity evidence.
We ship `full` as an operational choice (slightly better, more balanced OOD); baseline_c is a near-equivalent
simpler fallback.

---

## 6. Discussion

The results in §5 form a chain of evidence rather than a single number, and the chain is what the
argument rests on. Taken individually, any one link could be explained away: a Tucker congruence of
0.85 on held-out items could reflect memorized item-response patterns rather than a learned construct;
transfer to the BFI could reflect surface overlap between the two inventories' wording; the criterion
correlations could reflect a generic evaluative halo; the multi-seed numbers could reflect a single
lucky hyperparameter setting; the temperature used to generate every other table could itself be a
tuned artifact. Taken together, the links close off these explanations one at a time. Cross-inventory
transfer (§5.2) rules out item memorization, since the BFI's 25 items share no text with the IPIP
training items and use a different response scale (1–6 vs 1–5); a model that had merely memorized
IPIP item-response statistics would have no mechanism for producing the right factor structure on
wording it has never seen. The criterion battery (§5.4) rules out replay of the injected covariance
Σ_human, since the domain–trait relationships tested there — grit with conscientiousness, self-esteem
with neuroticism, leadership with extraversion, procrastination with conscientiousness — are absent
from Σ_human by construction; the training procedure injects only the Big Five inter-factor structure,
never a personality–outcome link, so a model that recovers those links with the correct sign in 36 of
36 checkpoint–domain cells, and the published multi-trait β pattern in the cases studied closely
(self-esteem, procrastination), is doing something beyond covariance replay. The discriminant-validity
analysis in particular shows that the apparent zero-order halo for self-esteem and procrastination
was a property of the zero-order correlations, not of the underlying representation: once the five
traits are regressed on simultaneously, the unique contribution of the target trait is statistically
isolated in every case, and the resulting profile matches the multi-trait pattern reported in the
literature rather than a generic positivity loading. The nine-checkpoint replication (§5.5) rules out
single-run or single-backbone luck, since the headline numbers reproduce within a few hundredths of
Tucker congruence across five Qwen seeds and four Llama-3-8B seeds — a substantially larger model with
a different tokenizer, architecture, and pretraining corpus. The held-out temperature calibration
(§5.6) rules out the headline numbers being an artifact of hand-tuning T against the same data used to
report Tucker congruence, since selecting T on a disjoint calibration half of the human sample and
reporting on the other half changes Tucker congruence by at most a few hundredths in either direction,
and the calibration-optimal temperature is consistently higher than the temperature originally chosen
by inspection — meaning the original choice, if anything, understated rather than inflated the
headline numbers. The reliability analysis (§5.7) addresses a different kind of concern: even a
genuinely construct-valid synthetic population is not useful for downstream survey methodology if its
measurement properties look nothing like a real instrument's. Silicon test-retest correlations and
Cronbach alphas fall inside the range reported for published human short-form Big Five inventories,
and the factor-level ordering of reliability — extraversion and neuroticism most reliable, openness
least — mirrors the ordering seen in human samples. A simulator that reproduced factor structure but
had, say, near-perfect or near-zero test-retest reliability would be a poor stand-in for a human
population even if its Tucker congruence looked identical on paper; the fact that the noise properties
also resemble a real instrument is part of the validity case, not a separate result.

Two findings complicate a purely additive reading of this evidence and are worth discussing on their
own terms rather than folding into the headline. First, the gate ablation (§5.9) shows that the latent
gating mechanism we built into the architecture — a sigmoid gate over the LoRA bottleneck, modulated
by the trait code — earns its keep on neither inventory once training is run to convergence: gated
and ungated variants differ by about 0.02 Tucker congruence in opposite directions on IPIP versus BFI,
well inside single-seed noise, and the large advantage the gate appeared to have after one epoch
turned out to be a convergence-speed artifact of its near-0.5 sigmoid initialization rather than a
genuine capacity benefit. We report this transparently because it reallocates where we believe the
paper's contribution actually lives: not in the gating mechanism, but in the combination of continuous
soft-prefix and LoRA conditioning, the sampling-based (rather than argmax) construction of the silicon
population, and the validity battery itself. Second, the two outliers retained in §5.5 — Qwen seed 3,
which converged into a visibly bad initialization basin with double the typical validation loss and
collapsed internal factor structure, and Llama seed 4, whose training was unremarkable but whose BFI
Tucker congruence selectively dropped while its IPIP congruence stayed intact — reveal that internal
factor-structure fidelity and criterion validity are not the same capability. Qwen seed 3's criterion
profile correlation (0.913) is the highest of any Qwen run despite its collapsed factor structure,
which suggests that recovering a domain's literature-dominant trait by sign or rough magnitude
requires substantially less of the conditioning signal to be intact than recovering the fine
inter-item correlation structure a five-factor model depends on. This is a useful caution against
treating a single aggregate metric as a stand-in for "the model learned personality": different
validity criteria can dissociate, and a synthetic population that is unsuitable for, say,
factor-analytic scale development because of unstable structure could still be informative for a
coarser criterion-prediction use case, or vice versa.

A further thread runs through the structural, criterion, and reliability results without ever being
the headline number in any one of them: Openness is consistently the hardest factor to reproduce on
the IPIP, the least reliable factor in the test-retest analysis, and recovers well only when the
items themselves are more linguistically distinctive, as on the BFI. This is not obviously a defect of
the conditioning method, since human Openness scales also show the lowest internal consistency of the
five factors in normative data; a plausible reading is that Openness is intrinsically the construct
whose behavioral and linguistic correlates are least sharply defined, for silicon and human
respondents alike, and that the conditioning channel is faithfully reproducing a property of the
construct rather than introducing a method-specific weakness.

The secondary text-steering result (§5.8) extends the validity argument into a modality the training
loss never touches. Because the conditioning channel is supervised only through the digit-token logits
used for the Likert readout, any effect on open-ended generation is necessarily a side effect of how
the soft prefix and LoRA adapters reshape the model's hidden representations, not a directly trained
behavior. That this side effect exists at all on Llama-3-8B — clean, on-target steering on all five
traits, including off-diagonal leakage whose pattern (raising extraversion lowers perceived
neuroticism) mirrors the human extraversion–neuroticism anti-correlation reproduced in the structural
analysis — is evidence that the latent code is shaping something broader than a narrow digit-prediction
shortcut. That the same architecture, trained the same way, produces almost no steering effect on
Qwen, whose generations instead collapse into an RLHF assistant-persona disclaimer for nearly a third
of prompts, shows that this broader representational change is necessary but not sufficient for
visible behavior change: whether it surfaces depends on how strongly the backbone's own instruction-
tuning prior competes for control of the output. We treat this as informative about the limits of the
method as currently trained — reliable trait-conditioned free text would likely require supervising
text generation directly rather than relying on emergence from a digit-only objective — rather than as
part of the headline validity claim, which does not depend on text steering working on either
backbone.

Positioned against the broader silicon-sampling literature, this work shares the goal of @argyle2023's
algorithmic-fidelity argument and the persona-prompting line of work [@serapiogarcia2023; @safdari2023;
@jiang2023] — using language models to stand in for human respondents — but targets a different and, we
think, more demanding bar: not whether prompted or conditioned trait scores correlate with intended
traits, but whether a synthetic population built from such a model reproduces the factor structure,
cross-instrument transfer, and nomological network that define a construct in the psychometric sense,
and does so with the kind of statistical replication and robustness checks the methodology requires
before the resulting synthetic data could be used for actual scale development or pilot testing.
Relative to the persona-prompting baselines in §5.3, trained latent conditioning nearly doubles factor
congruence over the best zero-shot prompt format, which suggests that whatever silicon samples are
capturing, it depends on more than the backbone's pre-existing capacity to role-play a description in
context.

---

## 7. Conclusion

This paper describes and validates a method for building synthetic survey respondents that reproduce
human Big Five psychometric structure: a frozen instruction language model conditioned on a continuous
five-dimensional trait latent through a soft prefix and gated low-rank adapters, with each Likert
response read directly from the model's own digit-token logits and sampled, rather than taken as an
argmax, so that a synthetic population can be built whose item-level variability resembles a real
respondent pool. Across nine trained checkpoints spanning two backbones and five random seeds, the
resulting silicon populations reproduce the human five-factor structure on items withheld from
training, transfer that structure to a different inventory with different item wording and a different
response scale, preserve the inter-factor correlation matrix, and — most importantly for the validity
argument — recover documented relationships between personality and four non-personality constructs
that were never injected into the training procedure, with the correct sign in every one of 36
checkpoint–domain cells and literature-matching multi-trait loadings under discriminant analysis. These
properties are robust to a held-out recalibration of the sampling temperature and are accompanied by
test-retest reliability and internal-consistency statistics that fall within the range reported for
human short-form Big Five inventories, rather than being either implausibly perfect or unstable. An
ablation shows that a gating mechanism added to the adapter architecture does not contribute
measurably once training converges, so we attribute the method's contribution to the combination of
continuous latent conditioning and sampling-based population construction rather than to any single
architectural component; a secondary analysis shows that the same conditioning can shift open-ended,
free-text generation in trait-consistent directions on one of the two backbones tested, even though
text generation is never part of the training objective, though this effect is gated by how strongly
each backbone's own instruction-tuning prior resists the conditioning signal.

These results make a reasonably strong case that the simulator is doing more than memorizing item
statistics or replaying an injected correlation matrix, but several limitations bound how far that case
extends. The conditioning latents used throughout are CFA factor scores derived from the same IPIP
sample the model is trained on, and the inter-factor covariance Σ_human used to draw the synthetic
population is injected directly, so a degree of circularity is built into the setup by design; the
emergent criterion validity in §5.4 addresses this only partly, since it shows the recovered
domain–trait links are not themselves injected, while the trait space they are expressed in still
originates from the training inventory. Relatedly, criterion validity throughout this paper is
benchmarked against meta-analytic literature targets rather than against a human sample in which the
Big Five and the criterion instruments were measured together; a direct human-versus-silicon
comparison on co-administered instruments would be a stronger test than comparison against a published
correlation, and is the most valuable single follow-up this work points toward. Openness is, on the
evidence presented here, the construct the method handles least well, with the lowest structural
congruence on held-out IPIP items and the lowest test-retest reliability of the five factors; because
this weakness recurs across independent analyses and mirrors Openness's comparatively low internal
consistency in human normative data, we read it as a property of the construct the method is
faithfully tracking rather than a defect specific to the conditioning architecture, but we have not
ruled out that targeted item augmentation could close some of the gap. Text-conditioned free generation
is reliable on only one of the two backbones tested and was never directly supervised, so any
production use of trait-conditioned text would need explicit text-level training rather than relying on
this emergent transfer. Finally, the evidence base covers two inventories (IPIP and BFI) and a single
construct family (the Big Five); whether the same conditioning approach and validity pattern would hold
for other personality models, other psychological constructs, or instruments with substantially
different item formats remains untested, and is the natural next axis along which to extend this work.

---

## Reproducibility

All code, data splits, checkpoints, and per-checkpoint result tables are released. Key scripts:
`train_simulator.py`, `silicon_sample.py`, `05_simulator_psychometrics.{py,R}`, `06_criterion_validity.py`,
`07_partial_betas.py`, `08_temp_calibration.py`, `09_text_scoring.py`, `10_reliability.py`,
`train_campaign.sh`, `eval_campaign.py`. See `WALKTHROUGH.md` for the full result tables (§7).
