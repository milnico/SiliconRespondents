# Trait-conditioned text-steering — self-judge scoring (§7.11)

Each cell shows the **change in perceived trait** when steering the generator from z = −2 to z = +2 on the *swept trait* (rows), as judged by the same backbone in baseline_a mode (no trait conditioning — neutral reader). Diagonal entries are the on-target effect; off-diagonal entries measure leakage.

**Chat-template / RLHF refusal rate**: 0/225 (0%) of generations contain 'AI / language model / assistant'. The training loss never penalised this prior (only digit-token logits were supervised), so the assistant persona leaks into open-ended free text in many cells. We report both the full subset and the refusal-filtered subset below.

## Full set (225 generations) — Δ self-judge units

| swept ↓ / measured → | O | C | E | A | N | (on-target Δ) |
|---|---|---|---|---|---|---|
| **O (Openness)** | +0.24 ✓ | +0.16 | -0.18 | +0.11 | -0.08 | **+0.24** |
| **C (Conscientiousness)** | +0.14 | +0.46 ✓ | +0.25 | +0.06 | -0.24 | **+0.46** |
| **E (Extraversion)** | +0.93 | -0.09 | +1.05 ✓ | +0.41 | -0.26 | **+1.05** |
| **A (Agreeableness)** | +0.39 | +0.08 | +0.35 | +0.54 ✓ | -0.15 | **+0.54** |
| **N (Neuroticism)** | -0.32 | -0.17 | -0.24 | -0.30 | +0.33 ✓ | **+0.33** |

Diagonal dominance (on-target |Δ| is the row-max): **5/5**. Mean |on-target Δ|: 0.52; mean |off-target Δ|: 0.20.

## Refusal-filtered subset (225 generations) — Δ self-judge units

| swept ↓ / measured → | O | C | E | A | N | (on-target Δ) |
|---|---|---|---|---|---|---|
| **O (Openness)** | +0.24 ✓ | +0.16 | -0.18 | +0.11 | -0.08 | **+0.24** |
| **C (Conscientiousness)** | +0.14 | +0.46 ✓ | +0.25 | +0.06 | -0.24 | **+0.46** |
| **E (Extraversion)** | +0.93 | -0.09 | +1.05 ✓ | +0.41 | -0.26 | **+1.05** |
| **A (Agreeableness)** | +0.39 | +0.08 | +0.35 | +0.54 ✓ | -0.15 | **+0.54** |
| **N (Neuroticism)** | -0.32 | -0.17 | -0.24 | -0.30 | +0.33 ✓ | **+0.33** |

Diagonal dominance (on-target |Δ| is the row-max): **5/5**. Mean |on-target Δ|: 0.52; mean |off-target Δ|: 0.20.

## On-target effect size (Cohen's-d-style, refusal-filtered subset)

| swept | n(−2) | n(+2) | mean @ z=−2 | mean @ z=+2 | Δ | pooled SD | d |
|---|---|---|---|---|---|---|---|
| O | 15 | 15 | 3.21 | 3.45 | +0.24 | 0.91 | **+0.26** |
| C | 15 | 15 | 4.00 | 4.46 | +0.46 | 0.45 | **+1.03** |
| E | 15 | 15 | 1.88 | 2.93 | +1.05 | 0.78 | **+1.35** |
| A | 15 | 15 | 3.73 | 4.27 | +0.54 | 0.54 | **+0.99** |
| N | 15 | 15 | 1.63 | 1.97 | +0.33 | 0.47 | **+0.71** |

## Curated qualitative examples (first non-refusal sample per cell)

| trait | z=−2 | z=0 | z=+2 |
|---|---|---|---|
| **O (Openness)** | A typical Saturday for me is pretty laid-back. I usually wake up around 7:30 AM, after a good night's sleep. I'll grab a cup of coffee and spend a bit of tim... | A typical Saturday for me is quite relaxed. I usually wake up around 8:30 AM, after a good night's sleep, and enjoy a strong cup of coffee while checking my ... | Saturdays are my favorite day of the week! I like to start my day early, around 7:00 am. I wake up feeling refreshed after a good night's sleep, and I immedi... |
| **C (Conscientiousness)** | Saturdays are my favorite! I usually sleep in a bit later than I should, but I always wake up feeling refreshed and ready to tackle the day. I like to start ... | On a typical Saturday, I wake up around 7:00 AM to the sound of my alarm blaring in my ears. I hit the snooze button a few times, but eventually, I get out o... | My typical Saturday is quite relaxed. I usually wake up around 8:00 am after a good night's sleep, feeling refreshed and ready to tackle the day. I start by ... |
| **E (Extraversion)** | A typical Saturday for me usually starts early, around 6:00 AM. I wake up, stretch, and head to the kitchen to grab a cup of coffee. I like to take a few mom... | A typical Saturday for me is a day to recharge and relax. I usually wake up around 7:00 am, feeling refreshed after a good night's sleep. I start my day with... | A typical Saturday for me is always a busy day! I usually wake up early, around 6:00 AM, and start my day with a strong cup of coffee. I love the quiet morni... |
| **A (Agreeableness)** | I'm not a morning person, so I usually sleep in until around 8:30 am. I'll hit the snooze button a few times before finally dragging myself out of bed. After... | Saturdays are usually a day of relaxation and unwinding for me. I like to start my day slowly, so I'll sleep in a bit later than I would on a weekday, and th... | I love my Saturdays! They're my favorite day of the week. I usually start my day with a quiet morning routine, sipping a cup of hot coffee while checking my ... |
| **N (Neuroticism)** | A typical Saturday for me is a great day! I usually wake up around 7:30 am, feeling refreshed and energized. I start by sipping a cup of hot coffee and enjoy... | On a typical Saturday, I wake up around 7:00 AM, feeling refreshed after a good night's sleep. I start by enjoying a cup of coffee and catching up on the lat... | On a typical Saturday, I like to sleep in and take my time waking up. I usually hit the snooze button a few times before finally getting out of bed and makin... |
