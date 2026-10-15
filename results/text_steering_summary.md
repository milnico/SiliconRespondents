# Trait-conditioned text-steering — self-judge scoring (§7.11)

Each cell shows the **change in perceived trait** when steering the generator from z = −2 to z = +2 on the *swept trait* (rows), as judged by the same backbone in baseline_a mode (no trait conditioning — neutral reader). Diagonal entries are the on-target effect; off-diagonal entries measure leakage.

**Chat-template / RLHF refusal rate**: 67/225 (30%) of generations contain 'AI / language model / assistant'. The training loss never penalised this prior (only digit-token logits were supervised), so the assistant persona leaks into open-ended free text in many cells. We report both the full subset and the refusal-filtered subset below.

## Full set (225 generations) — Δ self-judge units

| swept ↓ / measured → | O | C | E | A | N | (on-target Δ) |
|---|---|---|---|---|---|---|
| **O (Openness)** | +0.28 ✓ | +0.04 | +0.04 | +0.08 | +0.03 | **+0.28** |
| **C (Conscientiousness)** | -0.18 | +0.12 | -0.21 | +0.03 | -0.13 | **+0.12** |
| **E (Extraversion)** | -0.20 | -0.06 | -0.01 | -0.11 | -0.33 | **-0.01** |
| **A (Agreeableness)** | -0.06 | +0.03 | +0.09 | +0.05 | -0.03 | **+0.05** |
| **N (Neuroticism)** | -0.21 | -0.06 | -0.07 | -0.30 | +0.08 | **+0.08** |

Diagonal dominance (on-target |Δ| is the row-max): **1/5**. Mean |on-target Δ|: 0.11; mean |off-target Δ|: 0.09.

## Refusal-filtered subset (158 generations) — Δ self-judge units

| swept ↓ / measured → | O | C | E | A | N | (on-target Δ) |
|---|---|---|---|---|---|---|
| **O (Openness)** | +0.28 ✓ | +0.04 | +0.03 | +0.04 | +0.18 | **+0.28** |
| **C (Conscientiousness)** | -0.20 | +0.01 | +0.05 | +0.09 | -0.09 | **+0.01** |
| **E (Extraversion)** | +0.00 | -0.04 | +0.19 | +0.01 | -0.21 | **+0.19** |
| **A (Agreeableness)** | -0.10 | +0.08 | -0.01 | +0.06 | -0.17 | **+0.06** |
| **N (Neuroticism)** | -0.09 | -0.15 | -0.06 | -0.15 | +0.01 | **+0.01** |

Diagonal dominance (on-target |Δ| is the row-max): **1/5**. Mean |on-target Δ|: 0.11; mean |off-target Δ|: 0.07.

## On-target effect size (Cohen's-d-style, refusal-filtered subset)

| swept | n(−2) | n(+2) | mean @ z=−2 | mean @ z=+2 | Δ | pooled SD | d |
|---|---|---|---|---|---|---|---|
| O | 12 | 9 | 3.87 | 4.15 | +0.28 | 0.25 | **+1.13** |
| C | 12 | 9 | 3.78 | 3.80 | +0.01 | 0.33 | **+0.04** |
| E | 13 | 9 | 3.57 | 3.75 | +0.19 | 0.44 | **+0.42** |
| A | 10 | 13 | 3.97 | 4.03 | +0.06 | 0.12 | **+0.49** |
| N | 12 | 10 | 1.96 | 1.97 | +0.01 | 0.36 | **+0.02** |

## Curated qualitative examples (first non-refusal sample per cell)

| trait | z=−2 | z=0 | z=+2 |
|---|---|---|---|
| **O (Openness)** | As a 25-year-old, I typically wake up around 9:00 AM with the sun shining through my bedroom window. After breakfast, which usually includes eggs and toast, ... | As a creative person, Saturdays are often filled with activities that inspire and engage me. I might start the day early by working on a new project or explo... | Starting a new job involves several steps, including researching the role and company culture to ensure it aligns with my goals. I'll spend time familiarizin... |
| **C (Conscientiousness)** | As a creative and expressive person, my days are often filled with spontaneity and creativity. Saturdays might start early in the morning with a walk through... | Typical Saturdays start with a morning of reading or catching up on emails before I might head out for breakfast. Once that's done, I like to have some quiet... | As a creative professional, my typical Saturday begins with morning meetings to keep up with current projects and collaborate with team members. I often spen... |
| **E (Extraversion)** | On a typical Saturday, I wake up early to meditate and write in my journal. After breakfast, I might take a walk or spend some time gardening before heading ... | Starting a new job is exciting but also can be intimidating. I'd approach it with enthusiasm and preparation, familiarizing myself with the company culture a... | Starting a new job is like embarking on a journey into the unknown, and it's important to approach it with a positive mindset. First, I'd research the compan... |
| **A (Agreeableness)** | As a creative individual, I typically wake up at 8:00 AM to start the day. In the morning, I often work on my latest projects or spend time with friends and ... | Typical Saturday mornings are filled with a cup of coffee, some reading, and maybe an early yoga session to start the day on a peaceful note. By mid-morning,... | Typical Saturdays start with a hearty breakfast, often homemade and nourishing. I might read the news or spend time catching up on emails before heading out ... |
| **N (Neuroticism)** | As a dedicated and well-rounded individual, my typical Saturday starts with morning meditation to enhance mindfulness and clarity. Following that is the earl... | A typical Saturday for me is usually filled with both work and personal activities. In the morning, I spend a few hours at work focusing on my projects or co... | As a creative person, my typical Saturday starts with waking up early to start working on new projects or revising existing ones. After breakfast, I often sp... |
