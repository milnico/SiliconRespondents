# Held-out temperature calibration (§5.6)

Per (checkpoint, eval), humans are split 50/50 by deterministic permutation (seed 42) into a **calibration** half and a **reporting** half. The calibration half picks T\* by minimising a per-item moment-match objective (mean+std RMSE between human and silicon response distributions, BFI rescaled to 1–5). The reporting half is held out for the headline psychometrics. Aggregates use Qwen trimmed n=4 (excluding seed 3, bad-init) and Llama n=4.

## Aggregate calibration-optimal T\* (mean ± SD across seeds)

| Backbone | Inventory | Prior T | Mean T\* | SD T\* | Mean calib obj | SD calib obj |
|---|---|---|---|---|---|---|
| Qwen (n=4 trim) | IPIP | 0.6 | 1.20 | 0.00 | 0.537 | 0.034 |
| Qwen (n=4 trim) | BFI | 1.0 | 1.12 | 0.05 | 0.394 | 0.022 |
| Llama (n=4) | IPIP | 0.6 | 1.20 | 0.00 | 0.518 | 0.054 |
| Llama (n=4) | BFI | 1.0 | 1.20 | 0.00 | 0.421 | 0.043 |

## Aggregate Tucker sensitivity (mean ± SD, rep-half humans)

| Backbone × inventory | T=0.4 | T=0.6 | T=0.8 | T=1.0 | T=1.2 | Mean spread |
|---|---|---|---|---|---|---|
| Qwen (n=4 trim) × IPIP | 0.861 ± 0.010 | 0.867 ± 0.006 | 0.870 ± 0.009 | 0.872 ± 0.003 | 0.868 ± 0.010 | 0.015 ± 0.007 |
| Qwen (n=4 trim) × BFI | 0.869 ± 0.010 | 0.865 ± 0.009 | 0.862 ± 0.013 | 0.858 ± 0.017 | 0.849 ± 0.014 | 0.021 ± 0.007 |
| Llama (n=4) × IPIP | 0.900 ± 0.035 | 0.898 ± 0.029 | 0.888 ± 0.029 | 0.877 ± 0.033 | 0.861 ± 0.035 | 0.046 ± 0.008 |
| Llama (n=4) × BFI | 0.855 ± 0.043 | 0.847 ± 0.044 | 0.837 ± 0.052 | 0.831 ± 0.048 | 0.820 ± 0.056 | 0.035 ± 0.013 |

## Per-seed detail

| ckpt | eval | T\* | calib obj | Tucker @ prior T | Tucker @ T\* | Δ |
|---|---|---|---|---|---|---|
| qwen/seed0 | ipip | 1.20 | 0.517 | 0.868 (T=0.6) | 0.864 | -0.004 |
| qwen/seed0 | bfi | 1.10 | 0.403 | 0.834 (T=1.0) | 0.834 | +0.000 |
| qwen/seed1 | ipip | 1.20 | 0.530 | 0.872 (T=0.6) | 0.872 | +0.000 |
| qwen/seed1 | bfi | 1.10 | 0.376 | 0.858 (T=1.0) | 0.852 | -0.006 |
| qwen/seed2 | ipip | 1.20 | 0.587 | 0.858 (T=0.6) | 0.856 | -0.002 |
| qwen/seed2 | bfi | 1.20 | 0.376 | 0.866 (T=1.0) | 0.858 | -0.008 |
| qwen/seed3 | ipip | 1.20 | 0.652 | 0.570 (T=0.6) | 0.636 | +0.066 |
| qwen/seed3 | bfi | 1.10 | 0.429 | 0.712 (T=1.0) | 0.706 | -0.006 |
| qwen/seed4 | ipip | 1.20 | 0.515 | 0.870 (T=0.6) | 0.878 | +0.008 |
| qwen/seed4 | bfi | 1.10 | 0.421 | 0.874 (T=1.0) | 0.872 | -0.002 |
| llama/seed1 | ipip | 1.20 | 0.504 | 0.922 (T=0.6) | 0.888 | -0.034 |
| llama/seed1 | bfi | 1.20 | 0.406 | 0.866 (T=1.0) | 0.858 | -0.008 |
| llama/seed2 | ipip | 1.20 | 0.509 | 0.896 (T=0.6) | 0.862 | -0.034 |
| llama/seed2 | bfi | 1.20 | 0.418 | 0.830 (T=1.0) | 0.824 | -0.006 |
| llama/seed3 | ipip | 1.20 | 0.466 | 0.916 (T=0.6) | 0.882 | -0.034 |
| llama/seed3 | bfi | 1.20 | 0.379 | 0.866 (T=1.0) | 0.858 | -0.008 |
| llama/seed4 | ipip | 1.20 | 0.593 | 0.858 (T=0.6) | 0.810 | -0.048 |
| llama/seed4 | bfi | 1.20 | 0.481 | 0.764 (T=1.0) | 0.740 | -0.024 |
