# Pipeline validation — human split-half (item_holdout.parquet)

Two disjoint random halves of the human sample (seed 42): half A = 1972 ppl, half B = 1972 ppl.

```
Common items: 20  ->  A2, A3, A7, A8, C1, C2, C6, C9, E1, E10, E3, E7, N2, N5, N6, N9, O1, O2, O6, O9

Eigenvalues (top 6):
  human:   3.98 2.23 1.98 1.65 1.38 1 
  silicon: 3.95 2.35 1.85 1.62 1.34 0.97 

Per-scale Cronbach alpha and mean corrected item-total (r.drop):
  O: alpha human=0.505 silicon=0.529 | mean r.drop human=0.298 silicon=0.318 (k=4)
  C: alpha human=0.694 silicon=0.710 | mean r.drop human=0.481 silicon=0.499 (k=4)
  E: alpha human=0.815 silicon=0.810 | mean r.drop human=0.637 silicon=0.629 (k=4)
  A: alpha human=0.674 silicon=0.636 | mean r.drop human=0.465 silicon=0.426 (k=4)
  N: alpha human=0.747 silicon=0.744 | mean r.drop human=0.543 silicon=0.539 (k=4)
  item-total profile corr (human vs silicon) = 0.975  |  MAE = 0.026

Tucker factor congruence (human factor -> best silicon factor):
  per factor: 0.99 0.99 0.99 0.99 0.98 
  MEAN congruence = 0.988   (replication target >= 0.90)
  Phi (inter-factor corr) Frobenius error = 0.200
```
