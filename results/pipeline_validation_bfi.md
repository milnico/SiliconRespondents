# Pipeline validation — human split-half (bfi_human_long.csv)

Two disjoint random halves of the human sample (seed 42): half A = 1218 ppl, half B = 1218 ppl.

```
Common items: 25  ->  A1, A2, A3, A4, A5, C1, C2, C3, C4, C5, E1, E2, E3, E4, E5, N1, N2, N3, N4, N5, O1, O2, O3, O4, O5

Eigenvalues (top 6):
  human:   5.14 2.58 2.31 1.9 1.53 1.13 
  silicon: 5.14 2.94 2 1.82 1.58 1.05 

Per-scale Cronbach alpha and mean corrected item-total (r.drop):
  O: alpha human=0.630 silicon=0.584 | mean r.drop human=0.389 silicon=0.347 (k=5)
  C: alpha human=0.738 silicon=0.736 | mean r.drop human=0.504 silicon=0.502 (k=5)
  E: alpha human=0.769 silicon=0.761 | mean r.drop human=0.541 silicon=0.531 (k=5)
  A: alpha human=0.718 silicon=0.714 | mean r.drop human=0.484 silicon=0.482 (k=5)
  N: alpha human=0.807 silicon=0.827 | mean r.drop human=0.594 silicon=0.624 (k=5)
  item-total profile corr (human vs silicon) = 0.971  |  MAE = 0.026

Tucker factor congruence (human factor -> best silicon factor):
  per factor: 0.99 0.96 0.96 0.99 0.97 
  MEAN congruence = 0.974   (replication target >= 0.90)
  Phi (inter-factor corr) Frobenius error = 0.192
```
