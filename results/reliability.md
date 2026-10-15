# Silicon test-retest reliability (§7.12)

Respondent population held FIXED (same latents z); instrument re-administered twice under fresh response-sampling noise (sample_seed 101 vs 202). `retest_r_scale` = Pearson r of reverse-coded scale scores across respondents; `mean_item_retest_r` = mean per-item retest r (key-agnostic); `identity_r` = scale-score r when the second administration reuses sample_seed 101 (determinism check, should be 1.0).


## qwen_seed0_ipip  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.515 | 0.29 | 4 | 0.391 | 0.34 | n/a |
| C | 0.613 | 0.34 | 4 | 0.546 | 0.503 | n/a |
| E | 0.81 | 0.564 | 4 | 0.785 | 0.746 | n/a |
| A | 0.625 | 0.376 | 4 | 0.536 | 0.547 | n/a |
| N | 0.663 | 0.343 | 4 | 0.586 | 0.568 | n/a |

Mean scale-score retest r: **0.645**.

## qwen_seed0_bfi  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.46 | 0.301 | 5 | 0.34 | 0.341 | n/a |
| C | 0.582 | 0.272 | 5 | 0.492 | 0.56 | n/a |
| E | 0.776 | 0.462 | 5 | 0.717 | 0.712 | n/a |
| A | 0.746 | 0.418 | 5 | 0.755 | 0.779 | n/a |
| N | 0.825 | 0.493 | 5 | 0.821 | 0.818 | n/a |

Mean scale-score retest r: **0.678**.

## qwen_seed1_ipip  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.589 | 0.355 | 4 | 0.436 | 0.413 | n/a |
| C | 0.577 | 0.284 | 4 | 0.539 | 0.528 | n/a |
| E | 0.819 | 0.568 | 4 | 0.79 | 0.769 | n/a |
| A | 0.484 | 0.3 | 4 | 0.407 | 0.381 | n/a |
| N | 0.709 | 0.377 | 4 | 0.663 | 0.623 | n/a |

Mean scale-score retest r: **0.636**.

## qwen_seed1_bfi  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.622 | 0.383 | 5 | 0.515 | 0.508 | n/a |
| C | 0.572 | 0.256 | 5 | 0.486 | 0.55 | n/a |
| E | 0.776 | 0.453 | 5 | 0.738 | 0.725 | n/a |
| A | 0.719 | 0.388 | 5 | 0.73 | 0.752 | n/a |
| N | 0.82 | 0.486 | 5 | 0.813 | 0.813 | n/a |

Mean scale-score retest r: **0.702**.

## qwen_seed2_ipip  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.482 | 0.293 | 4 | 0.316 | 0.276 | n/a |
| C | 0.653 | 0.358 | 4 | 0.582 | 0.563 | n/a |
| E | 0.805 | 0.547 | 4 | 0.772 | 0.763 | n/a |
| A | 0.572 | 0.329 | 4 | 0.478 | 0.482 | n/a |
| N | 0.643 | 0.342 | 4 | 0.553 | 0.536 | n/a |

Mean scale-score retest r: **0.631**.

## qwen_seed2_bfi  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.589 | 0.358 | 5 | 0.492 | 0.474 | n/a |
| C | 0.62 | 0.291 | 5 | 0.532 | 0.596 | n/a |
| E | 0.757 | 0.435 | 5 | 0.716 | 0.7 | n/a |
| A | 0.682 | 0.348 | 5 | 0.686 | 0.714 | n/a |
| N | 0.818 | 0.493 | 5 | 0.817 | 0.817 | n/a |

Mean scale-score retest r: **0.693**.

## qwen_seed3_ipip  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.508 | 0.289 | 4 | 0.292 | 0.284 | n/a |
| C | 0.544 | 0.282 | 4 | 0.447 | 0.468 | n/a |
| E | 0.644 | 0.351 | 4 | 0.605 | 0.548 | n/a |
| A | 0.354 | 0.234 | 4 | 0.27 | 0.277 | n/a |
| N | 0.491 | 0.305 | 4 | 0.319 | 0.276 | n/a |

Mean scale-score retest r: **0.508**.

## qwen_seed3_bfi  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.461 | 0.298 | 5 | 0.297 | 0.31 | n/a |
| C | 0.528 | 0.275 | 5 | 0.401 | 0.491 | n/a |
| E | 0.694 | 0.351 | 5 | 0.627 | 0.633 | n/a |
| A | 0.646 | 0.334 | 5 | 0.647 | 0.701 | n/a |
| N | 0.813 | 0.482 | 5 | 0.793 | 0.79 | n/a |

Mean scale-score retest r: **0.628**.

## qwen_seed4_ipip  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.485 | 0.276 | 4 | 0.335 | 0.321 | n/a |
| C | 0.588 | 0.282 | 4 | 0.529 | 0.511 | n/a |
| E | 0.769 | 0.49 | 4 | 0.751 | 0.718 | n/a |
| A | 0.532 | 0.312 | 4 | 0.505 | 0.457 | n/a |
| N | 0.604 | 0.319 | 4 | 0.515 | 0.485 | n/a |

Mean scale-score retest r: **0.596**.

## qwen_seed4_bfi  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.604 | 0.342 | 5 | 0.501 | 0.481 | n/a |
| C | 0.59 | 0.278 | 5 | 0.506 | 0.563 | n/a |
| E | 0.766 | 0.435 | 5 | 0.74 | 0.738 | n/a |
| A | 0.701 | 0.355 | 5 | 0.717 | 0.738 | n/a |
| N | 0.823 | 0.488 | 5 | 0.815 | 0.805 | n/a |

Mean scale-score retest r: **0.697**.

## llama_seed1_ipip  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.484 | 0.239 | 4 | 0.405 | 0.368 | n/a |
| C | 0.548 | 0.245 | 4 | 0.504 | 0.51 | n/a |
| E | 0.727 | 0.44 | 4 | 0.673 | 0.648 | n/a |
| A | 0.67 | 0.395 | 4 | 0.609 | 0.592 | n/a |
| N | 0.708 | 0.387 | 4 | 0.684 | 0.645 | n/a |

Mean scale-score retest r: **0.627**.

## llama_seed1_bfi  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.585 | 0.297 | 5 | 0.48 | 0.497 | n/a |
| C | 0.513 | 0.22 | 5 | 0.443 | 0.505 | n/a |
| E | 0.674 | 0.337 | 5 | 0.641 | 0.656 | n/a |
| A | 0.734 | 0.386 | 5 | 0.704 | 0.747 | n/a |
| N | 0.839 | 0.525 | 5 | 0.839 | 0.826 | n/a |

Mean scale-score retest r: **0.669**.

## llama_seed2_ipip  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.519 | 0.255 | 4 | 0.429 | 0.394 | n/a |
| C | 0.521 | 0.254 | 4 | 0.486 | 0.408 | n/a |
| E | 0.744 | 0.473 | 4 | 0.662 | 0.588 | n/a |
| A | 0.717 | 0.444 | 4 | 0.65 | 0.648 | n/a |
| N | 0.758 | 0.428 | 4 | 0.695 | 0.667 | n/a |

Mean scale-score retest r: **0.652**.

## llama_seed2_bfi  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.612 | 0.324 | 5 | 0.519 | 0.523 | n/a |
| C | 0.556 | 0.277 | 5 | 0.478 | 0.533 | n/a |
| E | 0.721 | 0.404 | 5 | 0.678 | 0.695 | n/a |
| A | 0.772 | 0.426 | 5 | 0.744 | 0.761 | n/a |
| N | 0.871 | 0.578 | 5 | 0.865 | 0.862 | n/a |

Mean scale-score retest r: **0.706**.

## llama_seed3_ipip  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.558 | 0.279 | 4 | 0.458 | 0.459 | n/a |
| C | 0.488 | 0.224 | 4 | 0.46 | 0.442 | n/a |
| E | 0.741 | 0.436 | 4 | 0.712 | 0.676 | n/a |
| A | 0.693 | 0.416 | 4 | 0.645 | 0.621 | n/a |
| N | 0.701 | 0.367 | 4 | 0.643 | 0.615 | n/a |

Mean scale-score retest r: **0.636**.

## llama_seed3_bfi  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.637 | 0.342 | 5 | 0.566 | 0.557 | n/a |
| C | 0.606 | 0.285 | 5 | 0.525 | 0.598 | n/a |
| E | 0.718 | 0.374 | 5 | 0.678 | 0.7 | n/a |
| A | 0.755 | 0.429 | 5 | 0.752 | 0.77 | n/a |
| N | 0.85 | 0.537 | 5 | 0.845 | 0.839 | n/a |

Mean scale-score retest r: **0.713**.

## llama_seed4_ipip  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.478 | 0.273 | 4 | 0.355 | 0.286 | n/a |
| C | 0.476 | 0.246 | 4 | 0.373 | 0.382 | n/a |
| E | 0.674 | 0.377 | 4 | 0.6 | 0.547 | n/a |
| A | 0.689 | 0.404 | 4 | 0.667 | 0.627 | n/a |
| N | 0.66 | 0.338 | 4 | 0.632 | 0.604 | n/a |

Mean scale-score retest r: **0.595**.

## llama_seed4_bfi  (n=500)

| factor | retest r (scale) | mean item retest r | n items | alpha A | alpha B | identity r |
|---|---|---|---|---|---|---|
| O | 0.601 | 0.348 | 5 | 0.464 | 0.483 | n/a |
| C | 0.537 | 0.255 | 5 | 0.455 | 0.529 | n/a |
| E | 0.689 | 0.358 | 5 | 0.626 | 0.628 | n/a |
| A | 0.671 | 0.341 | 5 | 0.657 | 0.697 | n/a |
| N | 0.845 | 0.528 | 5 | 0.841 | 0.834 | n/a |

Mean scale-score retest r: **0.669**.
