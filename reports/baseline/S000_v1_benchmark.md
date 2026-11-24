# S000 v1 vs matched random entries

Generated 2026-11-24T16:49:24 | commit `426d33b` | 50 matched draws per opportunity (same minute of day, same direction, other days, same stop distance) | 10000 resamples

p = share of random same-size sets at least as good as S000 (one-sided). Gross NIFTY points.

| Config | n | S000 hit | Random hit (5-95% of sets) | p(hit) | S000 avg pts | Random avg pts | p(pts) |
|---|---|---|---|---|---|---|---|
| T30_S20_H60 | 104 | 45.2% | 32.6% (25.0-39.4) | 0.0029 | +5.65 | +0.19 | 0.0062 |
| T30_SS000_H60 | 104 | 35.6% | 27.0% (20.2-33.7) | 0.0294 | +4.15 | +0.11 | 0.0204 |
| T30_S15_H60 | 104 | 39.4% | 27.6% (21.2-34.6) | 0.005 | +5.11 | -0.15 | 0.0032 |
| T40_S20_H60 | 104 | 35.6% | 23.3% (17.3-29.8) | 0.0025 | +5.63 | -0.35 | 0.0094 |

## Sensitivity: whole-day resampling (paired S000 minus random, per opportunity)

Days are resampled with replacement, so opportunities sharing a day's conditions are not treated as independent. Same-day random entries keep the day fixed and vary only the minute: if S000 still beats them, the edge is in WHEN it enters within a day, not only WHICH days it trades.

| Config | Versus | Hit diff (pp) [95% CI] | p(diff <= 0) | Points diff [95% CI] | p(diff <= 0) |
|---|---|---|---|---|---|
| T30_S20_H60 | vs other day matched | +12.6 [2.7, 22.2] | 0.0061 | +5.46 [0.62, 10.19] | 0.0137 |
| T30_S20_H60 | vs same day random | +1.5 [-8.2, 11.1] | 0.3863 | +1.34 [-3.37, 5.94] | 0.2903 |
| T30_SS000_H60 | vs other day matched | +8.7 [0.5, 17.2] | 0.0185 | +3.93 [0.11, 7.75] | 0.0216 |
| T30_SS000_H60 | vs same day random | -1.9 [-10.2, 6.7] | 0.6692 | +0.50 [-3.18, 4.27] | 0.4023 |
| T30_S15_H60 | vs other day matched | +11.0 [1.8, 20.3] | 0.0099 | +4.81 [0.65, 8.94] | 0.0114 |
| T30_S15_H60 | vs same day random | +1.3 [-8.1, 10.7] | 0.3961 | +1.53 [-2.62, 5.68] | 0.2354 |
| T40_S20_H60 | vs other day matched | +11.7 [2.2, 21.6] | 0.0079 | +5.35 [-0.33, 11.1] | 0.0338 |
| T40_S20_H60 | vs same day random | +0.6 [-8.2, 10.0] | 0.4537 | +0.18 [-5.05, 5.55] | 0.4839 |
