# S000 v1 vs matched random entries

Generated 2026-11-24T14:37:44 | commit `acd1833` | 50 matched draws per opportunity (same minute of day, same direction, other days, same stop distance) | 10000 resamples

p = share of random same-size sets at least as good as S000 (one-sided). Gross NIFTY points.

| Config | n | S000 hit | Random hit (5-95% of sets) | p(hit) | S000 avg pts | Random avg pts | p(pts) |
|---|---|---|---|---|---|---|---|
| T30_S20_H60 | 104 | 45.2% | 32.6% (25.0-39.4) | 0.0029 | +5.65 | +0.19 | 0.0062 |
| T30_SS000_H60 | 104 | 35.6% | 27.0% (20.2-33.7) | 0.0294 | +4.15 | +0.11 | 0.0204 |
| T30_S15_H60 | 104 | 39.4% | 27.6% (21.2-34.6) | 0.005 | +5.11 | -0.15 | 0.0032 |
| T40_S20_H60 | 104 | 35.6% | 23.3% (17.3-29.8) | 0.0025 | +5.63 | -0.35 | 0.0094 |
