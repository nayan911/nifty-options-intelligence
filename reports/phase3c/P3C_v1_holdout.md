# P3C_v1 holdout evaluation (2026-09-01..2026-11-23)

Pre-registration commit `b4b2a2b`, code `ff24831`, primary T30_S20_H60. Evaluated once.
Gated in selection: ['H1_day_type_aligned', 'H2_rvol_above_normal', 'H3_near_expiry']; not gated: ['H4_heavyweights_confirm'].

| Hypothesis | Predicts | Pass: hit (95%) n | Fail: hit (95%) n | Diff pp | Pass / fail avg pts | Unavail | p_one_sided | p_holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| H1_day_type_aligned | higher | 46.7% [24.8, 69.9] n=15 | 54.5% [28.0, 78.7] n=11 | -7.8 | 3.33 / 7.27 | 0 | 0.7858 | 1.0 | NOT_SUPPORTED |
| H2_rvol_above_normal | higher | 53.3% [30.1, 75.2] n=15 | 45.5% [21.3, 72.0] n=11 | 7.8 | 6.67 / 2.73 | 0 | 0.5014 | 1.0 | NOT_SUPPORTED |
| H3_near_expiry | lower | 33.3% [6.1, 79.2] n=3 | 52.2% [33.0, 70.8] n=23 | -18.9 | -3.33 / 6.09 | 0 | 0.5056 | 1.0 | INSUFFICIENT |

SUPPORTED = worth re-testing on new data, not a rule to trade. NIFTY points, gross of costs.
