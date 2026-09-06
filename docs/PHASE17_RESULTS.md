# Phase 17 — Resilience Campaign Results

## Scope

Phase 17 studies AURORA navigation resilience under simultaneous GNSS and star-tracker outages.

The campaign intentionally progresses from observation to controlled intervention:

1. test orbital phase,
2. discover timing sensitivity,
3. examine estimator maturity as a mechanism,
4. intervene directly on estimator age,
5. sweep outage duration,
6. combine maturity and duration,
7. build an empirical degradation envelope.

---

## 017-C — Orbital-phase sweep

- 8 phases × 8 replicates = 64 missions
- fixed outage: 60–75 min

Main result:

- outage RMSE: Friedman p ≈ 0.731, Kendall's W ≈ 0.079
- end error: Friedman p ≈ 0.635, Kendall's W ≈ 0.093

**Interpretation:** no statistically significant phase dependence was detected under the tested conditions.

---

## 017-D — Outage-timing sweep

- outage starts: 15, 30, 45, 60, 75, 90, 105, 120 min
- fixed duration: 15 min
- 8 replicates per timing = 64 missions

Main result:

- outage RMSE: p ≈ 0.000018, W ≈ 0.606
- end error: p ≈ 0.000037, W ≈ 0.576
- outage-start sigma strongly associated with outage error
- PDOP not significantly correlated with outage RMSE

**Interpretation:** outage timing strongly changes vulnerability, but PDOP alone does not explain the effect.

---

## 017-E — Estimator-maturity mechanism analysis

Zero new missions; 64 missions loaded from 017-D.

Main result:

- mission time vs outage-start sigma: ρ ≈ -0.921
- mission time vs outage RMSE: ρ ≈ -0.761
- start sigma vs outage RMSE: ρ ≈ 0.754
- descriptive convergence fit: R² ≈ 0.984, τ ≈ 34.3 min
- sigma vs RMSE controlling time: p ≈ 0.092
- within-timing sigma variation only ≈ 0.28% of total variation

**Interpretation:** estimator maturity is a plausible mechanism, but mission time and estimator uncertainty are too strongly confounded for causal attribution from 017-D alone.

---

## 017-F — Controlled estimator-maturity intervention

- outage fixed at 60–75 min
- estimator ages: 5, 15, 30, 45, 60 min
- 8 replicates per age = 40 missions

| Estimator age | Start sigma [m] | Outage RMSE [m] | End error [m] |
|---:|---:|---:|---:|
| 5 min | 0.947 | 4.476 | 7.570 |
| 15 min | 0.674 | 2.217 | 3.956 |
| 30 min | 0.562 | 1.546 | 2.609 |
| 45 min | 0.462 | 1.240 | 2.045 |
| 60 min | 0.390 | 0.803 | 1.179 |

Main result:

- outage RMSE: p ≈ 0.000331, W ≈ 0.653
- end error: p ≈ 0.000167, W ≈ 0.700
- age vs outage RMSE: ρ ≈ -0.668
- PDOP vs outage RMSE: not significant

Mean outage RMSE fell by about **82%** between the 5-min and 60-min estimator conditions.

**Interpretation:** controlled navigation-estimator maturity materially affects resilience under the same physical outage timing.

---

## 017-G — Outage-duration sweep

- mature estimator
- outage start fixed at 60 min
- durations: 5, 10, 15, 20, 30 min
- 8 replicates each = 40 missions

| Duration | Outage RMSE [m] | End error [m] | End sigma [m] |
|---:|---:|---:|---:|
| 5 min | 0.396 | 0.458 | 0.561 |
| 10 min | 0.485 | 0.659 | 0.807 |
| 15 min | 0.611 | 0.947 | 1.132 |
| 20 min | 0.781 | 1.333 | 1.545 |
| 30 min | 1.257 | 2.402 | 2.629 |

Main result:

- outage RMSE: p ≈ 0.000002, W = 1.000
- end error: p ≈ 0.000002, W = 1.000
- end sigma: p ≈ 0.000002, W = 1.000
- post-recovery RMSE: p ≈ 0.121, not significant

**Interpretation:** longer outages systematically increase in-outage error and covariance, while the tested post-recovery metric does not show significant duration dependence.

---

## 017-H — Combined maturity × duration stress

- estimator ages: 5, 30, 60 min
- durations: 15 and 30 min
- 8 replicates per cell = 48 missions

| Estimator age | 15-min RMSE | 30-min RMSE | 15-min end error | 30-min end error |
|---:|---:|---:|---:|---:|
| 5 min | 2.773 | 5.667 | 4.438 | 10.482 |
| 30 min | 1.291 | 2.624 | 2.073 | 4.829 |
| 60 min | 0.599 | 1.006 | 0.856 | 1.672 |

RMSE penalty from extending 15 → 30 min:

- 5-min estimator: +2.894 m
- 30-min estimator: +1.334 m
- 60-min estimator: +0.407 m

Interaction-style diagnostic:

- RMSE penalty vs maturity: p ≈ 0.0111, W ≈ 0.562
- end-sigma penalty vs maturity: p ≈ 0.000335, W = 1.000
- end-error penalty: p ≈ 0.197, not significant

**Interpretation:** maturity moderates the impact of long outages on RMSE and covariance growth.

---

## 017-I — Empirical resilience envelope

Zero new missions.

Reference:

- estimator age: 60 min
- outage duration: 5 min
- outage RMSE: 0.396 m
- end error: 0.458 m

Descriptive bands:

- LOW_DEGRADATION: ≤ 2× reference
- MODERATE_DEGRADATION: > 2× and ≤ 5×
- HIGH_DEGRADATION: > 5×

These bands are **not safety or certification limits**.

Selected points:

| Estimator age | Duration | RMSE [m] | End error [m] | Severity | Band |
|---:|---:|---:|---:|---:|---|
| 60 | 5 | 0.396 | 0.458 | 1.00 | Low |
| 60 | 10 | 0.485 | 0.659 | 1.44 | Low |
| 60 | 15 | 0.671 | 0.994 | 2.17 | Moderate |
| 60 | 20 | 0.781 | 1.333 | 2.91 | Moderate |
| 60 | 30 | 1.131 | 2.037 | 4.45 | Moderate |
| 45 | 15 | 1.240 | 2.045 | 4.46 | Moderate |
| 30 | 15 | 1.419 | 2.341 | 5.11 | High |
| 15 | 15 | 2.217 | 3.956 | 8.63 | High |
| 5 | 15 | 3.624 | 6.004 | 13.10 | High |
| 5 | 30 | 5.667 | 10.482 | 22.87 | High |

---

## Final Phase 17 conclusion

Within the tested AURORA configuration, the dominant operational factors are:

- **navigation-estimator maturity**
- **duration of simultaneous measurement denial**

Initial orbital phase was not detected as a significant driver in the fixed-outage phase sweep.

A mature estimator acts as a strong resilience buffer against prolonged measurement loss.
