# Results Directory

This directory should contain **selected reproducibility outputs**, not every local artifact generated during development.

Recommended tracked outputs:

## Tables

- Phase 17 summary tables
- statistical-test tables
- validation tables
- final empirical resilience-envelope table

## Figures

Recommended final figure set:

1. outage timing vs RMSE
2. estimator-sigma convergence
3. controlled estimator age vs outage RMSE
4. outage duration vs end error
5. outage duration vs end sigma
6. maturity × duration RMSE
7. empirical resilience envelope

Keep filenames stable once the LaTeX report begins so figure paths do not need to be rewritten.

## Do not commit

- temporary images
- duplicate figures
- local exploratory plots
- massive raw outputs
- machine-specific logs
