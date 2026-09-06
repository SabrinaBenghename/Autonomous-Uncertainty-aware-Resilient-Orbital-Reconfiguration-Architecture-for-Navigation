# Reproducibility

## Environment

Recommended:

- Python 3.11+
- isolated virtual environment
- repository root as working directory

Create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Phase 17 execution order

Run the final mechanism and resilience stages in order:

```powershell
python -m experiments.experiment_017f_controlled_estimator_maturity
python -m experiments.experiment_017g_outage_duration_sweep
python -m experiments.experiment_017h_combined_stress
python -m experiments.experiment_017i_resilience_envelope
```

Earlier completed stages:

```powershell
python -m experiments.experiment_017c_dual_outage_phase_sweep
python -m experiments.experiment_017d_outage_timing_sweep
python -m experiments.experiment_017e_estimator_maturity
```

## Checkpoint behavior

Experiments 017-F/G/H write condition-level CSV results during execution.

If a run is interrupted:

- do not delete the corresponding result CSV,
- restart the same module,
- completed condition/replicate keys should be detected and skipped.

## Results policy

Recommended public-repo policy:

Commit:

- summary CSVs,
- statistical-result CSVs,
- final validation CSVs,
- selected final plots.

Avoid committing:

- duplicate intermediate exports,
- local caches,
- environment folders,
- temporary checkpoints not required for reproducibility,
- very large raw simulation outputs.

## Validation

Each final experiment prints a global validation flag.

Expected:

```text
VALIDATION GLOBALE 017-F : True
VALIDATION GLOBALE 017-G : True
VALIDATION GLOBALE 017-H : True
VALIDATION GLOBALE 017-I : True
```

Do not interpret a scientific hypothesis flag as a validation flag. Validation checks data integrity and design consistency; hypothesis results are scientific outcomes and may legitimately be significant or non-significant.

## Scientific caution

The final low/moderate/high degradation bands are relative empirical categories. They must not be presented as flight-safety, mission-acceptance, or certification thresholds without an external mission requirement.
