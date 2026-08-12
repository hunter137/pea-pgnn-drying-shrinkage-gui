# Model card: PEA-PGNN V1.0.0 deployment ensemble

## Intended use

The bundled model provides research-use point predictions of concrete
drying-shrinkage magnitude and compares them with B3-, GL2000-, and
ACI-209-compatible reference calculations. It is intended for scientific
review, method demonstration, sensitivity exploration, and engineering
calculation records within the documented input population.

It is not a design certificate, a code-compliance system, or a substitute for
experimental assessment and independent engineering judgement.

## Model structure

- Three independently initialized deployment members: seeds 42, 123, and 456.
- Frozen 39-variable input schema.
- 104,200 trainable parameters per member.
- Prior-anchored magnitude and characteristic-time corrections.
- Forward constraints enforce non-negative, non-decreasing trajectories
  bounded by the implemented corrected magnitude.

The deployment ensemble was fitted on all 7,492 development records at or
before 365 days after the epoch count was selected from the strict audit. It
must not be described as an independent test fit.

## Recorded audit evidence

The frozen condition-disjoint later-age audit contains 1,237 records. The
recorded ensemble statistics are:

| Statistic | Value |
|---|---:|
| RMSE | 66.18 microstrain |
| MAE | 54.67 microstrain |
| Bias | -15.98 microstrain |
| R-squared | 0.504 |
| Condition-macro RMSE | 63.42 microstrain |
| Trajectory-macro RMSE | 56.54 microstrain |

The private research database and frozen split are not distributed in this
software repository. Their SHA-256 identities, population counts, package
versions, training hardware, and deployment file hashes are recorded in
`artifacts/deployment/manifest.json`.

## Uncertainty and scope

The variation among the three optimization seeds is displayed only as seed
dispersion. It is not a calibrated prediction interval. The GUI deliberately
does not claim a nominal confidence or prediction interval.

Input-support diagnostics combine recorded marginal limits with a robust
nearest-profile distance. A result marked within recorded support is not proof
of code compliance, causal validity, or project-specific suitability.

## Custom formulas

User formulas are comparison curves. They do not replace the B3/GL2000/ACI
prior features embedded in the frozen deployment model. Changing internal
priors requires a revised feature contract, a new model version, and retraining.

