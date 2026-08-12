## Summary

Describe the user-visible or scientific change.

## Verification

- [ ] `python -m unittest discover -s tests -v`
- [ ] `python run_gui.py --smoke`
- [ ] Deployment manifest hashes still validate, or the model version and
      manifest were deliberately regenerated.
- [ ] No private data, user formulas, generated reports, credentials, or
      unpublished manuscript files are included.

## Scientific contract

- [ ] This change does not silently alter the 39-variable feature contract,
      native priors, architecture, or trained weights; or the required model
      version and retraining evidence are included.

