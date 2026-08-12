# Contributing

Contributions that improve reproducibility, accessibility, numerical safety,
documentation, or Windows usability are welcome.

1. Create a topic branch from `main`.
2. Keep scientific computation outside the GUI layer.
3. Add or update tests for every behavioral change.
4. Run `python -m unittest discover -s tests -v` and
   `python run_gui.py --smoke` before opening a pull request.
5. Do not commit research databases, frozen split files, unpublished
   manuscript material, user formula data, or generated reports.

Changes to the 39-variable feature contract, embedded priors, model
architecture, or deployment weights require an explicit model-version change
and retraining audit. A custom comparison formula alone must not silently
alter the frozen neural model.

