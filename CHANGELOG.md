# Changelog

All notable changes to the **Shipment Delay Intelligence Pipeline** are documented here.

**Project Author and Maintainer:** ANIKET RAJAK

## [Unreleased] - 2026-11-05

### Documentation and identity

- Rewrote `README.md` from scratch with verified project information, installation and execution instructions, directory structure, architecture and workflow diagrams, actual model results, operational recommendations, limitations, and future improvements.
- Consolidated the project identity so that **ANIKET RAJAK** is consistently identified as the project author and maintainer across project documentation.
- Added a neutral repository-history and licensing note. The Git history is preserved, and no historical commits are rewritten.

### Reproducibility and input validation

- Updated `generate_data.py` to resolve the `data/` directory relative to the script location instead of the current working directory.
- Added explicit input validation to `predict_delay.py` for required columns, unknown hub IDs, unknown service types, invalid timestamps, missing model features, and non-finite feature values.
- Clarified the dataset-specific 48-shipment guard in `predict_delay.py`.

### Repository maintenance

- Expanded `.gitignore` with Python cache, test/tool cache, virtual-environment, OS, and build/distribution entries.

### Verification

- Re-verified the committed Part A, Part B, Part C, and Part D outputs against the source code and data.
- Re-verified the saved classifier, model reload check, and the 48 new-shipment predictions.
- No ML algorithm, hyperparameter, random seed, train/validation/test split, classification threshold, feature list, mathematical calculation, data file, model artifact, or prediction rule was changed.

## [2026-11-03] - Reproducibility and maintenance

### Reproducibility

- Replaced hard-coded `/content` paths in `part_a_clean_eda.py`, `part_b_regression.py`, `part_c_classification.py`, `part_d_segments.py`, and `predict_delay.py` with paths resolved from the script location.
- Restored the data-quality and distribution checks in Part A so that `outputs/part_a_checks.csv` is reproduced with the same 26 metrics present in the committed artifact.

### Dependencies

- Declared `scipy==1.13.1`, which `part_d_segments.py` imports.
- Aligned `scikit-learn` to `1.6.1`, the version embedded in the committed `models/delay_model.pkl` artifact.

### Readability

- Added module and function docstrings to `src/shipment_prep.py`.
- Renamed the local `columns` variable to `impute_columns` in `src/shipment_prep.py`.
- Renamed the local `km3` variable to `final_kmeans` in `part_d_segments.py`.

### Explicitly unchanged

- No ML algorithm, hyperparameter, random seed, train/validation/test split, classification threshold, feature list, mathematical calculation, model output, or prediction rule was changed.
- The committed `data/`, `models/`, and `outputs/` artifacts were not modified.
- The repository does not include a `LICENSE` file. No license is invented or reassigned.
