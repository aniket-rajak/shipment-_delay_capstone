# Changelog

All notable changes to this customized version of the Shipment Delay
Intelligence Pipeline are documented here.

The original implementation was created by Manish Dhanuk
(`mrundefined099-debug`) as an AI/ML capstone project. This changelog records
only the customizations made to that inherited codebase.

## [Unreleased] - 2026-11-03

### Customized by Aniket Rajak

#### Documentation and identity
- Renamed the documented project title to "Shipment Delay Intelligence
  Pipeline" and added the customize description and maintainer identity.
- Added "Authorship & Attribution", "Ownership & Maintenance", and
  "Customization Notes" sections to `README.md`.
- Added local run guidance, including the recommended Python version and the
  scikit-learn version used to build the saved model.
- Added this `CHANGELOG.md`.

#### Reproducibility
- Replaced hard-coded `/content` paths in `part_a_clean_eda.py`,
  `part_b_regression.py`, `part_c_classification.py`, `part_d_segments.py`,
  and `predict_delay.py` with paths resolved from the script location.
  Scripts now run from the project root on any platform.
- Restored the data-quality and distribution checks in Part A so that
  `outputs/part_a_checks.csv` is reproduced with the same 26 metrics already
  present in the committed artifact.

#### Dependencies
- Declared `scipy==1.13.1`, which `part_d_segments.py` already imports.
- Aligned `scikit-learn` to `1.6.1`, the version embedded in the committed
  `models/delay_model.pkl` artifact.

#### Readability
- Added module/function docstrings to `src/shipment_prep.py`.
- Renamed local `columns` to `impute_columns` in `src/shipment_prep.py`.
- Renamed local `km3` to `final_kmeans` in `part_d_segments.py`.

### Explicitly unchanged
- No ML algorithm, hyperparameter, random seed, train/validation/test split,
  classification threshold, feature list, mathematical calculation, model
  output, or prediction rule was changed.
- Original `data/`, `models/`, and `outputs/` artifacts were not modified.
- `generate_data.py` and `package_model.py` were not modified.
- Original authorship and attribution notices were preserved. No license file
  was added or reassigned.
