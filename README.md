## Student Burnout Prediction


A custom machine learning pipeline for predicting student burnout risk from tabular survey and behavioral data.

## Project Goals

- Build reproducible burnout prediction experiments.
- Compare baseline and boosted models.
- Support explainability and counterfactual analysis.
- Keep dependency setup stable for Python 3.10.

## Repository Structure

```text
.
|-- README.md
|-- REQUIREMENT.txt
|-- data/
|   |-- raw/
|   |   |-- datasetA.csv
|   |   |-- datasetB.csv
|   |   `-- survey.csv
|   `-- processed/
`-- src/
```

## Setup

1. Use Python 3.10.
2. Activate the project virtual environment if you are working locally:

```bash
.venv\Scripts\activate
```

3. Install dependencies from the single universal requirements file:

```bash
python -m pip install -r REQUIREMENT.txt
```

## Notes on Dependencies

- Dependency versions are pinned for reproducibility.
- `numpy==1.24.4` is used for compatibility with `seaborn==0.12.2`.
- `scipy==1.11.4` is used instead of `1.11.0` because `1.11.0` is yanked on PyPI.
- Dev tools (`black`, `flake8`) are included in the same file to keep setup simple.
- Only one requirements file is used: `REQUIREMENT.txt`.

## Suggested Development Workflow

```bash
black src
flake8 src
```

## Git Readiness

This repository is prepared for Git push, with:

- A single universal dependency file: `REQUIREMENT.txt`
- Standard Python `.gitignore`
- Project documentation in this README
- One local development virtual environment at `.venv/`

## Status

No push has been performed.

