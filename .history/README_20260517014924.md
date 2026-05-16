# player-performance-mlops

Project scaffold for player performance ML/Ops.

Structure:

- Place your dataset at `data/new-players-data-full.csv`.
- Notebook: `notebooks/eda.ipynb`.
- Code: `src/preprocess.py`, `src/train.py`.
- API entry: `api/main.py`.
- CI workflow: `.github/workflows/train.yml`.
- Models directory: `models/` (empty for now).

To get started:

1. Paste your dataset into `data/new-players-data-full.csv`.
2. Create a virtualenv and install requirements: `pip install -r requirements.txt`.
3. Run `python src/preprocess.py` and `python src/train.py` as needed.
