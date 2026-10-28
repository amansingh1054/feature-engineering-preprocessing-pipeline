# Feature Engineering & Preprocessing Pipeline

A reusable, leak-aware Scikit-Learn workflow for tabular machine-learning data. The notebook demonstrates missing-value imputation, numerical scaling, categorical one-hot encoding, numeric correlation analysis, model evaluation, and mutual-information feature ranking.

## Repository contents

- `feature_engineering_preprocessing_pipeline.ipynb` — main Jupyter Notebook with the complete workflow.
- `requirements.txt` — Python dependencies.
- `.gitignore` — ignores local environments and datasets.

## Run it

### Option 1: Google Colab
1. Download this repository ZIP and extract it.
2. Open [Google Colab](https://colab.research.google.com/).
3. Upload `feature_engineering_preprocessing_pipeline.ipynb`.
4. Run the cells from top to bottom.

### Option 2: Local Python
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Open `feature_engineering_preprocessing_pipeline.ipynb` and run all cells.

## Method

1. Generate a synthetic housing-prices-style dataset with a few missing values.
2. Inspect missingness and calculate correlations for numeric columns.
3. Split into training and test sets before fitting transformations.
4. Use `ColumnTransformer` with pipelines:
   - numeric: median imputation + `StandardScaler`
   - categorical: most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`
5. Fit a `RandomForestRegressor` and report MAE, RMSE, and R² on the held-out test set.
6. Rank transformed training features using mutual information.

## Important notes

- The notebook uses synthetic data so it can run without downloading a dataset. It is not a real housing dataset and its scores are not evidence of real-world performance.
- Preprocessing is fitted on training data only, through the pipeline, to reduce leakage risk.
- One-hot encoding is used; target encoding is not implemented in this starter.
- Before using real data, verify data permissions, provenance, target definition, feature availability at prediction time, and appropriate split strategy.
- This is an educational starter, not a production-ready or mentor-approved implementation.

## Suggested GitHub topics

`machine-learning` `feature-engineering` `scikit-learn` `data-preprocessing` `python` `jupyter-notebook`
