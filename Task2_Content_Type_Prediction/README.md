# Task 2 (Easy) — Netflix Content Type Prediction Model

**Author:** Muhammad Abdul Rafay — ML Intern

## Problem
Given only the descriptive information of a Netflix catalogue entry,
predict whether it is a **Movie** or a **TV Show** — a binary
classification problem.

## Dataset
The Netflix titles dataset
(`netflix_dataset.csv`, not included in this repository — see "How to Run" below):
**8,790 rows** (6,126 Movies / 2,664 TV Shows). Split 80/20 with
stratification: 7,032 train / 1,758 test rows.

## Approach
Features engineered from the raw columns:
- **Numeric:** release_year, year/month the title was added to Netflix,
  whether a director is credited, number of words in the title,
  number of genres listed.
- **Categorical (one-hot):** rating grouped into a maturity band,
  genre *theme* of the first listed genre, main country (top 10,
  rest = "Other").

**Leakage control (important):** three raw columns would hand the model
the answer, so they were neutralised:
- `duration` ("90 min" vs "2 Seasons") — excluded completely.
- `listed_in` literally contains "TV Dramas" vs "…Movies" — reduced to
  the genre theme only (Drama, Comedy, Crime, …).
- `rating` ("TV-MA" vs "R") — grouped into Kids / Family / Teens / Adult.

A first version without these fixes scored a meaningless 100%, which is
exactly why leakage checks matter. Three classifiers were then trained
in identical preprocessing pipelines: Logistic Regression, Decision
Tree and Random Forest.

## Real Results (test set, 1,758 titles — from the actual run)
| Model | Accuracy |
|---|---|
| Logistic Regression | 0.9158 |
| Decision Tree | 0.8862 |
| **Random Forest (best)** | **0.9192** |

Best model detail (Random Forest): Movie precision 0.93 / recall 0.95;
TV Show precision 0.89 / recall 0.84. TV Shows are the harder class —
they are the minority (30%) and overlap with movies on most features.

## Screenshots
![Confusion matrix](confusion_matrix.png)
![Model comparison](model_comparison.png)

Full per-model classification reports: `run_output.txt` ·
Numbers: `results.json`

## How to Run
```bash
pip install pandas scikit-learn matplotlib
python content_type_prediction.py
```
The script loads the dataset from `../../data/netflix_dataset.csv` relative to this folder — that is, a `data/` folder sitting next to your clone of this repository (the code sets `DATA_PATH = HERE.parents[1] / "data" / "netflix_dataset.csv"`). The dataset is not included in this repository, so place `netflix_dataset.csv` there before running.

## Interpreting the output
- `Dataset rows: 8790` — the full catalogue loaded; `Class balance: {'Movie': 6126, 'TV Show': 2664}` shows the 70/30 imbalance before any training.
- `Train: 7032 | Test: 1758` — the 80/20 stratified split, so both sets keep the same 70/30 movie/show mix.
- One `Name: accuracy = 0.XXXX` line per model (Logistic Regression, Decision Tree, Random Forest), each followed by a classification report — accuracy to 4 decimals, while the `accuracy` row inside each report is the same number rounded to 2 decimals (0.92 for the best model).
- In each report the `support` column is the test-set count per class: Movie 1,225, TV Show 533 (total 1,758). Precision and recall are per class — Random Forest's TV Show recall of 0.84 means 16% of real TV shows were predicted as movies.
- TV Show recall is the weak spot in all three models (0.80 / 0.81 / 0.84) — the minority class (30%) that overlaps most with movies on these features, which is why the best-model detail above calls it the harder class.
- `Best model: Random Forest (0.9192)` — the highest of the three test accuracies (0.9158 / 0.8862 / 0.9192), matching the Real Results table above.
- The final `Saved:` line lists the three files the run writes: `confusion_matrix.png` (best model's matrix), `model_comparison.png` (accuracy chart) and `results.json` (dataset/test row counts, per-model accuracies, best model name) — the full per-model reports are in `run_output.txt`.

## Files
- `content_type_prediction.py` — main script
- `confusion_matrix.png` — best-model confusion matrix (screenshot)
- `model_comparison.png` — accuracy comparison chart (screenshot)
- `results.json`, `run_output.txt` — results and full run log

---
Muhammad Abdul Rafay — ML Intern
