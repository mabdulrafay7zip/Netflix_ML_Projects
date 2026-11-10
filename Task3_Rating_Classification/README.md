# Task 3 (Medium) — Netflix Audience Rating Classification

**Author:** Muhammad Abdul Rafay — ML Intern

## Problem
Predict a title's **audience rating** (TV-MA, TV-14, TV-PG, R, PG-13,
TV-Y7, …) from its catalogue features — a multi-class classification
problem. If ratings can be predicted reliably, missing or inconsistent
ratings in a catalogue can be flagged automatically.

## Dataset
The Netflix titles dataset
(`netflix_dataset.csv`, not included in this repository — see "How to Run" below):
**8,790 rows, all 8,790 with a usable rating**. Ratings appearing fewer
than 100 times (NC-17, TV-Y7-FV, UR) were merged into an **"Other"**
class, giving **10 classes**: TV-MA (3,205), TV-14 (2,157), TV-PG (861),
R (799), PG-13 (490), TV-Y7 (333), TV-Y (306), PG (287), TV-G (220),
Other (132). Split 80/20 stratified: 7,032 train / 1,758 test.

## Approach
Features: type, release year, parsed duration value + unit
(minutes vs seasons), main genre, main country (top 10, rest = "Other"),
whether a director is credited, and content age. Categoricals are
one-hot encoded inside a scikit-learn pipeline.

Models compared:
1. **Decision Tree**
2. **Random Forest** (300 trees)
3. **Random Forest (tuned)** — GridSearchCV (3-fold) over
   n_estimators {200, 400}, max_depth {None, 20},
   min_samples_leaf {1, 2}. Best parameters found:
   `max_depth=20, min_samples_leaf=2, n_estimators=200`.

## Real Results (test set — from the actual run)
| Model | Accuracy |
|---|---|
| Decision Tree | 0.4306 |
| Random Forest | 0.4494 |
| **Random Forest (tuned, best)** | **0.5085** |

Honest interpretation: with 10 classes, the tuned forest's **50.85%**
is well above the majority-class baseline (~36.5%, always predicting
TV-MA), and tuning added +5.9 points over the plain forest. Per-class
detail (see `classification_report.txt`): TV-MA is recognised best
(F1 0.66, recall 0.80), while neighbouring bands are often confused —
e.g. TV-PG (F1 0.16) and TV-G (F1 0.04) blur into TV-14 / TV-Y7, and the
small "Other" class is never predicted. That is expected: a rating is a
human content judgement, and metadata alone only partly determines it.

## Screenshots
![Accuracy comparison](accuracy_comparison.png)
![Confusion matrix](confusion_matrix.png)

Full classification report: `classification_report.txt` ·
Numbers: `results.json` · Full run log: `run_output.txt`

## How to Run
```bash
pip install pandas scikit-learn matplotlib
python rating_classification.py
```
The script loads the dataset from `../../data/netflix_dataset.csv` relative to this folder — that is, a `data/` folder sitting next to your clone of this repository (the code sets `DATA_PATH = HERE.parents[1] / "data" / "netflix_dataset.csv"`). The dataset is not included in this repository, so place `netflix_dataset.csv` there before running.

Note: the grid-search step makes this script take ~2–3 minutes.

## Interpreting the output

- `Rows with a usable rating: 8790 of 8790` — nothing was dropped: every row in the dataset has a rating, so all 8,790 titles are used.
- `Rating classes: {...}` lists all 10 classes with their counts — TV-MA dominates with 3,205 titles (~36.5%), which is also the majority-class baseline: always guessing TV-MA would score ~36.5%.
- `Train: 7032 | Test: 1758 | Classes: 10` — the 80/20 stratified split, so every class keeps its share in both halves.
- One accuracy line per model — Decision Tree `0.4306`, Random Forest `0.4494`, Tuned Random Forest `0.5085` — with the tuned line also printing the winning grid-search parameters (`max_depth=20, min_samples_leaf=2, n_estimators=200`); that grid search is why the run takes ~2–3 minutes.
- `Best model: Random Forest (tuned) (0.5085)` is followed by the full classification report: the `support` column is the test-set count per class (TV-MA 641 down to Other 27), and the `accuracy` row at the bottom (0.51) is the same 0.5085 rounded to two decimals.
- Read the report's per-class rows, not just the headline: TV-MA is recognised best (recall 0.80, F1 0.66), TV-G is barely detected at all (recall 0.02, F1 0.04), and `Other` scores 0.00 on every metric — the model never predicts it.
- `macro avg` (F1 0.35) averages the ten classes equally, so the tiny classes drag it down; `weighted avg` (F1 0.48) weights by support and sits close to the overall accuracy.
- The final `Saved:` line lists the four files the run writes: `classification_report.txt` (best model, its accuracy and the full report), `accuracy_comparison.png` (bar chart of the three accuracies), `confusion_matrix.png` (best model only, rows = actual rating, columns = predicted rating) and `results.json` (dataset rows, class count, per-model accuracies, best model and its best parameters).

## Files
- `rating_classification.py` — main script
- `accuracy_comparison.png`, `confusion_matrix.png` — screenshots
- `classification_report.txt`, `results.json`, `run_output.txt`

---
Muhammad Abdul Rafay — ML Intern
