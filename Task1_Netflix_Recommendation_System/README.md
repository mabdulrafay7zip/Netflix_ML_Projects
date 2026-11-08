# Task 1 (Easy) — Netflix Content Recommendation System

**Author:** Muhammad Abdul Rafay — ML Intern

## Problem
Netflix has thousands of titles. When a viewer likes a show, how do we
automatically suggest other titles they are likely to enjoy, without any
user-rating history? This project builds a **content-based recommender**
that finds titles similar to a given title from the catalogue metadata
alone.

## Dataset
The Netflix titles dataset
(`netflix_dataset.csv`, not included in this repository — see "How to Run" below): **8,790 rows** with
columns show_id, type, title, director, country, date_added,
release_year, rating, duration and listed_in. After removing 3 duplicated
titles, **8,787 unique titles** are indexed.

## Approach
1. Fill missing director / country / rating / genre values with "Unknown"
   and drop duplicated titles.
2. Build one text "profile" per title from **listed_in (genres), country,
   director, type and rating**. Names have spaces removed
   (e.g. `MikeFlanagan`) so TF-IDF treats a person as a single token.
3. Vectorise the profiles with **TF-IDF** (5,398 vocabulary terms).
4. Compute **cosine similarity** between titles.
5. `recommend(title, top_n=10)` looks the title up (case-insensitive),
   ranks every other title by similarity and returns the top 10 with
   their similarity scores.

## Real Results (from the actual run in this folder)
- TF-IDF matrix: **8,787 titles × 5,398 features**
- For **"Midnight Mass"** the top recommendations were *The Haunting of
  Hill House*, *Ratched* and *The Haunting of Bly Manor* (all similarity
  **0.7267**) — all US TV-MA horror/drama series, two of them also by
  Mike Flanagan, which confirms the recommender picks up director +
  genre signals correctly.
- For **"Ganglands"** the top pick was *Sentinelle* (**0.7528**),
  followed by *Crime Time* and *Lupin* — French crime/action titles.
- For **"The Great British Baking Show"** all top 10 picks were British
  reality TV shows (top similarity **0.6346**).

Full example outputs: `recommendations_example.txt` ·
Similarity chart: `recommendations_plot.png` · Raw run log: `run_output.txt`

## Screenshots
![Top recommendations for Midnight Mass](recommendations_plot.png)

## How to Run
```bash
pip install pandas scikit-learn matplotlib
python recommendation_system.py
```
The script loads the dataset from `../../data/netflix_dataset.csv` relative to this folder — that is, a `data/` folder sitting next to your clone of this repository (the code sets `DATA_PATH = HERE.parents[1] / "data" / "netflix_dataset.csv"`). The dataset is not included in this repository, so place `netflix_dataset.csv` there before running.

## Interpreting the output

- `Catalogue loaded: 8787 unique titles` — the 8,790-row CSV minus the 3 duplicated titles dropped in `load_catalogue()`; the recommender indexes 8,787 titles.
- `Columns used for profiles: listed_in, country, director, type, rating` — the five columns combined into each title's text profile; `show_id`, `date_added` and `duration` are not used.
- `TF-IDF matrix shape: (8787, 5398)` — 8,787 titles × 5,398 vocabulary terms; the second number is the size of the TF-IDF vocabulary, not a score.
- One `Top 10 recommendations for 'X':` table per example title (Midnight Mass, Ganglands, The Great British Baking Show), with columns `title`, `similarity`, `type`, `country`, `release_year`, `rating`, `listed_in`. `similarity` is the cosine similarity rounded to 4 decimals — 0.7267 means 72.67% profile overlap, not a rating.
- Identical scores are real, not a bug: the first four picks for Midnight Mass all score exactly 0.7267 — titles with near-identical profiles can tie once the score is rounded to 4 decimals.
- The title you ask about never appears in its own list — `recommend()` drops it (it would score 1.0 against itself).
- The final `Saved: ...` line lists the three files the run writes: `recommendations_example.txt` (the three top-10 tables), `results.json` (`catalogue_titles`, `tfidf_features`, and the example title lists), and `recommendations_plot.png` (a horizontal bar chart of the Midnight Mass top 10 only).
- If the title is not in the catalogue the script stops with `'X' was not found in the catalogue.` — the lookup is case-insensitive, so check the spelling.

## Files
- `recommendation_system.py` — main script (recommender class + examples)
- `recommendations_example.txt` — top-10 lists for 3 example titles
- `recommendations_plot.png` — similarity bar chart (submission screenshot)
- `results.json`, `run_output.txt` — machine-readable results and run log

---
Muhammad Abdul Rafay — ML Intern
