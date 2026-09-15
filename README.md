# AG News Text Classification — Multinomial Naive Bayes

A multi-class text classification project that categorizes news articles into 4 categories (World, Sports, Business, Sci/Tech) using the [AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) from Kaggle (120,000 training rows, 7,600 test rows).

## What it does

- Implements **Multinomial Naive Bayes from scratch**, including the underlying log-probability math (no black-box `sklearn.naive_bayes` used for the core algorithm).
- Full preprocessing pipeline: text cleaning, stopword removal, and TF-IDF / Count vectorization (`scikit-learn`, `pandas`, `NumPy`).
- **Hyperparameter tuning** via grid search combined with **5-fold stratified cross-validation**, optimizing for macro-F1 (appropriate for a balanced 4-class problem).
- Trains a final model on the full training set and evaluates on the held-out test set.
- Addresses **class imbalance** (under/over-sampling) and compares approaches empirically.
- Adds basic **explainability**: surfaces the most characteristic words per class via `feature_log_prob_`, and shows which words drove an individual prediction.

## Tech stack

Python · pandas · NumPy · scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `StratifiedKFold`, metrics) · matplotlib

## Notes

- The dataset itself (`train.csv` / `test.csv`) is not included in this repo — download it from Kaggle and place both files in the project root before running.
- This was completed as a course assignment. AI-assisted development tools (Claude) were used for scaffolding boilerplate functions, checking math implementations, and code refactoring; all logic was reviewed, adapted, and validated independently against the assignment requirements. This is documented in full inside the notebook itself.
