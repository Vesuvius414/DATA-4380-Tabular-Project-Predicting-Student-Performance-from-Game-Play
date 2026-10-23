# Predicting Student Performance from Game Play

This project predicts whether a student answers each in-game question correctly, using the event logs from the educational game *Jo Wilder and the Capitol Case*. It is based on the Kaggle competition [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play).

The final model is a tuned **LightGBM** classifier. On a held-out test set of unseen sessions, it reaches **75.2% accuracy** and a **ROC-AUC of 0.773**. The majority-class baseline reaches 70.4% accuracy.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
└── tabular/
    ├── Dataset Loading - EDA.ipynb
    ├── Preproccessing - Baseline.ipynb
    ├── Feature Selection - Model Selection.ipynb
    ├── Hyperparameter Tuning - Final Model Test.ipynb
    └── model_artifacts/                  # created when the notebooks run
        ├── train_data.pkl
        ├── val_data.pkl
        ├── test_data.pkl
        ├── event_count_columns.pkl
        └── final_model/
            ├── lightgbm_final.joblib
            ├── final_config.json
            ├── test_metrics.csv
            ├── feature_importance.csv
            ├── validation_model_comparison.csv
            └── validation_test_comparison.csv
```

All notebooks are fully executed, so the plots, tables, and metrics are visible on GitHub without running anything.

---

## Dataset

The raw data is **not included** in this repository because `train.csv` is about 4.5 GB.

| File | Description |
|---|---|
| `train.csv` | 26,296,946 game events across 23,562 sessions (20 columns) |
| `train_labels.csv` | 424,116 labels: one row per session per question (18 questions) |
| `test.csv`, `sample_submission.csv` | Kaggle's small example test files (used only for format inspection) |

### Downloading the data

1. Create a Kaggle account and accept the competition rules on the [competition page](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/rules).
2. Set up the [Kaggle API](https://github.com/Kaggle/kaggle-api) with your `kaggle.json` token.
3. From the repository root, download and unzip the data into `tabular/`:

```bash
kaggle competitions download -c predict-student-performance-from-game-play -p tabular/
unzip tabular/predict-student-performance-from-game-play.zip -d tabular/
```

You can also download the files manually from the competition's **Data** tab and place them in `tabular/`.

---

## Environment Setup

The notebooks were run with **Python 3.11.9**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Main dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`, `joblib`, and `jupyter`.

### Set the data path

Notebooks 1, 3, and 4 use a hard-coded data directory:

```python
DATA_DIR = Path(r"C:\DATA 4380\Tabular Project")
```

Change `DATA_DIR` to the folder that holds the CSV files (for example `Path("tabular")` or `Path(".")` if you run Jupyter from inside `tabular/`). Notebook 2 already uses relative paths (`train.csv` and `model_artifacts/`), so run it from the folder that contains the data.

---

## How to Run

Run the notebooks **in order**. Each one depends on artifacts saved by the previous one.

| # | Notebook | Purpose | Output |
|---|---|---|---|
| 1 | `Dataset Loading - EDA.ipynb` | Explore the data | Plots and summary tables |
| 2 | `Preproccessing - Baseline.ipynb` | Build features, split the data, train baselines | `model_artifacts/*.pkl` |
| 3 | `Feature Selection - Model Selection.ipynb` | Compare feature sets and model families | Validation leaderboard |
| 4 | `Hyperparameter Tuning - Final Model Test.ipynb` | Tune LightGBM and evaluate on the test set | `model_artifacts/final_model/` |

Notebooks 1 and 2 read `train.csv` in chunks of 250,000 rows, so they run without loading the full 4.5 GB file into memory. Expect them to take several minutes.

---

## Approach

### 1. Exploratory Data Analysis

- **Target balance:** 70.6% of answers are correct and 29.4% are incorrect, so the classes are moderately imbalanced.
- **Question difficulty varies widely:** Question 2 is answered correctly 97.9% of the time, while Question 13 is answered correctly only 27.5% of the time. This made the question number an important feature.
- **Level groups:** The game has three level groups (`0-4`, `5-12`, `13-22`). Questions 1–3, 4–13, and 14–18 are asked at the end of each group.
- **Missing values are structural:** Columns such as `page`, `hover_duration`, and the coordinate fields are missing because they do not apply to certain event types (for example, `hover_duration` only exists for hover events). They are not random gaps.
- **Heavy skew:** `elapsed_time` and `hover_duration` have extreme outliers (skewness of 28.9 and 87.2), likely from idle or abandoned sessions.
- **Data integrity:** No duplicate rows were found, every session has all 18 labels, and the session IDs in the event log and the labels match exactly.

### 2. Preprocessing and Feature Engineering

- **Event-count features:** For each session and level group, the notebook counts how many times each of the 11 event types occurred (for example `navigate_click`, `person_click`, `object_click`). This turns millions of event rows into one compact row per session and level group.
- **Matching labels to gameplay:** Each question is joined only to the event counts from the level group that comes *before* it. This mirrors the competition setup, where the model only sees gameplay up to the point the question is asked, and it prevents leakage from later play.
- **Final feature set:** `question_number` (one-hot encoded) plus the 11 event counts, for 12 input features.

### 3. Train / Validation / Test Split

The data is split **by session**, not by row, into 70% train, 15% validation, and 15% test (`random_state=42`).

| Split | Sessions | Question rows |
|---|---|---|
| Train | 16,492 | 296,856 |
| Validation | 3,535 | 63,630 |
| Test | 3,535 | 63,630 |

Splitting by session keeps all 18 answers from one student in the same split. A row-level split would let the model see part of a student's answers during training and be tested on the rest, which would inflate the scores. The notebooks assert that no session appears in more than one split.

Kaggle's real test labels are hidden behind the competition's API, so this project uses its own held-out test split for the final evaluation.

### 4. Baselines

Validation results:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Majority class | 0.7041 | 0.7041 | 1.0000 | 0.8263 |
| Logistic regression (question only) | 0.7309 | 0.7447 | 0.9402 | 0.8311 |
| Logistic regression (question + event counts) | 0.7442 | 0.7583 | 0.9345 | 0.8373 |

The gameplay features add signal beyond question difficulty alone.

### 5. Feature and Model Selection

Three feature sets were compared:

- **Original:** question number and the 11 event counts (12 features)
- **Activity summary:** Original plus total events, event-type diversity, and event entropy (15 features)
- **Full engineered:** Activity summary plus the proportion of each event type (26 features)

Each set was tested with five models: majority class, logistic regression, random forest, LightGBM, and HistGradientBoosting.

- **LightGBM** had the best ROC-AUC (0.7654), with HistGradientBoosting close behind.
- The extra engineered features gave **no meaningful gain** (all LightGBM variants scored 0.7654), so the simpler **Original** feature set was kept.

ROC-AUC was used as the main selection metric because it measures ranking quality across all thresholds and is not dominated by the majority class the way accuracy is.

### 6. Class Imbalance and Threshold

- **Class weights:** Balanced class weights raised F1 on the incorrect class (0.5668 vs. 0.4368) but lowered accuracy (0.6810 vs. 0.7491). ROC-AUC was nearly identical (0.7656 vs. 0.7654). The **unweighted** model was kept because it ranked equally well and had higher overall accuracy.
- **Threshold:** Thresholds from 0.30 to 0.90 were tested. Accuracy peaked at the default **0.50**, so that threshold was kept. Raising it to about 0.70 would catch more incorrect answers at the cost of accuracy, if that trade-off were preferred.

### 7. Hyperparameter Tuning

A random search over 20 LightGBM configurations was scored on validation ROC-AUC. Best parameters:

| Parameter | Value |
|---|---|
| `n_estimators` | 700 |
| `learning_rate` | 0.03 |
| `num_leaves` | 47 |
| `max_depth` | 5 |
| `min_child_samples` | 200 |
| `subsample` | 0.85 |
| `colsample_bytree` | 1.0 |
| `reg_lambda` | 0.0 |

Tuning gave a small improvement in validation ROC-AUC (0.7654 → 0.7663). This suggests the model is limited more by the features than by its settings.

---

## Results

Final tuned LightGBM, threshold 0.50:

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC | F1 (incorrect) |
|---|---|---|---|---|---|---|
| Validation | 0.7490 | 0.7660 | 0.9266 | 0.8387 | 0.7663 | 0.4351 |
| **Test** | **0.7516** | **0.7667** | **0.9290** | **0.8401** | **0.7726** | **0.4438** |

The test scores are in line with the validation scores, so the model generalizes to unseen students without signs of overfitting.

**Most important features (by gain):** the indicators for Questions 13 and 18, followed by `object_click`, `map_click`, and `navigate_click` counts. Question identity carries the most signal, and how much a student explores and interacts with objects adds to it.

---

## Limitations and Future Work

- The model detects correct answers well but is weaker at catching incorrect ones (F1 of about 0.44 on that class).
- Only event *counts* are used. Timing (`elapsed_time`, time between events), text and room identifiers (`fqid`, `room_fqid`, `text_fqid`), and hover durations were not used and are likely to add signal.
- Training one model per question or per level group, as many top competition solutions did, could capture question-specific behavior better.
- Cross-validation grouped by session would give more stable estimates than a single validation split.

---

## Acknowledgments

Data from the Kaggle competition [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play), provided by The Learning Agency Lab and Field Day Lab.
