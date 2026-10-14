# Hotel Booking ML Capstone

23CSE301 Machine Learning capstone, Team 21. One dataset, three tracks: predict a booking's nightly rate (regression), predict whether it will be cancelled (classification), and find customer segments without labels (clustering). A Streamlit dashboard serves the trained models.

| Member | Roll no. | Primary track |
|---|---|---|
| Ananya Bottu | CB.SC.U4CSE24408 | Regression, dashboard |
| Bhamini V | CB.SC.U4CSE24467 | Classification |
| Neha Tutika | CB.SC.U4CSE24456 | Clustering |

## Problem statement

Hotels lose revenue in two ways: pricing a room wrongly and holding a room for a guest who cancels. Using only what is known at the moment a booking is made, we want to:

1. **Regression:** predict `adr`, the average daily rate of a booking in EUR.
2. **Classification:** predict `is_canceled`, whether the booking will be cancelled.
3. **Clustering:** group bookings into customer segments with no labels, then check whether the segments mean anything (for example, whether they differ in cancellation rate).

## Dataset

**Hotel Booking Demand** - Antonio, Almeida & Nunes (2019), *Hotel booking demand datasets*, Data in Brief 22:41-49. Bookings for a resort hotel and a city hotel in Portugal, July 2015 to August 2017.

- 119,390 rows x 32 columns; 85,693 rows after cleaning
- Mix of numeric (lead time, stay length, guests, previous cancellations, special requests) and categorical (hotel, month, meal, country, market segment, channel, room type, deposit type, customer type) columns
- Targets: `adr` (continuous, median about EUR 95) and `is_canceled` (63% / 37% split)

Cleaning decisions shared by all three notebooks (see `features.py`):

- **Leakage removed:** `reservation_status` reproduces `is_canceled` exactly, `reservation_status_date` carries the same information, and `assigned_room_type` is only known at check-in. All three are dropped.
- **Missing values:** `company` is 94% missing, so it becomes a `has_company` flag; a blank `agent` means "no agent" and becomes its own level plus a `has_agent` flag; `children` is filled with 0; `country` with "Unknown"; `meal = "Undefined"` is merged with "SC" (both mean no meal).
- **Invalid rows:** negative and extreme (> EUR 1,000) rates, zero-guest bookings, and 33,515 exact duplicates are dropped.
- **High cardinality:** `country` keeps its top 20 levels and `agent` its top 15; the rest become "Other".
- **Engineered features:** total guests, family flag, total nights, weekend ratio, month number, cyclical week-of-year (sin/cos), previous bookings and previous cancellation rate.

All encoders and scalers are fitted on the training split only, inside a scikit-learn `Pipeline`. Splits are 80:20 with `random_state=42` (stratified for classification).

## Results

### Regression - target `adr` (10 algorithms, same test split, ranked by R2)

| Model | R2 | RMSE (EUR) | MAE (EUR) |
|---|---|---|---|
| Random Forest | 0.7806 | 24.28 | 15.51 |
| Hist Gradient Boosting | 0.7554 | 25.64 | 17.56 |
| K-Nearest Neighbours | 0.7201 | 27.43 | 18.30 |
| Support Vector Regression | 0.6940 | 28.67 | 19.09 |
| Decision Tree | 0.5831 | 33.47 | 22.19 |
| Linear Regression | 0.5375 | 35.25 | 25.73 |
| Ridge | 0.5375 | 35.25 | 25.73 |
| Lasso | 0.5372 | 35.26 | 25.71 |
| ElasticNet | 0.5329 | 35.43 | 25.74 |
| Polynomial Ridge | 0.5149 | 36.10 | 26.72 |

Hyperparameter tuning (GridSearchCV / RandomizedSearchCV):

| Model | Default R2 | Tuned R2 | Tuned RMSE | Tuned MAE |
|---|---|---|---|---|
| Hist Gradient Boosting | 0.7554 | **0.8233** | 21.79 | 14.02 |
| Random Forest | 0.7806 | 0.7786 | 24.39 | 15.62 |

5-fold CV R2 for the two best models: Random Forest 0.7666 (std 0.0064), Hist Gradient Boosting 0.7520 (std 0.0028).

**Final model: tuned Hist Gradient Boosting**, R2 0.823, RMSE EUR 21.79, MAE EUR 14.02. Tree ensembles win because price depends on interactions (hotel x season x room type) that no linear model can express. Every model in this notebook is scored after a PCA(0.95) step; the notebook shows PCA cost about 0.11 of R2, which tuning only partly recovers.

### Classification - target `is_canceled` (all 10 algorithms, ranked by weighted F1)

| Model | Part | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Bagging | B | 0.8488 | 0.8452 | 0.8488 | **0.8462** | **0.9102** |
| Random Forest | B | **0.8494** | 0.8449 | 0.8494 | 0.8452 | 0.9089 |
| MLP | B | 0.8351 | 0.8306 | 0.8351 | 0.8318 | 0.9005 |
| Decision Tree | A | 0.8271 | 0.8207 | 0.8271 | 0.8213 | 0.8902 |
| Gradient Boosting | B | 0.8289 | 0.8223 | 0.8289 | 0.8206 | 0.8924 |
| Support Vector Machine | A | 0.8203 | 0.8127 | 0.8203 | 0.8107 | 0.8673 |
| K-Nearest Neighbors | A | 0.8035 | 0.7950 | 0.8035 | 0.7966 | 0.8443 |
| Logistic Regression | A | 0.7989 | 0.7885 | 0.7989 | 0.7889 | 0.8562 |
| AdaBoost | B | 0.8031 | 0.7936 | 0.8031 | 0.7878 | 0.8581 |
| Gaussian Naive Bayes | A | 0.6502 | 0.7557 | 0.6502 | 0.6686 | 0.7641 |

Hyperparameter tuning (GridSearchCV, 3-fold, weighted F1):

| Model | Baseline F1 | Tuned F1 | Best parameters |
|---|---|---|---|
| Random Forest | 0.8452 | 0.8447 | `n_estimators=200` |
| AdaBoost | 0.7878 | 0.7948 | `n_estimators=200, learning_rate=1.0` |
| MLP | 0.8318 | 0.8379 | `hidden_layer_sizes=(50,), activation=relu` |

Cross-validated weighted F1 for the two best models: Bagging 0.8398 (std 0.0004), Random Forest 0.8386 (std 0.0003).

**Final model: Bagging** (decision-tree base estimator). It has the highest weighted F1 and ROC-AUC on the test set and the highest mean CV F1. Random Forest is within 0.001 on every metric and makes fewer false positives, so it is the better choice if wrongly flagging a booking as a cancellation is the costlier mistake.

### Clustering - customer segments (no labels used in fitting)

| Model | k | Silhouette | Davies-Bouldin | Calinski-Harabasz | Smallest cluster |
|---|---|---|---|---|---|
| Agglomerative (ward) | 4 | 0.221 | 1.880 | 907.7 | 4.0% |
| **K-Means** | 4 | 0.220 | 1.888 | 946.4 | 4.2% |
| Agglomerative (complete) | 4 | 0.164 | 1.483 | 388.2 | 0.8% |
| Agglomerative (average) | 4 | 0.441 | 0.726 | 149.6 | 0.01% (degenerate) |
| Agglomerative (single) | 4 | 0.400 | 0.521 | 9.7 | 0.01% (degenerate) |

k = 4 was chosen from the elbow curve, the dendrogram and the silhouette plateau. Average and single linkage score well on silhouette only by putting 99% of bookings in one cluster, so they are excluded. K-Means and Ward agree at ARI 0.905; **K-Means is the final model** because it is 10x faster and can assign new bookings. It holds up on 17,139 unseen bookings (silhouette 0.223 vs 0.224 on train).

| Segment | Share | Lead time (days) | ADR (EUR) | Cancel rate |
|---|---|---|---|---|
| Mainstream online bookers | 74% | 61 | 99 | 30% |
| Families | 10% | 64 | 152 | 35% |
| Repeat corporate guests | 4% | 3 | 65 | 8% |
| Direct last-minute bookers | 11% | 5 | 70 | 14% |

The clusters never saw `is_canceled`, yet they separate cancellation risk from 8% to 35%. PCA (2 components) and t-SNE plots are in the notebook.

## Repository structure

```
README.md
requirements.txt          dashboard dependencies (pinned)
requirements-dev.txt      notebook and script dependencies (pinned)
data/download.py          downloads hotel_bookings.csv
notebooks/
  regression.ipynb
  classification.ipynb    Part A (sections 1-10) and Part B (sections 11-15)
  clustering.ipynb
features.py               cleaning and feature engineering shared by the scripts and the app
models/                   saved pipelines (.joblib), their input schemas, and metric tables
scripts/
  train_models.py         refits each track's winner and saves it to models/
  harvest_metrics.py      extracts the comparison tables from notebook outputs into models/*.csv
  test_app.py             checks the dashboard's charts and prediction path
app.py                    Streamlit dashboard
docs/                     capstone guidelines and dataset notes
```

## Environment setup

Python 3.12 (the version the pinned requirements were tested with).

```bash
git clone https://github.com/AnanyaBottu/ml-capstone.git
cd ml-capstone
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

## How to run

1. **Get the data** (saves `data/hotel_bookings.csv`):
   ```bash
   python data/download.py
   ```
2. **Run the notebooks** top to bottom in `regression.ipynb`, `classification.ipynb`, `clustering.ipynb`:
   ```bash
   jupyter lab notebooks/
   ```
   SVC and SVR are fitted on a fixed 15,000-row training subsample, and the clustering comparison on an 8,000-row working sample (`random_state=42`), because these algorithms scale as O(n^2) or worse.
3. **Launch the dashboard.** The trained pipelines are already committed in `models/`, so this works straight after step 1:
   ```bash
   streamlit run app.py
   ```
   The dashboard has one tab per track: enter a booking to get a predicted rate or cancellation probability, and compare how every algorithm scored.
4. **Optional, rebuild the artefacts** after changing a notebook:
   ```bash
   python scripts/harvest_metrics.py   # refresh models/*.csv from notebook outputs
   python scripts/train_models.py      # refit the saved pipelines (1-2 minutes)
   python scripts/test_app.py          # sanity-check the dashboard
   ```

## AI assistance

Generative AI was used for code scaffolding, the Streamlit dashboard and this README. All analysis, interpretation and feature-engineering decisions in the notebooks are the team's own.

## License

See [LICENSE](LICENSE).
