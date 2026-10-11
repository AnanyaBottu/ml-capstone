"""Fit each track's winning pipeline once and save it for the app.

The notebooks compare algorithms but never persist one, so the dashboard has
nothing to predict with. This refits only the winner per track, with the
hyperparameters the notebooks' searches landed on, and writes:

    models/<track>_pipeline.joblib   the fitted pipeline
    models/<track>_schema.json       form fields, levels, defaults, holdout score

Run: .venv/bin/python scripts/train_models.py        (~1-2 minutes)
"""
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import features  # noqa: E402

RANDOM_STATE = 42
DATA = ROOT / "data" / "hotel_bookings.csv"
OUT = ROOT / "models"

# The notebooks' winners, and whether that notebook's build() put a PCA step
# between the preprocessor and the model.
#
# Regression: tuned Hist Gradient Boosting, test R2 0.8233 (section 8). Its
# build() inserts PCA(0.95), so every regression model in that notebook is
# scored on components rather than the 99 encoded features. We keep the PCA
# so the app's predictions and the leaderboard describe the same model -
# dropping it scores R2 0.9043 instead, which is a finding for the notebook,
# not something to change silently here.
#
# Classification: Decision Tree, weighted F1 0.8213 (section 8). That
# notebook's build() has no PCA; it probed PCA and LDA separately.
WINNERS = {
    "regression": ("Hist Gradient Boosting (tuned)", HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=127,
        min_samples_leaf=20, l2_regularization=0.0, random_state=RANDOM_STATE),
        True),
    "classification": ("Decision Tree", DecisionTreeClassifier(
        max_depth=10, random_state=RANDOM_STATE), False),
}

PCA_VARIANCE = 0.95


def build_pipeline(X, estimator, use_pca):
    categorical = X.select_dtypes(include=["object", "str"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    steps = [("pre", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical),
        ("num", StandardScaler(), numeric),
    ]))]
    if use_pca:
        steps.append(("pca", PCA(n_components=PCA_VARIANCE,
                                 random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps), categorical, numeric


def schema(X, categorical, numeric, levels, track, model_name, scores):
    """Everything the form needs: which fields to ask for, their legal values,
    and a default so the user can change three things and hit Predict."""
    fields = {}
    for col in categorical:
        counts = X[col].value_counts()
        fields[col] = {"kind": "category",
                       "levels": sorted(counts.index.tolist()),
                       "default": counts.index[0]}
    for col in numeric:
        if col in features.ENGINEERED:
            continue  # derived from the raw answers, never asked
        s = X[col]
        integer = bool((s.dropna() % 1 == 0).all())
        fields[col] = {"kind": "integer" if integer else "number",
                       "min": float(s.min()), "max": float(s.max()),
                       "default": float(s.median())}
    return {
        "track": track,
        "target": features.TARGET[track],
        "model": model_name,
        "columns": X.columns.tolist(),
        "rare_levels": levels,
        "fields": fields,
        "engineered": features.ENGINEERED,
        "holdout": scores,
    }


def main():
    if not DATA.exists():
        print(f"missing {DATA} - run data/download.py first", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)
    df_raw = pd.read_csv(DATA)

    for track, (model_name, estimator, use_pca) in WINNERS.items():
        X, y, levels = features.prepare(df_raw, track)

        # Same split as the notebooks, so the score below is comparable:
        # ADR deciles for regression, the class itself for classification.
        strat = (pd.qcut(y, q=10, labels=False, duplicates="drop")
                 if track == "regression" else y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=strat)

        pipe, categorical, numeric = build_pipeline(X, estimator, use_pca)
        start = time.time()
        pipe.fit(X_train, y_train)
        elapsed = time.time() - start

        pred = pipe.predict(X_test)
        if track == "regression":
            scores = {"R2": r2_score(y_test, pred),
                      "RMSE": root_mean_squared_error(y_test, pred),
                      "MAE": mean_absolute_error(y_test, pred)}
        else:
            proba = pipe.predict_proba(X_test)[:, 1]
            scores = {"accuracy": accuracy_score(y_test, pred),
                      "f1_weighted": f1_score(y_test, pred, average="weighted"),
                      "roc_auc": roc_auc_score(y_test, proba)}
        scores = {k: round(v, 4) for k, v in scores.items()}

        # The threshold slider needs held-out probabilities, not the raw data.
        # Exporting them here keeps the deployed app off the 17 MB CSV.
        if track == "classification":
            pd.DataFrame({"is_canceled": y_test.to_numpy(),
                          "proba": proba}).to_csv(
                OUT / "classification_holdout.csv", index=False)

        joblib.dump(pipe, OUT / f"{track}_pipeline.joblib")
        (OUT / f"{track}_schema.json").write_text(json.dumps(
            schema(X, categorical, numeric, levels, track, model_name, scores),
            indent=2))

        print(f"{track:15s} {model_name}")
        print(f"                fit {elapsed:.0f}s on {len(X_train):,} rows  "
              f"{scores}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
