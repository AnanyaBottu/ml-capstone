"""Fails if a chart spec stops compiling or a harvested column goes missing.

Run: .venv/bin/python scripts/test_charts.py
Importing app.py also executes the dashboard body in bare mode, which is
harmless but chatty, hence the logging squelch.
"""
import logging
import sys
from pathlib import Path

logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

reg = app.load("regression_models")
cv = app.load("regression_cv")
tuning = app.load("regression_tuning")
clf = app.load("classification_models").join(
    app.load("classification_timing")[["fit (s)", "train rows"]])
clf_cv = app.load("classification_cv")
repr_df = app.load("classification_repr")

for name, df in [("regression", reg), ("regression_cv", cv),
                 ("tuning", tuning), ("classification", clf),
                 ("classification_cv", clf_cv), ("repr", repr_df)]:
    assert df is not None and not df.empty, f"{name} table missing or empty"

for metric, higher in [("R2", True), ("RMSE", False), ("MAE", False)]:
    assert app.ranked_bars(reg, metric, higher).to_dict()
for metric in ["f1_weighted", "accuracy", "roc_auc", "precision_weighted",
               "recall_weighted"]:
    assert app.ranked_bars(clf, metric, True).to_dict()
assert app.ranked_bars(repr_df, "f1_weighted", True).to_dict()

assert app.paired_bars(reg, "train R2", "R2", "R2", ("train", "test")).to_dict()
assert app.paired_bars(tuning, "default R2", "tuned R2", "Test R2").to_dict()
assert app.cost_scatter(reg, "R2").to_dict()
assert app.cost_scatter(clf, "f1_weighted").to_dict()
assert app.cv_bars(cv, "CV mean R2", "CV std").to_dict()
assert app.cv_bars(clf_cv, "CV mean F1", "CV std").to_dict()

# the cost scatter uses a log x-axis, which dies on a non-positive fit time
for name, df in [("regression", reg), ("classification", clf)]:
    assert (df["fit (s)"] > 0).all(), f"{name} has a non-positive fit time"

# the winner picked for the stat row must be the actual best, not row order
assert reg["R2"].idxmax() == "Random Forest", reg["R2"].idxmax()
assert clf["f1_weighted"].idxmax() == "Decision Tree", clf["f1_weighted"].idxmax()

print(f"ok: {len(reg)} regression + {len(clf)} classification models, "
      "all chart specs compile")


# ---- the prediction path: a filled form must reach the model unchanged ------
def answers_from(schema):
    """What the form produces when nothing is touched: every default."""
    out = {app.MONTH_FIELD: app.MONTH_SPEC["default"]}
    for column, spec in schema["fields"].items():
        if column == "has_agent":
            continue  # derived in build_row
        out[column] = (spec["default"] if spec["kind"] == "category"
                       else (int if spec["kind"] == "integer" else float)(spec["default"]))
    return out


for track in ("regression", "classification"):
    pipe, schema = app.load_pipeline(track), app.load_schema(track)
    assert pipe is not None, f"{track}: no fitted pipeline, run train_models.py"
    assert schema is not None, f"{track}: no schema"

    row = app.build_row(answers_from(schema), schema)
    assert list(row.columns) == schema["columns"], f"{track}: column mismatch"
    missing = row.columns[row.isna().any()].tolist()
    assert not missing, f"{track}: empty model inputs {missing}"

    # every engineered feature must actually have been computed
    for column in schema["engineered"]:
        assert column in row.columns and row[column].notna().all(), \
            f"{track}: {column} not engineered"

    if track == "regression":
        adr = float(pipe.predict(row)[0])
        assert 0 <= adr <= 1000, f"ADR out of the trained range: {adr}"
    else:
        risk = float(pipe.predict_proba(row)[:, 1][0])
        assert 0 <= risk <= 1, risk

    # a changed answer must move the prediction, or the form is decorative
    low, high = answers_from(schema), answers_from(schema)
    low["lead_time"], high["lead_time"] = 0, 300
    predict = ((lambda r: pipe.predict(r)[0]) if track == "regression"
               else (lambda r: pipe.predict_proba(r)[:, 1][0]))
    a = float(predict(app.build_row(low, schema)))
    b = float(predict(app.build_row(high, schema)))
    assert a != b, f"{track}: lead_time 0 and 300 both predict {a}"
    print(f"ok: {track} responds to input ({a:.4f} -> {b:.4f} over lead_time)")

print("ok: prediction path intact for both tracks")
