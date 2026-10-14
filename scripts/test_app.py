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

import pandas as pd  # noqa: E402
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

# the cost scatter uses a log x-axis, which dies on a non-positive fit time;
# Part B has no comparable timing, so only the timed rows are plotted
for name, df in [("regression", reg), ("classification", clf)]:
    timed = df["fit (s)"].dropna()
    assert len(timed) >= 5 and (timed > 0).all(), f"{name} fit times broken"
assert app.cost_scatter(clf.dropna(subset=["fit (s)"]), "f1_weighted").to_dict()

# Part A and Part B together, and not the tuned-candidates table that shares
# its columns (whose names carry "(Baseline)" / "(Tuned)")
assert len(clf) == 10, f"classification has {len(clf)} models, expected 10"
assert not clf.index.str.contains(r"\(").any(), clf.index.tolist()
assert app.load("classification_cv_b").index.tolist() == ["Bagging",
                                                          "Random Forest"]

# the winner picked for the stat row must be the actual best, not row order
assert reg["R2"].idxmax() == "Random Forest", reg["R2"].idxmax()
assert clf["f1_weighted"].idxmax() == "Bagging", clf["f1_weighted"].idxmax()

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


# ---- the sweep mechanics, across numeric and categorical fields ------------
# Responsiveness is only asserted for lead_time. A shallow tree genuinely can
# be flat in a given field for a given booking - the classification model
# never splits on hotel along the path a median booking takes - so demanding
# that every field move the prediction would be testing the data, not the code.
for track in ("regression", "classification"):
    pipe, schema = app.load_pipeline(track), app.load_schema(track)
    base = answers_from(schema)
    specs = {app.MONTH_FIELD: app.MONTH_SPEC,
             **{k: v for k, v in schema["fields"].items() if k != "has_agent"}}

    for column in ("lead_time", "hotel", app.MONTH_FIELD):
        spec = specs[column]
        values = app.sweep_values(spec)
        assert len(values) > 1, f"{column}: nothing to sweep"
        rows = pd.concat([app.build_row({**base, column: v}, schema)
                          for v in values], ignore_index=True)
        assert not rows.isna().any().any(), f"{track}/{column}: gaps in inputs"
        out = app.predict_values(pipe, rows, track)
        assert len(out) == len(values), f"{track}/{column}: wrong output length"
        assert pd.notna(out).all(), f"{track}/{column}: produced NaN"
        if column == "lead_time":
            assert len(set(out.round(6))) > 1, \
                f"{track}: sweeping lead_time never changes the prediction"
    print(f"ok: {track} sweeps run clean and respond to lead_time")



# ---- the threshold slider must reshape the confusion matrix ----------------
scored = app.holdout_probabilities("classification")
assert scored is not None, "no holdout probabilities (missing data or pipeline)"
y_true, proba = scored
assert len(y_true) == len(proba) and len(y_true) > 1000, len(y_true)
assert 0 <= proba.min() and proba.max() <= 1

import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, recall_score  # noqa: E402

shapes, recalls = [], []
for t in (0.2, 0.5, 0.8):
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    assert tn + fp + fn + tp == len(y_true), f"threshold {t}: counts lost rows"
    shapes.append((tn, fp, fn, tp))
    recalls.append(recall_score(y_true, pred, zero_division=0))

assert len(set(shapes)) == 3, f"threshold does not move the matrix: {shapes}"
# a stricter threshold can only catch fewer cancellations
assert recalls == sorted(recalls, reverse=True), recalls
assert app.confusion_heatmap(
    [("kept", "kept", 1), ("kept", "cancelled", 2),
     ("cancelled", "kept", 3), ("cancelled", "cancelled", 4)]).to_dict()
# ---- dark mode must build the same charts off the other surface ----------
app.PALETTE = app.DARK
app.SURFACE, app.INK, app.MUTED, app.GRID = (
    app.DARK[k] for k in ("surface", "ink", "muted", "grid"))
dark_bars = app.ranked_bars(reg, "R2", True).to_dict()
assert app.DARK["surface"] in str(dark_bars), "dark bars still on the light surface"
assert app.DARK["ramp"] in str(app.confusion_heatmap(
    [("kept", "kept", 1), ("kept", "cancelled", 2),
     ("cancelled", "kept", 3), ("cancelled", "cancelled", 4)]).to_dict()), \
    "confusion ramp did not follow the dark surface"
print("ok: both modes compile against their own surface")

print(f"ok: threshold moves recall {recalls[0]:.3f} -> {recalls[-1]:.3f} "
      f"over {len(y_true):,} held-out bookings")


# ---- clustering: every chart compiles, and the degenerate linkages are flagged
clu = app.cluster_models()
assert clu is not None and len(clu) == 5, "clustering_models missing"
assert clu["usable"].dtype == bool, "usable must be a real boolean"
assert clu.loc["K-Means", "usable"], "K-Means must count as usable"
assert clu.loc[["Agglomerative (average)", "Agglomerative (single)"],
               "usable"].eq(False).all(), "degenerate linkages not flagged"
for metric, higher in [("Silhouette", True), ("Davies-Bouldin", False),
                       ("Calinski-Harabasz", True)]:
    assert app.cluster_bars(clu, metric, higher).to_dict()
assert app.balance_bars(clu).to_dict()
kmeans, ward = app.load("clustering_kmeans_k"), app.load("clustering_ward_k")
assert kmeans is not None and ward is not None
assert "inertia" in kmeans and "inertia" not in ward, \
    "the Ward sweep was matched to the K-Means table"
for index in ("silhouette", "davies_bouldin", "calinski_harabasz"):
    assert app.k_lines(kmeans, ward, index).to_dict()
assert app.elbow(kmeans).to_dict()
assert app.K_CHOSEN in kmeans.index, "chosen k is not in the sweep"
agree = app.load("clustering_agreement")
assert (agree.to_numpy().diagonal() == 1).all(), "ARI diagonal must be 1"
assert app.agreement_heatmap(agree).to_dict()
profile = app.load("clustering_profile")
assert abs(profile["share_%"].sum() - 100) < 0.5, "segment shares do not sum to 100"
for feature, fmt in app.SEGMENT_FEATURES.items():
    assert feature in profile, f"profile lost column {feature}"
    assert app.segment_bars(profile, feature, fmt).to_dict()
print("ok: clustering tables load and every clustering chart compiles")


# ---- playground: merges mint new ids and end in one cluster ----------------
pts = app.seed_points()
for method in app.LINKAGES:
    owner, slots, z = app.merge_state(pts, method, 0)
    assert owner == list(range(len(pts))), "step 0 must leave every point alone"
    owner, slots, _ = app.merge_state(pts, method, 1)
    assert slots == {len(pts): 0}, "the first pair should take the first hue"
    assert len(set(owner)) == len(pts) - 1 and max(owner) == len(pts), owner
    owner, slots, _ = app.merge_state(pts, method, len(pts) - 1)
    assert len(set(owner)) == 1, f"{method}: did not end in one cluster"
    # the seeded blobs are well separated, so three clusters recover them
    owner, slots, _ = app.merge_state(pts, method, len(pts) - 3)
    assert set(slots) == set(owner) and len(set(slots.values())) == 3, slots
    # a hue follows its cluster: every step keeps the surviving bigger half's
    for step in range(1, len(pts)):
        before = app.merge_state(pts, method, step - 1)[1]
        after = app.merge_state(pts, method, step)[1]
        assert len(set(after.values())) == len(after), f"{method}: shared hue"
        assert all(after.get(c, v) == v for c, v in before.items()), \
            f"{method}: step {step} repainted a cluster it did not touch"
    groups = {tuple(i for i, o in enumerate(owner) if o == c) for c in owner}
    assert groups == {tuple(range(b, b + 6)) for b in (0, 6, 12)}, groups
assert app.playground_chart(pts, owner, slots, max(owner), 0).to_dict()
_, slots, z = app.merge_state(pts, "ward", 5)
tree = app.dendrogram_chart(z, 5, slots).to_dict()
rows = tree["datasets"][next(iter(tree["datasets"]))]
assert sum(r["latest"] for r in rows) == 3, "latest merge not marked"
done = {r["hue"] for r in rows if r["merge"] <= 5}
assert done == set(slots.values()), f"tree hues {done} != canvas {slots}"
# the k cut sits between the last merge done and the next one
for step in (0, 5, len(pts) - 2):
    spec = app.dendrogram_chart(z, step, {}).to_dict()
    cuts = [r["y"] for rows in spec["datasets"].values() for r in rows
            if set(r) == {"y"}]
    below = z[step - 1, 2] if step else 0
    assert len(cuts) == 1 and below < cuts[0] < z[step, 2], (step, cuts)
assert "layer" not in app.dendrogram_chart(z, len(pts) - 1, {}).to_dict(), \
    "k = 1 should draw no cut"
print("ok: playground merges step cleanly for every linkage")

print("ok: all checks passed")
