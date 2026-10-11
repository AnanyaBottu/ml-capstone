"""Dashboard for the three capstone tracks: enter a booking and get a
prediction, or compare how the algorithms scored.

Metric tables come from scripts/harvest_metrics.py (scraped out of the
notebook outputs); the fitted pipelines come from scripts/train_models.py.
Nothing is fitted here, so every control is instant.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import features

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e8e5"

# Algorithms are coloured by family, not by rank, so a model keeps its colour
# when the ranking metric changes or the selection narrows. Three families is
# the ceiling: a fourth hue fails the all-pairs colourblind gate against one of
# these three, and the scatter below needs every pair to be distinguishable.
FAMILY = {
    "Linear Regression": "Linear & probabilistic",
    "Ridge": "Linear & probabilistic",
    "Lasso": "Linear & probabilistic",
    "ElasticNet": "Linear & probabilistic",
    "Polynomial Ridge": "Linear & probabilistic",
    "Logistic Regression": "Linear & probabilistic",
    "Gaussian Naive Bayes": "Linear & probabilistic",
    "Decision Tree": "Tree & ensemble",
    "Random Forest": "Tree & ensemble",
    "Hist Gradient Boosting": "Tree & ensemble",
    # the two notebooks spell the neighbours model differently
    "K-Nearest Neighbours": "Instance & kernel",
    "K-Nearest Neighbors": "Instance & kernel",
    "Support Vector Regression": "Instance & kernel",
    "Support Vector Machine": "Instance & kernel",
}
FAMILY_ORDER = ["Linear & probabilistic", "Tree & ensemble",
                "Instance & kernel", "Other"]
FAMILY_RANGE = [BLUE, ORANGE, AQUA, MUTED]


def family_color():
    return alt.Color("family:N", title=None,
                     scale=alt.Scale(domain=FAMILY_ORDER, range=FAMILY_RANGE),
                     legend=alt.Legend(orient="top", direction="horizontal"))

MODELS = Path(__file__).parent / "models"

st.set_page_config(page_title="Capstone Model Comparison", page_icon="📊",
                   layout="wide")


@st.cache_data
def load(name):
    path = MODELS / f"{name}.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


@st.cache_resource
def load_pipeline(track):
    # joblib unpickles, so this only ever reads models/ written by
    # scripts/train_models.py in this repo - never an uploaded or fetched file.
    path = MODELS / f"{track}_pipeline.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_schema(track):
    path = MODELS / f"{track}_schema.json"
    return json.loads(path.read_text()) if path.exists() else None


def style(chart, height):
    return (chart.properties(height=height, width="container")
            .configure_view(strokeWidth=0, fill=SURFACE)
            .configure_axis(grid=True, gridColor=GRID, gridWidth=1,
                            domainColor=GRID, tickColor=GRID,
                            labelColor=MUTED, titleColor=MUTED,
                            labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=MUTED, titleColor=MUTED, orient="top",
                              direction="horizontal", title=None)
            .configure_title(color=INK, fontSize=14, anchor="start"))


def tidy(df):
    """Index to a `model` column, plus the family that drives its colour."""
    d = df.reset_index().rename(columns={df.index.name or "index": "model"})
    d["family"] = d["model"].map(FAMILY).fillna("Other")
    return d


def ranked_bars(df, metric, higher_is_better, fmt=".3f", unit="",
                by_family=True):
    """One metric, every model, sorted best to worst, coloured by family.
    Each bar is directly labelled, which is also the relief the aqua family
    needs to clear the contrast gate on a light surface."""
    d = tidy(df)
    base = alt.Chart(d).encode(
        x=alt.X(f"{metric}:Q", title=f"{metric}{unit}",
                scale=alt.Scale(nice=True)),
        y=alt.Y("model:N", sort=alt.EncodingSortField(metric, order="descending"
                if higher_is_better else "ascending"), title=None),
        tooltip=[alt.Tooltip("model:N", title="Model"),
                 alt.Tooltip("family:N", title="Family"),
                 alt.Tooltip(f"{metric}:Q", format=fmt)],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=18)
    bars = (bars.encode(color=family_color()) if by_family
            else bars.encode(color=alt.value(BLUE)))
    labels = base.mark_text(align="left", dx=6, color=MUTED, fontSize=11).encode(
        text=alt.Text(f"{metric}:Q", format=fmt))
    return style(bars + labels, 30 * len(d) + 70)


def paired_bars(df, left, right, value_title, names=("default", "tuned")):
    """Two series on one shared scale. Legend plus direct labels, so the
    pairing never rests on colour alone."""
    d = (df[[left, right]].rename(columns=dict(zip([left, right], names)))
         .reset_index().melt(id_vars=df.index.name or "index",
                             var_name="setting", value_name="value")
         .rename(columns={df.index.name or "index": "model"}))
    base = alt.Chart(d).encode(
        x=alt.X("value:Q", title=value_title),
        y=alt.Y("model:N", title=None),
        yOffset=alt.YOffset("setting:N", sort=list(names)),
        color=alt.Color("setting:N", sort=list(names),
                        scale=alt.Scale(domain=list(names),
                                        range=[BLUE, ORANGE])),
        tooltip=["model:N", "setting:N", alt.Tooltip("value:Q", format=".4f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=3, height=14)
    labels = base.mark_text(align="left", dx=5, color=MUTED, fontSize=10).encode(
        text=alt.Text("value:Q", format=".3f"))
    return style(bars + labels, 46 * d["model"].nunique() + 80)


def cost_scatter(df, metric, time_col="fit (s)"):
    """Score against what it cost to fit, coloured by family. Three hues, not
    ten: every pair in a scatter has to be distinguishable, and the all-pairs
    colourblind gate only clears at three."""
    d = tidy(df)
    base = alt.Chart(d).encode(
        x=alt.X(f"{time_col}:Q", title="Fit time (seconds, log scale)",
                scale=alt.Scale(type="log", nice=False)),
        y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
        tooltip=["model:N", alt.Tooltip("family:N", title="Family"),
                 alt.Tooltip(f"{metric}:Q", format=".4f"),
                 alt.Tooltip(f"{time_col}:Q", format=".2f", title="fit (s)")],
    )
    points = base.mark_point(filled=True, size=160, stroke=SURFACE,
                             strokeWidth=2).encode(color=family_color())
    labels = base.mark_text(align="left", dx=10, dy=-6, color=MUTED,
                            fontSize=11).encode(text="model:N")
    return style(points + labels, 420)


def cv_bars(df, mean_col, std_col):
    """Cross-validated mean with a +/- 2 sd interval, so a lead inside the
    noise is visible as one."""
    d = tidy(df)
    d["lo"] = d[mean_col] - 2 * d[std_col]
    d["hi"] = d[mean_col] + 2 * d[std_col]
    y = alt.Y("model:N", sort=alt.EncodingSortField(mean_col, order="descending"),
              title=None)
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X(f"{mean_col}:Q", title=f"{mean_col} (bars show +/- 2 sd)"),
        y=y, color=family_color(),
        tooltip=["model:N", alt.Tooltip("family:N", title="Family"),
                 alt.Tooltip(f"{mean_col}:Q", format=".4f"),
                 alt.Tooltip(f"{std_col}:Q", format=".4f", title="sd")])
    rule = alt.Chart(d).mark_rule(color=INK, strokeWidth=1.5).encode(
        x="lo:Q", x2="hi:Q", y=y)
    return style(bars + rule, 30 * len(d) + 70)


def leaderboard(df, best_col, higher_is_better, headline, fmt="{:.4f}"):
    """Winner as a stat row: the headline is a number, not a chart. The
    ranking metric also shows the gap to the runner-up, which is the thing
    that tells you whether the lead is real."""
    ranked = df[best_col].sort_values(ascending=not higher_is_better)
    best = ranked.index[0]
    runner = ranked.index[1] if len(ranked) > 1 else None
    row = df.loc[best]

    family = FAMILY.get(best, "Other")
    swatch = FAMILY_RANGE[FAMILY_ORDER.index(family)]
    st.markdown(
        f'<div style="font-size:.8rem;color:{MUTED};letter-spacing:.04em;'
        f'text-transform:uppercase;">Best on {best_col}</div>'
        f'<div style="font-size:1.55rem;font-weight:700;margin:.1rem 0 .1rem;">'
        f'{best}</div>'
        f'<div style="font-size:.85rem;color:{MUTED};">'
        f'<span style="display:inline-block;width:.6rem;height:.6rem;'
        f'border-radius:50%;background:{swatch};margin-right:.4rem;"></span>'
        f'{family}</div>', unsafe_allow_html=True)
    st.write("")

    cols = st.columns(len(headline))
    for col, name in zip(cols, headline):
        if name not in row:
            col.metric(name, "n/a")
            continue
        delta = None
        if runner is not None and name == best_col:
            delta = f"{row[name] - df.loc[runner, name]:+.4f} vs {runner}"
        col.metric(name, fmt.format(row[name]), delta=delta)


def table_view(df, label="Full metric table"):
    with st.expander(label):
        st.dataframe(df, width="stretch")


def pick_models(df, key):
    """Which algorithms the charts below should show. Returns None when the
    user has cleared the lot, so callers can skip rendering."""
    names = df.index.tolist()
    chosen = st.multiselect("Algorithms on show", names, default=names,
                            key=key, help="Clear a few to compare a subset "
                            "without the weak models flattening the scale.")
    if not chosen:
        st.info("Pick at least one algorithm.")
        return None
    return df.loc[[n for n in names if n in chosen]]


def restrict(df, chosen):
    """Narrow a secondary table (CV, tuning) to the chosen algorithms. Those
    tables cover fewer models than the leaderboard, so an empty overlap is
    normal rather than an error."""
    if df is None:
        return None
    keep = [i for i in df.index if i in chosen]
    return df.loc[keep] if keep else None


# ------------------------------------------------------------- prediction form
# arrival_date_month is consumed by features.engineer() and dropped there, so
# it is not in the saved column list and has to be asked for separately.
MONTH_FIELD = "arrival_date_month"
MONTH_SPEC = {"kind": "category", "levels": features.MONTHS,
              "default": "August"}

FIELD_GROUPS = {
    "Stay": ["hotel", "arrival_date_year", MONTH_FIELD,
             "arrival_date_week_number", "arrival_date_day_of_month",
             "stays_in_weekend_nights", "stays_in_week_nights",
             "reserved_room_type", "meal"],
    "Guests": ["adults", "children", "babies", "country", "is_repeated_guest",
               "required_car_parking_spaces", "total_of_special_requests"],
    "Booking": ["lead_time", "market_segment", "distribution_channel",
                "deposit_type", "customer_type", "agent", "has_company",
                "booking_changes", "days_in_waiting_list", "adr"],
    "Guest history": ["previous_cancellations",
                      "previous_bookings_not_canceled"],
}


def widget(column, spec, key):
    label = column.replace("_", " ").capitalize()
    if spec["kind"] == "category":
        levels = spec["levels"]
        default = spec["default"] if spec["default"] in levels else levels[0]
        return st.selectbox(label, levels, index=levels.index(default), key=key)
    # a 0/1 integer is a flag, and a checkbox reads better than a spinner
    if spec["kind"] == "integer" and spec["min"] == 0 and spec["max"] == 1:
        return int(st.checkbox(label, value=bool(spec["default"]), key=key))
    cast = int if spec["kind"] == "integer" else float
    return cast(st.number_input(
        label, min_value=cast(spec["min"]), max_value=cast(spec["max"]),
        value=cast(spec["default"]), step=cast(1), key=key))


def build_row(answers, schema):
    """One answered form to one model-ready row, through the same functions
    the training script used."""
    row = pd.DataFrame([answers])
    # has_agent is the presence of an agent id, so it follows from that answer
    row["has_agent"] = int(row.at[0, "agent"] != "0")
    row = features.engineer(row)
    row = features.apply_rare_levels(row, schema["rare_levels"])
    return row.reindex(columns=schema["columns"])


def predict_form(track, title, caption):
    pipe, schema = load_pipeline(track), load_schema(track)
    if pipe is None or schema is None:
        st.info("No fitted pipeline yet. Run "
                "`.venv/bin/python scripts/train_models.py` (about a minute), "
                "then reload.")
        return

    st.subheader(title)
    st.caption(f"{caption} Predictions come from **{schema['model']}** - "
               + ", ".join(f"holdout {k} {v}"
                           for k, v in schema["holdout"].items()) + ".")

    specs = {MONTH_FIELD: MONTH_SPEC,
             **{k: v for k, v in schema["fields"].items() if k != "has_agent"}}
    grouped = {c for cols in FIELD_GROUPS.values() for c in cols}
    groups = {**FIELD_GROUPS,
              "Other": [c for c in specs if c not in grouped]}

    with st.form(f"{track}_form"):
        st.caption("Every field starts at the training-set median or most "
                   "common value, so change what you care about and submit.")
        answers = {}
        for group, columns in groups.items():
            present = [c for c in columns if c in specs]
            if not present:
                continue
            st.markdown(f"**{group}**")
            slots = st.columns(3)
            for i, column in enumerate(present):
                with slots[i % 3]:
                    answers[column] = widget(column, specs[column],
                                             f"{track}_{column}")
        submitted = st.form_submit_button("Predict", type="primary")

    # Keeping the submitted answers in session state is what lets the sweep
    # control below rerun without wiping the prediction.
    state_key = f"{track}_answers"
    if submitted:
        st.session_state[state_key] = answers
    stored = st.session_state.get(state_key)
    if stored is None:
        return

    row = build_row(stored, schema)
    blank = row.columns[row.isna().any()].tolist()
    if blank:
        st.error(f"These model inputs came out empty, so the prediction would "
                 f"be meaningless: {blank}. The form and the saved pipeline "
                 f"have drifted - rerun scripts/train_models.py.")
        return

    if track == "regression":
        adr = float(predict_values(pipe, row, track)[0])
        mae = schema["holdout"]["MAE"]
        left, right = st.columns([1, 2])
        left.metric("Predicted ADR", f"€{adr:,.2f}")
        right.caption(f"Typical error on held-out bookings is €{mae:.2f}, so "
                      f"read this as roughly €{adr - mae:,.0f} to "
                      f"€{adr + mae:,.0f} rather than an exact rate.")
    else:
        risk = float(predict_values(pipe, row, track)[0])
        left, right = st.columns([1, 2])
        left.metric("Cancellation risk", f"{risk * 100:.1f}%")
        with right:
            st.progress(risk)
            st.caption(f"Classed as **{'cancelled' if risk >= 0.5 else 'kept'}**"
                       " at the usual 0.5 cut-off. The dataset's base rate is "
                       "37%, so compare against that, not against zero.")

    sweep_section(track, stored, schema, pipe, specs)


def predict_values(pipe, rows, track):
    return (pipe.predict(rows) if track == "regression"
            else pipe.predict_proba(rows)[:, 1])


def sweep_values(spec):
    """The values to try for one field: every level, or 40 steps across the
    range the model was trained on."""
    if spec["kind"] == "category":
        return spec["levels"]
    steps = np.linspace(spec["min"], spec["max"], 40)
    if spec["kind"] == "integer":
        return np.unique(steps.round().astype(int)).tolist()
    return steps.tolist()


def sweep_section(track, answers, schema, pipe, specs):
    """Hold the booking fixed, move one field across its range, and plot what
    the model does. Turns a single prediction into something you can poke at."""
    st.markdown("**Move one field and watch the prediction**")
    sweepable = [c for c in specs if c in answers]
    column = st.selectbox("Field to vary", sorted(sweepable),
                          format_func=lambda c: c.replace("_", " ").capitalize(),
                          key=f"{track}_sweep")
    spec = specs[column]

    values = sweep_values(spec)
    rows = pd.concat([build_row({**answers, column: v}, schema) for v in values],
                     ignore_index=True)
    if rows.isna().any().any():
        st.caption("Cannot sweep this field without leaving gaps in the "
                   "model's inputs.")
        return

    is_adr = track == "regression"
    label = "Predicted ADR (EUR)" if is_adr else "Cancellation risk"
    fmt = ".2f" if is_adr else ".1%"
    d = pd.DataFrame({column: values,
                      label: predict_values(pipe, rows, track)})
    here = answers[column]

    y = alt.Y(f"{label}:Q", title=label, scale=alt.Scale(zero=False))
    tooltip = [alt.Tooltip(f"{column}:N" if spec["kind"] == "category"
                           else f"{column}:Q"),
               alt.Tooltip(f"{label}:Q", format=fmt)]

    if spec["kind"] == "category":
        # opacity, not a second hue, marks the level the user actually chose
        chart = alt.Chart(d).mark_bar(cornerRadiusEnd=4, color=BLUE).encode(
            x=alt.X(f"{column}:N", title=None, sort="-y",
                    axis=alt.Axis(labelAngle=-40)),
            y=y,
            opacity=alt.condition(alt.datum[column] == here, alt.value(1.0),
                                  alt.value(0.45)),
            tooltip=tooltip)
    else:
        line = alt.Chart(d).mark_line(color=BLUE, strokeWidth=2).encode(
            x=alt.X(f"{column}:Q", title=column.replace("_", " ")), y=y,
            tooltip=tooltip)
        mark = (alt.Chart(pd.DataFrame({column: [here]}))
                .mark_rule(color=ORANGE, strokeWidth=2, strokeDash=[])
                .encode(x=f"{column}:Q"))
        chart = line + mark
        st.caption(f"The orange rule is your value ({here}). Everything else "
                   "in the booking is held where you set it.")

    st.altair_chart(style(chart, 320))


# --------------------------------------------------------- operating point
@st.cache_data
def holdout_probabilities(track):
    """True labels and predicted probabilities on the test split, exported by
    scripts/train_models.py. Reading them back means the deployed app never
    needs the 17 MB source CSV, and the slider never rescores anything."""
    path = MODELS / f"{track}_holdout.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    return d.iloc[:, 0].to_numpy(), d["proba"].to_numpy()


def confusion_heatmap(counts):
    """2x2 counts on one sequential blue ramp, every cell labelled."""
    d = pd.DataFrame(counts, columns=["actual", "predicted", "bookings"])
    biggest = d["bookings"].max()
    base = alt.Chart(d).encode(
        x=alt.X("predicted:N", title="Predicted",
                sort=["kept", "cancelled"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("actual:N", title="Actual", sort=["kept", "cancelled"]))
    cells = base.mark_rect(stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color("bookings:Q", title="Bookings",
                        scale=alt.Scale(scheme="blues")),
        tooltip=["actual:N", "predicted:N", alt.Tooltip("bookings:Q", format=",")])
    labels = base.mark_text(fontSize=15, fontWeight="bold").encode(
        text=alt.Text("bookings:Q", format=","),
        # dark cells need light ink, so the count stays readable either way
        color=alt.condition(alt.datum.bookings > biggest / 2,
                            alt.value(SURFACE), alt.value(INK)))
    return style(cells + labels, 300)


def operating_point():
    scored = holdout_probabilities("classification")
    if scored is None:
        st.info("No held-out scores yet. Run "
                "`.venv/bin/python scripts/train_models.py`, then reload.")
        return
    y_true, proba = scored

    st.subheader("Choose an operating point")
    st.caption(f"Every threshold below is scored on the same "
               f"{len(y_true):,} held-out bookings. Lower it to catch more "
               "cancellations at the cost of flagging bookings that would "
               "have been kept.")
    threshold = st.slider("Flag as cancelled when the probability is at least",
                          0.05, 0.95, 0.50, 0.01, key="clf_threshold")
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    cells = [("kept", "kept", int(tn)), ("kept", "cancelled", int(fp)),
             ("cancelled", "kept", int(fn)), ("cancelled", "cancelled", int(tp))]

    left, right = st.columns([2, 3])
    with left:
        st.metric("Cancellations caught",
                  f"{recall_score(y_true, pred, zero_division=0) * 100:.1f}%")
        st.metric("Of those flagged, actually cancelled",
                  f"{precision_score(y_true, pred, zero_division=0) * 100:.1f}%")
        st.metric("F1 (cancelled class)",
                  f"{f1_score(y_true, pred, zero_division=0):.4f}")
        st.metric("Accuracy", f"{accuracy_score(y_true, pred):.4f}")
        st.caption(f"Missed {fn:,} cancellations and wrongly flagged "
                   f"{fp:,} kept bookings.")
    with right:
        st.altair_chart(confusion_heatmap(cells))



st.title("Hotel Booking ML Capstone")
st.caption("119,390 bookings · predict a nightly rate, score a cancellation, "
           "and compare fifteen algorithms across three tracks")

FAMILY_NOTE = ("Bars and points are coloured by algorithm family, so a model "
               "keeps its colour when you change the metric or narrow the "
               "selection.")

reg, clf, clu = st.tabs(["Regression", "Classification", "Clustering"])

# ---------------------------------------------------------------- regression
with reg:
    predict_form("regression", "Price a booking",
                 "Enter a booking and get its average daily rate in euros.")
    st.divider()

    models = load("regression_models")
    if models is None:
        st.warning("Run `python scripts/harvest_metrics.py` first.")
    else:
        st.header("How the ten algorithms compared")
        with st.container(border=True):
            leaderboard(models, "R2", True, ["R2", "RMSE", "MAE", "fit (s)"])

        st.caption(FAMILY_NOTE)
        shown = pick_models(models, "reg_pick")
        if shown is not None:
            chosen = shown.index.tolist()

            REG_METRICS = {"R2": (True, ".3f", ""),
                           "RMSE": (False, ".1f", " (EUR)"),
                           "MAE": (False, ".1f", " (EUR)")}
            with st.container(border=True):
                metric = st.radio("Rank by", list(REG_METRICS), horizontal=True,
                                  key="reg_metric")
                higher, fmt, unit = REG_METRICS[metric]
                st.caption("Higher is better." if higher
                           else "Lower is better.")
                st.altair_chart(ranked_bars(shown, metric, higher, fmt, unit))

            left, right = st.columns(2)
            with left, st.container(border=True):
                st.subheader("Does it generalise?")
                st.caption("Train R2 against test R2. A wide pair is "
                           "memorisation.")
                st.altair_chart(paired_bars(shown, "train R2", "R2", "R2",
                                            names=("train", "test")))
            with right, st.container(border=True):
                st.subheader("Is the accuracy worth the compute?")
                st.caption("Random Forest buys its lead with ~33x the fit "
                           "time of Hist Gradient Boosting.")
                st.altair_chart(cost_scatter(shown, "R2"))

            cv = restrict(load("regression_cv"), chosen)
            if cv is not None:
                with st.container(border=True):
                    st.subheader("Does the ranking survive cross-validation?")
                    st.altair_chart(cv_bars(cv, "CV mean R2", "CV std"))
                    table_view(cv, "Cross-validation table")

            tuning = restrict(load("regression_tuning"), chosen)
            if tuning is not None:
                with st.container(border=True):
                    st.subheader("What did tuning buy?")
                    st.altair_chart(paired_bars(tuning, "default R2",
                                                "tuned R2", "Test R2"))
                    table_view(tuning, "Tuning table")

            table_view(shown)

# ------------------------------------------------------------ classification
with clf:
    predict_form("classification", "Will this booking cancel?",
                 "Enter a booking and get its cancellation probability.")
    st.divider()

    with st.container(border=True):
        operating_point()
    st.divider()

    models = load("classification_models")
    timing = load("classification_timing")
    if models is None:
        st.warning("Run `python scripts/harvest_metrics.py` first.")
    else:
        if timing is not None:
            models = models.join(timing[["fit (s)", "train rows"]])
        st.header("How the five algorithms compared")
        with st.container(border=True):
            leaderboard(models, "f1_weighted", True,
                        ["accuracy", "f1_weighted", "roc_auc"])

        st.caption(FAMILY_NOTE)
        shown = pick_models(models, "clf_pick")
        if shown is not None:
            chosen = shown.index.tolist()

            CLF_METRICS = ["f1_weighted", "accuracy", "roc_auc",
                           "precision_weighted", "recall_weighted"]
            with st.container(border=True):
                metric = st.radio("Rank by", CLF_METRICS, horizontal=True,
                                  key="clf_metric")
                st.caption("Higher is better for all five.")
                st.altair_chart(ranked_bars(shown, metric, True, ".3f"))

            if "fit (s)" in shown:
                with st.container(border=True):
                    st.subheader("Is the accuracy worth the compute?")
                    st.altair_chart(cost_scatter(shown, metric))

            cv = restrict(load("classification_cv"), chosen)
            if cv is not None:
                with st.container(border=True):
                    st.subheader("Cross-validation")
                    st.altair_chart(cv_bars(cv, "CV mean F1", "CV std"))
                    table_view(cv, "Cross-validation table")

            repr_df = load("classification_repr")
            if repr_df is not None:
                with st.container(border=True):
                    st.subheader("Did dimensionality reduction help?")
                    st.caption("Same logistic-regression probe on three "
                               "representations. LDA matches 100 features "
                               "with one.")
                    st.altair_chart(ranked_bars(repr_df, "f1_weighted", True,
                                                ".4f", by_family=False))
                    table_view(repr_df, "Representation table")

            table_view(shown)

# ---------------------------------------------------------------- clustering
with clu:
  with st.container(border=True):
    st.subheader("Not built yet")
    st.write("`notebooks/clustering.ipynb` has no code cells, so there are "
             "no algorithms to compare. Once it fits a few models, add their "
             "comparison table to `WANTED` in `scripts/harvest_metrics.py` "
             "and this tab fills in the same way.")
    st.caption("Natural shape for the track: K-Means, Agglomerative and "
               "DBSCAN scored on silhouette, Calinski-Harabasz and "
               "Davies-Bouldin.")
