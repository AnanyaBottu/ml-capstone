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
from scipy.cluster.hierarchy import leaves_list, linkage
from streamlit import config as st_config

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import features

BLUE, ORANGE, AQUA = "#2a78d6", "#e4632f", "#1baf7a"

# One set of hues for both modes, so an algorithm keeps its colour when the
# lights go out; only the surface and the ink swap. Both triples clear the
# validator against their own surface (the orange is a step darker than it was
# so it fits the narrower dark lightness band). The aqua sits just under 3:1 on
# the light surface, which the direct bar labels are the relief for.
LIGHT = {"surface": "#fcfcfb", "secondary": "#f2f2ef", "ink": "#0b0b0b",
         "muted": "#52514e", "grid": "#e8e8e5", "ramp": "blues"}
DARK = {"surface": "#1a1a19", "secondary": "#242422", "ink": "#f4f4f1",
        "muted": "#a6a5a0", "grid": "#33332f", "ramp": "darkblue"}

st.set_page_config(page_title="Capstone Model Comparison", page_icon="📊",
                   layout="wide")

# ----------------------------------------------------------------- light/dark
# Streamlit has no per-session theme API, so the toggle writes the server's
# theme config and reruns; the config travels with the next run and the
# frontend re-themes itself.
# ponytail: server-global, so two people viewing at once would flip each
# other's theme. Fine for a demo, would need a CSS-variable theme to fix.
MODE_KEY = "dark_mode"
DARK_MODE = bool(st.session_state.get(MODE_KEY, False))
PALETTE = DARK if DARK_MODE else LIGHT
SURFACE, INK, MUTED, GRID = (PALETTE[k] for k in
                             ("surface", "ink", "muted", "grid"))


def sync_theme():
    """Push the chosen mode into the theme config, once, then rerun."""
    want = "dark" if DARK_MODE else "light"
    if st_config.get_option("theme.base") == want:
        return
    for key, value in {"base": want, "primaryColor": BLUE,
                       "backgroundColor": PALETTE["surface"],
                       "secondaryBackgroundColor": PALETTE["secondary"],
                       "textColor": PALETTE["ink"]}.items():
        st_config.set_option(f"theme.{key}", value)
    st.rerun()


sync_theme()

# The toolbar and the hosted "Manage app" badge are hosting chrome, not part
# of the dashboard. Settings lived in that menu, which is why the mode toggle
# is in the header instead.
st.markdown("""<style>
[data-testid="stToolbar"], [data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"], #MainMenu, footer {display: none !important;}
[data-testid="stHeader"] {display: none;}
[data-testid="stMainBlockContainer"] {padding-top: 2rem;}
</style>""", unsafe_allow_html=True)


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
    "Gradient Boosting": "Tree & ensemble",
    "Bagging": "Tree & ensemble",
    "AdaBoost": "Tree & ensemble",
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
                        scale=alt.Scale(scheme=PALETTE["ramp"])),
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



# -------------------------------------------------------------- clustering
# Two hues are enough: K-Means is the one centroid method, the other four are
# linkages of the same hierarchical algorithm. Each keeps its colour whichever
# metric is on show.
CLUSTER_FAMILY = {"K-Means": "Centroid"}
CLUSTER_ORDER = ["Centroid", "Hierarchical"]
CLUSTER_RANGE = [BLUE, ORANGE]
K_CHOSEN = 4  # the notebook's judgement call (section 6.4), not a metric result

# Names the notebook gives the four K-Means segments in section 9. They are an
# interpretation of the profile table, so an unknown cluster falls back to its
# number rather than borrowing a wrong name.
SEGMENT_NAMES = {0: "Mainstream online", 1: "Families", 2: "Repeat corporate",
                 3: "Direct last-minute"}


def cluster_family(model):
    return CLUSTER_FAMILY.get(model, "Hierarchical")


def cluster_models():
    """The five-algorithm table with `usable` as a real boolean (the harvest
    writes it as text) and a `solution` label that drives the bar colour."""
    df = load("clustering_models")
    if df is None:
        return None
    df = df.copy()
    df["usable"] = df["usable"].astype(str).eq("True")
    return df


def cluster_bars(df, metric, higher_is_better, fmt=".3f"):
    """One validity index, every algorithm. Bars that only score well by
    isolating a handful of bookings are greyed rather than ranked in colour,
    because a high silhouette there is an artefact, not a segmentation."""
    d = df.reset_index()
    d["solution"] = d["usable"].map({True: "usable partition",
                                     False: "degenerate (smallest cluster < 1%)"})
    d["family"] = d["model"].map(cluster_family)
    base = alt.Chart(d).encode(
        x=alt.X(f"{metric}:Q", title=metric, scale=alt.Scale(nice=True)),
        y=alt.Y("model:N", title=None, sort=alt.EncodingSortField(
            metric, order="descending" if higher_is_better else "ascending")),
        tooltip=["model:N", alt.Tooltip(f"{metric}:Q", format=fmt),
                 alt.Tooltip("smallest cluster %:Q", format=".2f"),
                 alt.Tooltip("largest cluster %:Q", format=".1f"), "solution:N"])
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.Color("family:N", title=None, scale=alt.Scale(
            domain=CLUSTER_ORDER, range=CLUSTER_RANGE),
            legend=alt.Legend(orient="top", direction="horizontal")),
        opacity=alt.condition(alt.datum.usable, alt.value(1.0), alt.value(0.35)))
    labels = base.mark_text(align="left", dx=6, color=MUTED, fontSize=11).encode(
        text=alt.Text(f"{metric}:Q", format=fmt))
    return style(bars + labels, 30 * len(d) + 70)


def balance_bars(df):
    """Share of all bookings held by the largest cluster. This is the number
    that exposes the degenerate linkages."""
    d = df.reset_index()
    base = alt.Chart(d).encode(
        x=alt.X("largest cluster %:Q", title="Largest cluster (% of bookings)",
                scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("model:N", title=None, sort=alt.EncodingSortField(
            "largest cluster %", order="descending")),
        tooltip=["model:N", alt.Tooltip("largest cluster %:Q", format=".1f"),
                 alt.Tooltip("smallest cluster %:Q", format=".3f")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=18, color=BLUE)
    labels = base.mark_text(align="left", dx=6, color=MUTED, fontSize=11).encode(
        text=alt.Text("largest cluster %:Q", format=".1f"))
    return style(bars + labels, 30 * len(d) + 70)


def k_lines(kmeans, ward, metric):
    """Validity index against k for both algorithms on one shared axis, with
    the notebook's chosen k marked."""
    parts = [kmeans[[metric]].assign(model="K-Means")]
    if ward is not None:
        parts.append(ward[[metric]].assign(model="Agglomerative (ward)"))
    d = pd.concat(parts).rename_axis("k").reset_index()
    base = alt.Chart(d).encode(
        x=alt.X("k:Q", title="Number of clusters (k)",
                axis=alt.Axis(tickMinStep=1)),
        y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", title=None, scale=alt.Scale(
            domain=["K-Means", "Agglomerative (ward)"],
            range=[BLUE, ORANGE]),
            legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=["model:N", "k:Q", alt.Tooltip(f"{metric}:Q", format=".3f")])
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(filled=True, size=70, stroke=SURFACE, strokeWidth=2)
    chosen = (alt.Chart(pd.DataFrame({"k": [K_CHOSEN]}))
              .mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="k:Q"))
    return style(lines + points + chosen, 320)


def elbow(kmeans):
    d = kmeans.rename_axis("k").reset_index()
    base = alt.Chart(d).encode(
        x=alt.X("k:Q", title="Number of clusters (k)",
                axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("inertia:Q", title="Inertia (within-cluster sum of squares)",
                scale=alt.Scale(zero=False)),
        tooltip=["k:Q", alt.Tooltip("inertia:Q", format=",.0f")])
    line = base.mark_line(color=BLUE, strokeWidth=2)
    points = base.mark_point(color=BLUE, filled=True, size=70, stroke=SURFACE,
                             strokeWidth=2)
    chosen = (alt.Chart(pd.DataFrame({"k": [K_CHOSEN]}))
              .mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="k:Q"))
    return style(line + points + chosen, 320)


def agreement_heatmap(df):
    """Adjusted Rand Index between every pair of algorithms, one sequential
    ramp, every cell labelled."""
    d = (df.rename_axis("a").reset_index()
         .melt(id_vars="a", var_name="b", value_name="ARI"))
    order = df.index.tolist()
    base = alt.Chart(d).encode(
        x=alt.X("b:N", title=None, sort=order, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("a:N", title=None, sort=order))
    cells = base.mark_rect(stroke=SURFACE, strokeWidth=2).encode(
        color=alt.Color("ARI:Q", scale=alt.Scale(
            scheme=PALETTE["ramp"], domain=[0, 1], clamp=True)),
        tooltip=["a:N", "b:N", alt.Tooltip("ARI:Q", format=".3f")])
    labels = base.mark_text(fontSize=12, fontWeight="bold").encode(
        text=alt.Text("ARI:Q", format=".2f"),
        color=alt.condition(alt.datum.ARI > 0.5, alt.value(SURFACE),
                            alt.value(INK)))
    return style(cells + labels, 340)


def segment_bars(profile, feature, fmt):
    d = profile[[feature]].copy()
    d["segment"] = [f"{i} - {SEGMENT_NAMES.get(i, 'cluster')}" for i in d.index]
    d["share"] = profile["share_%"].to_numpy()
    base = alt.Chart(d).encode(
        x=alt.X(f"{feature}:Q", title=feature.replace("_", " ")),
        y=alt.Y("segment:N", title=None, sort=None),
        tooltip=["segment:N", alt.Tooltip(f"{feature}:Q", format=fmt),
                 alt.Tooltip("share:Q", format=".1f", title="share of bookings (%)")])
    bars = base.mark_bar(cornerRadiusEnd=4, height=22, color=BLUE)
    labels = base.mark_text(align="left", dx=6, color=MUTED, fontSize=11).encode(
        text=alt.Text(f"{feature}:Q", format=fmt))
    return style(bars + labels, 40 * len(d) + 70)


# The profile columns worth charting, with a number format each.
SEGMENT_FEATURES = {"share_%": ".1f", "cancel_rate": ".0%", "adr": ".0f",
                    "lead_time": ".0f", "nights": ".1f", "guests": ".2f",
                    "family": ".0%", "repeat_guest": ".0%", "has_agent": ".0%",
                    "domestic": ".0%", "resort": ".0%", "non_refund": ".0%",
                    "special_requests": ".2f", "parking": ".0%"}


# ------------------------------------------------------- clustering playground
# A toy 2D canvas: click to drop points, then step through agglomerative
# merges one at a time. Each cluster of two or more points gets its own hue;
# lone points stay grey. A merged cluster keeps the hue of its bigger half, so
# colours follow clusters across steps instead of reshuffling.
PG_SIZE, PG_CELL = 10, 0.25
# The validated eight-slot categorical order, stepped per mode. In a scatter
# every pair of hues can touch and past three they are not all colourblind-
# safe, which the spokes to each centre make up for.
CLUSTER_HUES = (["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
                 "#008300", "#9085e9", "#e66767"] if DARK_MODE else
                ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                 "#008300", "#6250d6", "#e34948"])
LINKAGES = {"ward": "Ward", "complete": "Complete", "average": "Average",
            "single": "Single"}


def seed_points():
    rng = np.random.default_rng(7)
    centres = [(2.5, 7), (7, 7.5), (5, 2.5)]
    return [tuple(rng.normal(c, 0.7).clip(0.3, PG_SIZE - 0.3).round(2))
            for c in centres for _ in range(6)]


def merge_state(points, method, step):
    """Cluster id of every point after `step` merges, and the hue slot of
    every live multi-point cluster. Ids are scipy's node numbers, so a merge
    mints a new id and leaves the others untouched.
    Returns (owner, slots, linkage matrix)."""
    n = len(points)
    z = linkage(np.asarray(points), method)
    owner, slots = list(range(n)), {}
    free = list(range(len(CLUSTER_HUES)))
    size = lambda c: 1 if c < n else z[c - n, 3]
    for i in range(step):
        a, b = sorted((int(z[i, 0]), int(z[i, 1])), key=size, reverse=True)
        owner = [n + i if o in (a, b) else o for o in owner]
        keep = slots.pop(a, None)
        spare = slots.pop(b, None)
        if keep is None:
            keep, spare = spare, None
        if spare is not None:
            free.append(spare)
        if keep is None and free:
            free.sort()
            keep = free.pop(0)
        # ponytail: past eight live clusters the extras stay grey; a toy
        # canvas rarely gets there, small multiples would if it did
        if keep is not None:
            slots[n + i] = keep
    return owner, slots, z


def playground_chart(points, owner, slots, merged, latest):
    d = pd.DataFrame(points, columns=["x", "y"])
    d["cluster"] = owner
    d["hue"] = [slots.get(c, -1) for c in owner]
    d["merged"] = d["cluster"].eq(merged)
    d["latest"] = d.index == latest
    centres = d.groupby("cluster")[["x", "y"]].transform("mean")
    d["cx"], d["cy"] = centres["x"], centres["y"]
    axis = dict(scale=alt.Scale(domain=[0, PG_SIZE], nice=False), title=None)

    # invisible cells catch the clicks that add a point
    g = np.arange(0, PG_SIZE, PG_CELL)
    gx, gy = np.meshgrid(g, g)
    grid = pd.DataFrame({"gx": gx.ravel(), "gy": gy.ravel()})
    grid["gx2"], grid["gy2"] = grid["gx"] + PG_CELL, grid["gy"] + PG_CELL
    click = alt.selection_point(name="click", fields=["gx", "gy"],
                                on="click", clear=False)
    cells = alt.Chart(grid).mark_rect(opacity=0.001).encode(
        x=alt.X("gx:Q", **axis), x2="gx2", y=alt.Y("gy:Q", **axis),
        y2="gy2").add_params(click)

    spokes = alt.Chart(d).mark_rule(color=MUTED, strokeWidth=1,
                                    opacity=0.5).encode(
        x="x:Q", y="y:Q", x2="cx", y2="cy")
    # the just-merged cluster wears an ink ring, your latest point a diamond
    dots = alt.Chart(d).mark_point(filled=True, size=110, strokeWidth=2,
                                   opacity=1).encode(
        x="x:Q", y="y:Q",
        color=alt.Color("hue:N", legend=None, scale=alt.Scale(
            domain=[-1, *range(len(CLUSTER_HUES))],
            range=[MUTED, *CLUSTER_HUES])),
        stroke=alt.condition(alt.datum.merged, alt.value(INK),
                             alt.value(SURFACE)),
        shape=alt.condition(alt.datum.latest, alt.value("diamond"),
                            alt.value("circle")),
        size=alt.condition(alt.datum.latest, alt.value(170), alt.value(110)),
        tooltip=[alt.Tooltip("x:Q", format=".2f"),
                 alt.Tooltip("y:Q", format=".2f")])
    return style(cells + spokes + dots, 440)


def dendrogram_chart(z, step, slots):
    """The merge tree, drawn from the linkage matrix. Merges already done wear
    the hue of the cluster they now sit in, so the tree matches the canvas;
    the latest is drawn thicker and the ones still to come are faded. The
    dashed line is where the tree is cut to leave the chosen k clusters."""
    n = len(z) + 1
    x = {int(leaf): i for i, leaf in enumerate(leaves_list(z))}
    h = dict.fromkeys(range(n), 0.0)
    parent = {int(c): n + i for i in range(step) for c in z[i, :2]}

    def live(node):
        while node in parent:
            node = parent[node]
        return node

    rows = []
    for i, (a, b, dist, _) in enumerate(z):
        a, b, node = int(a), int(b), n + i
        x[node], h[node] = (x[a] + x[b]) / 2, dist
        hue = slots.get(live(node), -1) if i < step else -2
        rows += [(x[a], h[a], x[a], dist, hue, i + 1),
                 (x[b], h[b], x[b], dist, hue, i + 1),
                 (x[a], dist, x[b], dist, hue, i + 1)]
    d = pd.DataFrame(rows, columns=["x", "y", "x2", "y2", "hue", "merge"])
    d["latest"] = d["merge"] == step
    tree = alt.Chart(d).mark_rule().encode(
        x=alt.X("x:Q", title=None, axis=None), x2="x2",
        y=alt.Y("y:Q", title="Merge distance"), y2="y2",
        color=alt.Color("hue:N", legend=None, scale=alt.Scale(
            domain=[-2, -1, *range(len(CLUSTER_HUES))],
            range=[GRID, MUTED, *CLUSTER_HUES])),
        strokeWidth=alt.condition(alt.datum.latest, alt.value(4),
                                  alt.value(2)),
        tooltip=["merge:Q", alt.Tooltip("y2:Q", format=".2f",
                                        title="distance")])
    if step == n - 1:  # k = 1: nothing left to cut
        return style(tree, 440)
    below = z[step - 1, 2] if step else 0.0
    cut = alt.Chart(pd.DataFrame({"y": [(below + z[step, 2]) / 2]})).mark_rule(
        color=INK, strokeDash=[4, 4]).encode(y="y:Q")
    return style(tree + cut, 440)


def add_clicked(key):
    hit = st.session_state[key]["selection"].get("click") or []
    if hit:
        cx = hit[0]["gx"] + PG_CELL / 2
        cy = hit[0]["gy"] + PG_CELL / 2
        st.session_state.pg_points.append((round(cx, 2), round(cy, 2)))
        st.session_state.pg_latest = len(st.session_state.pg_points) - 1
        st.session_state.pg_round += 1  # fresh chart key clears the selection


# A fragment reruns on its own, so a click or a slider step redraws these two
# charts instead of the whole dashboard.
@st.fragment
def clustering_playground():
    st.header("Watch clusters form")
    ss = st.session_state
    if "pg_points" not in ss:
        ss.pg_points, ss.pg_latest, ss.pg_round = seed_points(), None, 0

    with st.container(border=True):
        method, reset, clear = st.columns([3, 1, 1],
                                          vertical_alignment="bottom")
        how = method.radio("Linkage", list(LINKAGES),
                           format_func=LINKAGES.get, horizontal=True,
                           key="pg_linkage")
        if reset.button("Reset points", width="stretch"):
            ss.pg_points, ss.pg_latest = seed_points(), None
            ss.pg_round += 1
        if clear.button("Clear", width="stretch"):
            ss.pg_points, ss.pg_latest = [], None
            ss.pg_round += 1

        points, n = ss.pg_points, len(ss.pg_points)
        if n < 2:
            st.caption("Click the canvas to add points; two are needed "
                       "before anything can merge.")
            owner, slots, z, step = list(range(n)), {}, None, 0
        else:
            # a new key per edit, so adding a point restarts the replay
            k = st.slider("Number of clusters (k)", 1, n, n,
                          key=f"pg_k_{ss.pg_round}",
                          help="Each step down joins the two closest "
                          "clusters.")
            step = n - k
            owner, slots, z = merge_state(points, how, step)

        merged = n + step - 1 if step else None
        key = f"pg_click_{ss.pg_round}"
        canvas, tree = st.columns([3, 2])
        canvas.altair_chart(playground_chart(points, owner, slots,
                                             merged, ss.pg_latest),
                            on_select=lambda: add_clicked(key), key=key,
                            selection_mode="click")
        if z is not None:
            tree.altair_chart(dendrogram_chart(z, step, slots))

        if step:
            a, b, dist, size = z[step - 1]
            sizes = [1 if c < n else int(z[int(c) - n, 3]) for c in (a, b)]
            st.caption(f"Merge {step}: a cluster of {sizes[0]} joined one of "
                       f"{sizes[1]} at distance {dist:.2f}, making "
                       f"{int(size)} points. {n - step} clusters left.")
        else:
            st.caption(f"Every point starts as its own cluster ({n}). Lower "
                       "k to merge them, or click to add a point.")
        st.caption("Each colour is one cluster, grey points are still on "
                   "their own, and lines join each point to its cluster's "
                   "centre. The ringed cluster is the one that just merged; "
                   "the diamond is your latest point. The tree on the right "
                   "is the full merge order in the same colours, tallest "
                   "joins last and the thick link is the latest; the dashed line "
                   "cuts it into k clusters. Try Single "
                   "linkage on a long chain of points: it merges "
                   "neighbour by neighbour, the same chaining that made it "
                   "degenerate on the booking data.")


def clustering_tab():
    clustering_playground()
    models = cluster_models()
    if models is None:
        st.warning("Run `python scripts/harvest_metrics.py` first.")
        return

    st.header("How the five algorithms compared")
    usable = models[models["usable"]]
    best = usable["mean rank"].idxmin()
    runner = usable["mean rank"].drop(best).idxmin()
    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:.8rem;color:{MUTED};letter-spacing:.04em;'
            f'text-transform:uppercase;">Best usable partition at k = '
            f'{K_CHOSEN}</div><div style="font-size:1.55rem;font-weight:700;'
            f'margin:.1rem 0;">{best}</div>', unsafe_allow_html=True)
        st.write("")
        cols = st.columns(4)
        for col, (name, fmt) in zip(cols, [("Silhouette", "{:.3f}"),
                                           ("Davies-Bouldin", "{:.3f}"),
                                           ("Calinski-Harabasz", "{:.0f}"),
                                           ("fit (s)", "{:.2f}")]):
            col.metric(name, fmt.format(models.at[best, name]))
        st.caption(
            f"{best} edges {runner} on mean rank "
            f"({usable.at[best, 'mean rank']:.2f} vs "
            f"{usable.at[runner, 'mean rank']:.2f}), but the two are "
            "effectively tied. The notebook picks K-Means as the final model "
            f"because it fits {models.at[best, 'fit (s)'] / models.at['K-Means', 'fit (s)']:.0f}x "
            "faster and can assign new bookings.")

    shown = pick_models(models, "clu_pick")
    if shown is None:
        return

    with st.container(border=True):
        metric = st.radio("Rank by", ["Silhouette", "Davies-Bouldin",
                                      "Calinski-Harabasz"], horizontal=True,
                          key="clu_metric")
        higher = metric != "Davies-Bouldin"
        st.caption("Higher is better." if higher else "Lower is better.")
        st.altair_chart(cluster_bars(shown, metric, higher,
                                     ".0f" if metric == "Calinski-Harabasz"
                                     else ".3f"))
        st.caption("Faded bars are not segmentations. Average and single "
                   "linkage score best on Silhouette and Davies-Bouldin by "
                   "isolating a handful of bookings, which is why the next "
                   "chart matters.")

    with st.container(border=True):
        st.subheader("Why the top scores are a trap")
        st.caption("Share of all bookings held by the biggest cluster. Near "
                   "100% means one cluster plus a few stray points.")
        st.altair_chart(balance_bars(shown))

    kmeans, ward = load("clustering_kmeans_k"), load("clustering_ward_k")
    if kmeans is not None:
        with st.container(border=True):
            st.subheader("Does any index pick a k?")
            index = st.radio("Index", ["silhouette", "davies_bouldin",
                                       "calinski_harabasz"], horizontal=True,
                             key="clu_index")
            st.altair_chart(k_lines(kmeans, ward, index))
            st.caption(f"The dashed line is k = {K_CHOSEN}, chosen because "
                       "it is the largest k where every cluster is a "
                       "different kind of booking. No index picks it: "
                       "silhouette peaks at k = 2.")
            st.altair_chart(elbow(kmeans))
            st.caption("The elbow is smooth, with no kink to read a k from.")

    agree = load("clustering_agreement")
    if agree is not None:
        with st.container(border=True):
            st.subheader("Do the algorithms find the same segments?")
            st.altair_chart(agreement_heatmap(agree))
            st.caption("Adjusted Rand Index, 1 = identical. K-Means and Ward "
                       "agree at 0.905, so they are one solution found twice; "
                       "the degenerate linkages agree with nothing.")

    shares, stability = load("clustering_shares"), load("clustering_stability")
    if shares is not None and stability is not None:
        with st.container(border=True):
            st.subheader("Does it hold up on unseen bookings?")
            left, right = st.columns(2)
            train, hold = stability.index[0], stability.index[1]
            left.metric("Silhouette, hold-out",
                        f"{stability.at[hold, 'silhouette']:.3f}",
                        delta=f"{stability.at[hold, 'silhouette'] - stability.at[train, 'silhouette']:+.3f} vs train")
            right.metric("Davies-Bouldin, hold-out",
                         f"{stability.at[hold, 'Davies-Bouldin']:.3f}",
                         delta=f"{stability.at[hold, 'Davies-Bouldin'] - stability.at[train, 'Davies-Bouldin']:+.3f} vs train",
                         delta_color="inverse")
            labelled = shares.copy()
            labelled.index = [f"Cluster {i}" for i in labelled.index]
            labelled.index.name = "cluster"
            st.altair_chart(paired_bars(labelled, "train %", "hold-out %",
                                        "Share of bookings (%)",
                                        names=("train", "hold-out")))
            st.caption("The final K-Means model on 17,139 bookings it never "
                       "saw. Calinski-Harabasz is left out because it grows "
                       "with the number of points.")

    profile = load("clustering_profile")
    if profile is not None:
        with st.container(border=True):
            st.subheader("Who are the four segments?")
            feature = st.selectbox(
                "Compare segments on", list(SEGMENT_FEATURES),
                format_func=lambda c: c.replace("_", " ").replace(
                    "share %", "share of bookings (%)"),
                index=list(SEGMENT_FEATURES).index("cancel_rate"),
                key="clu_feature")
            st.altair_chart(segment_bars(profile, feature,
                                         SEGMENT_FEATURES[feature]))
            st.caption("K-Means with k = 4 on all training rows. Cancel rate "
                       "is the reference label, shown for interpretation "
                       "only: the algorithm never saw it, yet it ranges from "
                       "8% to 35% across segments.")
            table_view(profile, "Segment profile table")

    external = load("clustering_external")
    if external is not None:
        table_view(external, "External validation against the reference labels")
    table_view(shown)


head, mode = st.columns([6, 1], vertical_alignment="bottom")
with head:
    st.title("Hotel Booking ML Capstone")
    st.caption("119,390 bookings · predict a nightly rate, score a "
               "cancellation, and compare twenty algorithms across three "
               "tracks")
with mode:
    st.toggle("Dark mode", key=MODE_KEY)

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
                times = models["fit (s)"]
                st.caption("Random Forest buys its lead with ~"
                           f"{times['Random Forest'] / times['Hist Gradient Boosting']:.0f}"
                           "x the fit time of Hist Gradient Boosting.")
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
        st.header("How the ten algorithms compared")
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

            # Part B fit times were measured on another machine, so only
            # Part A's five are placed on one time axis.
            timed = (shown.dropna(subset=["fit (s)"]) if "fit (s)" in shown
                     else shown.iloc[:0])
            if not timed.empty:
                with st.container(border=True):
                    st.subheader("Is the accuracy worth the compute?")
                    st.altair_chart(cost_scatter(timed, metric))
                    st.caption("Part A's five models only: Part B was timed "
                               "on a different machine, so its fit times "
                               "are not comparable.")

            cv_parts = [c[["CV mean F1", "CV std"]] for c in
                        (load("classification_cv"), load("classification_cv_b"))
                        if c is not None]
            cv = restrict(pd.concat(cv_parts), chosen) if cv_parts else None
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
    clustering_tab()
