"""Cleaning and feature engineering shared by the notebooks' pipelines.

scripts/train_models.py uses this to fit; app.py uses the identical functions
to prepare a single user-entered booking. Keeping one copy is the point: a
second implementation in the app would drift and the predictions would be
quietly wrong.

Mirrors sections 3 and 4 of notebooks/regression.ipynb and
notebooks/classification.ipynb.
"""
import numpy as np
import pandas as pd

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# Known at departure or check-in, so not available when a booking is made.
LEAKY = ["reservation_status", "reservation_status_date", "assigned_room_type"]

# Levels kept for the two high-cardinality columns; everything else is "Other".
TOP_N = {"country": 20, "agent": 15}

# Derived in engineer(), never asked of the user.
ENGINEERED = ["total_guests", "is_family", "total_nights", "weekend_ratio",
              "month_num", "week_sin", "week_cos", "prev_total",
              "prev_cancel_rate"]

TARGET = {"regression": "adr", "classification": "is_canceled"}


def clean(df, track):
    """Drop leaky columns, resolve the four missing-value cases, drop invalid
    rows. Row-order of the filters follows each notebook."""
    drop = LEAKY + (["is_canceled"] if track == "regression" else [])
    df = df.drop(columns=[c for c in drop if c in df])

    # company: 94% missing, the id is unusable but its presence is a signal
    df["has_company"] = df["company"].notna().astype(int)
    df = df.drop(columns=["company"])

    # agent: blank means "booked without an agent" - an informative absence
    df["has_agent"] = df["agent"].notna().astype(int)
    df["agent"] = df["agent"].fillna(0).astype(int).astype(str)

    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["meal"] = df["meal"].replace({"Undefined": "SC"})  # both mean no meal

    if track == "regression":
        df = df[df["adr"].between(0, 1000)]
        df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    else:
        df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
        df = df[df["adr"] >= 0]

    return df.drop_duplicates().reset_index(drop=True)


def engineer(df):
    """Stateless, row-wise, so it works the same on 69,000 rows or on one."""
    df = df.copy()

    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] + df["babies"]) > 0).astype(int)

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["weekend_ratio"] = (df["stays_in_weekend_nights"]
                           / df["total_nights"].replace(0, np.nan)).fillna(0)

    # week 52 and week 1 are adjacent in the calendar, 51 apart on a number line
    df["month_num"] = df["arrival_date_month"].map(
        {m: i + 1 for i, m in enumerate(MONTHS)})
    df["week_sin"] = np.sin(2 * np.pi * df["arrival_date_week_number"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["arrival_date_week_number"] / 52)

    df["prev_total"] = (df["previous_cancellations"]
                        + df["previous_bookings_not_canceled"])
    df["prev_cancel_rate"] = df["previous_cancellations"] / (df["prev_total"] + 1)

    return df.drop(columns=["arrival_date_month"])


def fit_rare_levels(df):
    """The levels to keep per high-cardinality column, learned on training data."""
    return {col: df[col].value_counts().nlargest(n).index.tolist()
            for col, n in TOP_N.items()}


def apply_rare_levels(df, levels):
    df = df.copy()
    for col, keep in levels.items():
        df[col] = np.where(df[col].isin(keep), df[col], "Other")
    return df


def prepare(df_raw, track):
    """Raw CSV to model-ready X, y plus the rare-level map the app needs."""
    df = engineer(clean(df_raw, track))
    levels = fit_rare_levels(df)
    df = apply_rare_levels(df, levels)
    target = TARGET[track]
    return df.drop(columns=[target]), df[target], levels
