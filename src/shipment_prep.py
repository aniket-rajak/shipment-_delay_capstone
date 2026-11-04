"""Shared cleaning and feature engineering for the shipment pipeline."""

import numpy as np
import pandas as pd

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}

FEATURES = [
    "parcel_weight_kg", "distance_km", "log_declared_value",
    "n_stops", "rain_flag", "sale_event_flag", "is_cod",
    "is_weekend", "dispatch_hour", "hub_tier",
    "hub_daily_capacity", "label_print_batch", "scanner_build_no",
    "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
    "promised_hours",
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no",
]


def clean_shipments(raw, medians=None):
    """Remove duplicates, normalize text fields, and impute numeric gaps.

    When ``medians`` is provided, those training-time values are reused so
    prediction data receives exactly the same imputation as training data.
    """
    df = raw.drop_duplicates().copy()

    df["service_type"] = (
        df["service_type"].astype("string").str.strip().str.title()
    )
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"].astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    ).astype(float)

    df = df.drop(columns=["declared_value"])

    df["dispatch_time"] = pd.to_datetime(
        df["dispatch_time"], errors="coerce"
    )

    df["parcel_weight_kg"] = df["parcel_weight_kg"].astype(float)
    df["distance_km"] = df["distance_km"].astype(float)

    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    impute_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]

    if medians is None:
        medians = {
            col: float(df[col].median())
            for col in impute_columns
        }
    else:
        medians = {
            col: float(medians[col])
            for col in impute_columns
        }

    for col in impute_columns:
        df[col] = df[col].fillna(medians[col])

    return df, medians


def build_features(clean, hubs):
    """Attach hub attributes and derive the model feature columns."""
    df = clean.merge(
        hubs[
            [
                "hub_id", "hub_city", "region",
                "hub_tier", "hub_daily_capacity",
            ]
        ],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    df["is_cod"] = df["payment_mode"].eq("COD").astype("int64")
    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek.ge(5).astype("int64")
    )
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype("int64")
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
