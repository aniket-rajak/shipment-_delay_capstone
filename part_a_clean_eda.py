
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = str(PROJECT_ROOT / "data")
OUTPUTS_DIR = str(PROJECT_ROOT / "outputs")
sys.path.insert(0, str(PROJECT_ROOT))

from src.shipment_prep import (  # noqa: E402
    FEATURES,
    build_features,
    clean_shipments,
)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

raw = pd.read_csv(DATA_DIR + "/raw_shipments.csv")
hubs = pd.read_csv(DATA_DIR + "/hubs.csv")

# Clean data and build features
clean_df, medians = clean_shipments(raw)
df = build_features(clean_df, hubs)

# Create delay target
df["is_delayed"] = (
    df["delivery_hours"] > df["promised_hours"]
).astype(int)

# Save modelling table
df.to_csv(DATA_DIR + "/modelling_table.csv", index=False)

# Data-quality and distribution checks.
# "Before" counts are measured on the deduplicated raw export, matching the
# cleaning stage that removes exact duplicate rows.
dedup_raw = raw.drop_duplicates()

hours_q1 = df["delivery_hours"].quantile(0.25)
hours_q3 = df["delivery_hours"].quantile(0.75)
hours_iqr = hours_q3 - hours_q1
hours_lower_bound = hours_q1 - 1.5 * hours_iqr
hours_upper_bound = hours_q3 + 1.5 * hours_iqr

checks = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", int(raw.duplicated().sum())),
    ("rows_after_dedup", len(df)),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("missing_distance_km_before", int(dedup_raw["distance_km"].isna().sum())),
    ("missing_parcel_weight_before", int(dedup_raw["parcel_weight_kg"].isna().sum())),
    ("sentinel_weight_rows", int((dedup_raw["parcel_weight_kg"] > 100).sum())),
    ("missing_declared_value_before", int(dedup_raw["declared_value"].isna().sum())),
    ("missing_payment_mode_before", int(dedup_raw["payment_mode"].isna().sum())),
    ("service_type_spellings_before", int(dedup_raw["service_type"].nunique())),
    ("service_type_spellings_after", int(df["service_type"].nunique())),
    ("rows_after_hub_merge", len(df)),
    ("nulls_in_features_and_target",
     int(df[FEATURES + ["is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(df["is_delayed"].sum())),
    ("delay_rate", df["is_delayed"].mean()),
    ("hours_q1", hours_q1),
    ("hours_q3", hours_q3),
    ("hours_lower_bound", hours_lower_bound),
    ("hours_upper_bound", hours_upper_bound),
    ("hours_outlier_count",
     int(((df["delivery_hours"] < hours_lower_bound)
          | (df["delivery_hours"] > hours_upper_bound)).sum())),
    ("declared_value_mean_inr", df["declared_value_inr"].mean()),
    ("declared_value_median_inr", df["declared_value_inr"].median()),
    ("log_declared_value_mean", df["log_declared_value"].mean()),
    ("log_declared_value_median", df["log_declared_value"].median()),
]

pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
    OUTPUTS_DIR + "/part_a_checks.csv", index=False, float_format="%.4f"
)

# Delay rate by service type, hub tier, and distance band
df["distance_band"] = pd.cut(
    df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True,
)

group_results = []

for dimension in ["service_type", "hub_tier", "distance_band"]:
    grouped = df.groupby(dimension, observed=False).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index()

    grouped = grouped.rename(columns={dimension: "group"})
    grouped.insert(0, "dimension", dimension)
    group_results.append(grouped)

pd.concat(group_results, ignore_index=True).to_csv(
    OUTPUTS_DIR + "/part_a_delay_by_group.csv",
    index=False,
    float_format="%.4f",
)

# Numeric correlation matrix
numeric_df = df.select_dtypes(include="number")
numeric_df.corr().to_csv(
    OUTPUTS_DIR + "/part_a_correlations.csv",
    index=True,
    float_format="%.4f",
)

print("Part A completed!")
print("Modelling table:", df.shape)
print("Delayed shipments:", int(df["is_delayed"].sum()))
print("\nFiles created:")
print(DATA_DIR + "/modelling_table.csv")
print(OUTPUTS_DIR + "/part_a_checks.csv")
print(OUTPUTS_DIR + "/part_a_delay_by_group.csv")
print(OUTPUTS_DIR + "/part_a_correlations.csv")
