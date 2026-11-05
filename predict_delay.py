"""Score new shipments with the packaged delay classifier."""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shipment_prep import PROMISED_HOURS, build_features, clean_shipments  # noqa: E402


REQUIRED_INPUT_COLUMNS = {
    "shipment_id",
    "hub_id",
    "service_type",
    "payment_mode",
    "parcel_weight_kg",
    "distance_km",
    "declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "dispatch_time",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
}


def main():
    data = ROOT / "data"
    outputs = ROOT / "outputs"
    models = ROOT / "models"
    outputs.mkdir(exist_ok=True)

    new_shipments = pd.read_csv(data / "new_shipments.csv")
    hubs = pd.read_csv(data / "hubs.csv")
    segments = pd.read_csv(outputs / "hub_segments.csv")

    missing_input_columns = sorted(
        REQUIRED_INPUT_COLUMNS - set(new_shipments.columns)
    )
    if missing_input_columns:
        raise ValueError(
            "new_shipments.csv is missing required columns: "
            f"{missing_input_columns}"
        )

    unknown_hubs = sorted(set(new_shipments["hub_id"]) - set(hubs["hub_id"]))
    if unknown_hubs:
        raise ValueError(
            f"new_shipments.csv contains unknown hub IDs: {unknown_hubs}"
        )

    medians_df = pd.read_csv(models / "impute_values.csv")
    medians = dict(zip(medians_df["column"], medians_df["median"]))
    features = pd.read_csv(models / "feature_order.csv")["feature"].tolist()
    threshold = float(
        pd.read_csv(models / "delay_threshold.csv")["threshold"].iloc[0]
    )
    model = joblib.load(models / "delay_model.pkl")

    clean, _ = clean_shipments(new_shipments, medians=medians)

    invalid_times = clean["dispatch_time"].isna()
    if invalid_times.any():
        bad_ids = clean.loc[invalid_times, "shipment_id"].tolist()
        raise ValueError(
            f"Invalid dispatch_time values for shipments: {bad_ids}"
        )

    unknown_services = sorted(
        set(clean["service_type"].dropna()) - set(PROMISED_HOURS)
    )
    if unknown_services:
        raise ValueError(f"Unknown service_type values: {unknown_services}")

    featured = build_features(clean, hubs)

    unknown_segments = sorted(
        set(featured["hub_id"]) - set(segments["hub_id"])
    )
    if unknown_segments:
        raise ValueError(
            f"Hub segments are missing for hub IDs: {unknown_segments}"
        )

    missing_features = [name for name in features if name not in featured.columns]
    if missing_features:
        raise ValueError(f"Missing required model features: {missing_features}")

    feature_matrix = featured[features]
    null_features = feature_matrix.columns[
        feature_matrix.isna().any()
    ].tolist()
    if null_features:
        raise ValueError(f"Missing values in model features: {null_features}")

    if not np.isfinite(feature_matrix.to_numpy(dtype=float)).all():
        raise ValueError("Model features contain non-finite values.")

    probabilities = model.predict_proba(feature_matrix)[:, 1]
    flagged = (probabilities >= threshold).astype(int)

    result = pd.DataFrame({
        "shipment_id": featured["shipment_id"],
        "hub_id": featured["hub_id"],
        "segment": featured["hub_id"].map(
            segments.set_index("hub_id")["segment"]
        ),
        "delay_probability": np.round(probabilities, 4),
        "flagged": flagged,
    })

    result["action"] = np.select(
        [
            (result["flagged"] == 1) & (result["segment"] == "Escalate"),
            result["flagged"] == 1,
        ],
        ["Escalate to hub manager", "Proactive customer call"],
        default="No action",
    )

    result = result[
        ["shipment_id", "hub_id", "segment",
         "delay_probability", "flagged", "action"]
    ]

    if len(result) != 48:
        raise ValueError(
            "Expected 48 shipments in data/new_shipments.csv, "
            f"got {len(result)}. Update this dataset-specific check "
            "when scoring a different input file."
        )
    if result.isna().any().any() or result["shipment_id"].duplicated().any():
        raise ValueError(
            "Output contains missing values or duplicate shipment IDs."
        )

    result.to_csv(outputs / "new_shipment_actions.csv", index=False)
    print("Success! 48 shipment predictions saved.")
    print("Flagged:", int(result["flagged"].sum()))
    print(result["action"].value_counts().to_string())
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
