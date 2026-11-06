import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS

def run_part_a():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    raw_rows = len(raw)
    raw_dedup_before = raw.drop_duplicates()
    
    service_type_spellings_before = raw_dedup_before["service_type"].nunique()
    missing_dist_before = raw_dedup_before["distance_km"].isna().sum()
    missing_weight_before = (raw_dedup_before["parcel_weight_kg"].isna() & (raw_dedup_before["parcel_weight_kg"] <= 100)).sum()
    sentinel_weight_rows = (raw_dedup_before["parcel_weight_kg"] > 100).sum()

    clean_val_str = (
        raw_dedup_before["declared_value"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    missing_val_before = pd.to_numeric(clean_val_str, errors="coerce").isna().sum()
    missing_pay_before = raw_dedup_before["payment_mode"].isna().sum()

    clean_df, medians = clean_shipments(raw)
    modelling_df = build_features(clean_df, hubs)
    modelling_df["is_delayed"] = (modelling_df["delivery_hours"] > modelling_df["promised_hours"]).astype(int)

    modelling_df.to_csv("data/modelling_table.csv", index=False)

    service_type_spellings_after = modelling_df["service_type"].nunique()
    dedup_rows = len(modelling_df)
    dup_rows_removed = raw_rows - dedup_rows
    nulls_in_feats_target = modelling_df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()
    delayed_cnt = int(modelling_df["is_delayed"].sum())
    delay_rate_val = float(delayed_cnt / dedup_rows)

    q1 = float(modelling_df["delivery_hours"].quantile(0.25))
    q3 = float(modelling_df["delivery_hours"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)
    outliers = int(((modelling_df["delivery_hours"] < lower_bound) | (modelling_df["delivery_hours"] > upper_bound)).sum())

    checks = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", dup_rows_removed),
        ("rows_after_dedup", dedup_rows),
        ("missing_distance_km_before", missing_dist_before),
        ("missing_parcel_weight_before", missing_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_val_before),
        ("missing_payment_mode_before", missing_pay_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("median_distance_km", round(medians["distance_km"], 4)),
        ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
        ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
        ("rows_after_hub_merge", len(modelling_df)),
        ("nulls_in_features_and_target", nulls_in_feats_target),
        ("delayed_count", delayed_cnt),
        ("delay_rate", round(delay_rate_val, 4)),
        ("hours_q1", round(q1, 4)),
        ("hours_q3", round(q3, 4)),
        ("hours_lower_bound", round(lower_bound, 4)),
        ("hours_upper_bound", round(upper_bound, 4)),
        ("hours_outlier_count", outliers),
        ("declared_value_mean_inr", round(float(modelling_df["declared_value_inr"].mean()), 4)),
        ("declared_value_median_inr", round(float(modelling_df["declared_value_inr"].median()), 4)),
        ("log_declared_value_mean", round(float(modelling_df["log_declared_value"].mean()), 4)),
        ("log_declared_value_median", round(float(modelling_df["log_declared_value"].median()), 4)),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)

    # Delay by Group
    records = []
    for st in ["Economy", "Express", "Standard"]:
        grp = modelling_df[modelling_df["service_type"] == st]
        records.append({
            "dimension": "service_type", "group": st,
            "shipments": len(grp), "delayed": int(grp["is_delayed"].sum()),
            "delay_rate": round(float(grp["is_delayed"].mean()), 4),
            "mean_hours": round(float(grp["delivery_hours"].mean()), 4)
        })

    for ht in [1, 2, 3]:
        grp = modelling_df[modelling_df["hub_tier"] == ht]
        records.append({
            "dimension": "hub_tier", "group": str(ht),
            "shipments": len(grp), "delayed": int(grp["is_delayed"].sum()),
            "delay_rate": round(float(grp["is_delayed"].mean()), 4),
            "mean_hours": round(float(grp["delivery_hours"].mean()), 4)
        })

    bins = [0, 200, 400, 600, 2000]
    labels = ["0-200", "201-400", "401-600", "601+"]
    modelling_df["distance_band"] = pd.cut(modelling_df["distance_km"], bins=bins, labels=labels, right=True)

    for db in labels:
        grp = modelling_df[modelling_df["distance_band"] == db]
        records.append({
            "dimension": "distance_band", "group": db,
            "shipments": len(grp), "delayed": int(grp["is_delayed"].sum()),
            "delay_rate": round(float(grp["is_delayed"].mean()), 4),
            "mean_hours": round(float(grp["delivery_hours"].mean()), 4)
        })

    pd.DataFrame(records).to_csv("outputs/part_a_delay_by_group.csv", index=False)

    num_cols = modelling_df.select_dtypes(include=[np.number]).columns
    corr_df = modelling_df[num_cols].corr().round(4)
    corr_df.to_csv("outputs/part_a_correlations.csv", index_label="feature")

if __name__ == "__main__":
    run_part_a()