import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import FEATURES, clean_shipments

def package_and_verify():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Load Data & Medians
    raw = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw)
    
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # Rebuild Stratified Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Read config from Part C
    cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    threshold = float(cfg["threshold"])
    
    # Parse hyperparams
    rf_kwargs = {"random_state": 42}
    if not pd.isna(cfg["n_estimators"]):
        rf_kwargs["n_estimators"] = int(cfg["n_estimators"])
    if str(cfg["max_depth"]).isdigit():
        rf_kwargs["max_depth"] = int(cfg["max_depth"])
    if not pd.isna(cfg["min_samples_leaf"]):
        rf_kwargs["min_samples_leaf"] = int(cfg["min_samples_leaf"])

    # Fit final pipeline on training set
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
        ("model", RandomForestClassifier(**rf_kwargs))
    ])
    final_pipeline.fit(X_train, y_train)

    # Export Artefacts
    joblib.dump(final_pipeline, "models/delay_model.pkl")
    pd.DataFrame([{"threshold": threshold}]).to_csv("models/delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    pd.DataFrame([{"column": k, "median": v} for k, v in medians.items()]).to_csv("models/impute_values.csv", index=False)

    # Verification Reload Test
    reloaded_pipeline = joblib.load("models/delay_model.pkl")
    
    in_mem_probs = final_pipeline.predict_proba(X_test)[:, 1]
    reload_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]
    max_diff = float(abs(in_mem_probs - reload_probs).max())

    reload_check_df = pd.DataFrame([
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", round(max_diff, 9)),
        ("threshold", round(threshold, 4))
    ], columns=["metric", "value"])
    reload_check_df.to_csv("outputs/part_d_reload_check.csv", index=False)

if __name__ == "__main__":
    package_and_verify()