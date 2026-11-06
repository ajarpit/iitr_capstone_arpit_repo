import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

def batch_score():
    # Load Model & Metadata
    model = joblib.load("models/delay_model.pkl")
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    
    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))
    
    hub_segments = pd.read_csv("outputs/hub_segments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    new_raw = pd.read_csv("data/new_shipments.csv")

    # Clean and Engineer Features
    clean_new, _ = clean_shipments(new_raw, medians=medians)
    featured_new = build_features(clean_new, hubs)

    # Reorder Features
    X_new = featured_new[feature_order]

    # Predict Probabilities
    probs = model.predict_proba(X_new)[:, 1]
    featured_new["delay_probability"] = probs.round(4)
    featured_new["flagged"] = (featured_new["delay_probability"] >= threshold).astype(int)

    # Merge Hub Segment
    featured_new = featured_new.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")

    # Operational Action Rules
    def assign_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            return "Proactive customer call"
        return "No action"

    featured_new["action"] = featured_new.apply(assign_action, axis=1)

    out_df = featured_new[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    out_df.to_csv("outputs/new_shipment_actions.csv", index=False)

if __name__ == "__main__":
    batch_score()