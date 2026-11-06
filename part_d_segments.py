import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Or set to your actual number of CPU cores
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist

def run_part_d():
    os.makedirs("outputs", exist_ok=True)

    # 1. Hub Profile
    df = pd.read_csv("data/modelling_table.csv")
    profile_df = df.groupby("hub_id").agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    ).reset_index().sort_values(by="hub_id").reset_index(drop=True)

    profile_df["mean_delivery_hours"] = profile_df["mean_delivery_hours"].round(4)
    profile_df["delay_rate"] = profile_df["delay_rate"].round(4)
    profile_df["mean_distance_km"] = profile_df["mean_distance_km"].round(4)
    profile_df["mean_n_stops"] = profile_df["mean_n_stops"].round(4)

    profile_df.to_csv("outputs/hub_profile.csv", index=False)

    # 2. Scaled Feature Matrix
    feat_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(profile_df[feat_cols])

    # 3. K-Means Separation Check (Across-to-Within Distance Ratio)
    sep_records = []
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        # Calculate cluster sizes
        counts = pd.Series(labels).value_counts().sort_values(ascending=False).values
        sizes_str = "-".join(map(str, counts))
        
        # Calculate across-to-within ratio over 276 pairs
        dist_matrix = pdist(X_scaled, metric="euclidean")
        n_hubs = len(X_scaled)
        
        within_dists = []
        across_dists = []
        
        idx = 0
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = dist_matrix[idx]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)
                idx += 1
                
        ratio = np.mean(across_dists) / np.mean(within_dists)
        sep_records.append({
            "K": k,
            "across_over_within": round(float(ratio), 4),
            "cluster_sizes": sizes_str
        })

    pd.DataFrame(sep_records).to_csv("outputs/part_d_separation.csv", index=False)

    # 4. K=3 Playbook Segmentation
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    profile_df["raw_cluster"] = km3.fit_predict(X_scaled)

    # Map raw clusters to delay-rate ordered names
    cluster_delay = profile_df.groupby("raw_cluster")["delay_rate"].mean().sort_values()
    name_map = {
        cluster_delay.index[0]: "Standard",
        cluster_delay.index[1]: "Watch",
        cluster_delay.index[2]: "Escalate"
    }
    profile_df["segment"] = profile_df["raw_cluster"].map(name_map)

    # Merge Hub Info
    hubs = pd.read_csv("data/hubs.csv")
    hub_segments = profile_df.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")
    hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]]
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)

    # Segment Profile Summary
    seg_prof = profile_df.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    ).loc[["Standard", "Watch", "Escalate"]].reset_index()

    seg_prof["mean_delivery_hours"] = seg_prof["mean_delivery_hours"].round(4)
    seg_prof["delay_rate"] = seg_prof["delay_rate"].round(4)
    seg_prof["mean_distance_km"] = seg_prof["mean_distance_km"].round(4)
    seg_prof["mean_n_stops"] = seg_prof["mean_n_stops"].round(4)
    seg_prof.to_csv("outputs/segment_profile.csv", index=False)

    # Tier Mismatches
    tier_implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
    mismatches = hub_segments[hub_segments["segment"] != hub_segments["hub_tier"].map(tier_implied)]
    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

if __name__ == "__main__":
    run_part_d()