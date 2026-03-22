"""
Clustering model for research center quality classification

Author: Atefeh Momeni
Description:
This script trains a KMeans clustering model to group research centers
into quality tiers based on internal capacity and nearby facilities.
"""

import os
import json
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


FEATURES = [
    "internalFacilitiesCount",
    "hospitals_10km",
    "pharmacies_10km",
    "facilityDiversity_10km",
    "facilityDensity_10km",
]


DATA_PATH = "research_centers.csv"


df = pd.read_csv(DATA_PATH)


X = df[FEATURES]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters


score = silhouette_score(X_scaled, clusters)

print("Silhouette score:", score)


cluster_summary = df.groupby("cluster")[FEATURES].mean()

cluster_rank = (
    cluster_summary.mean(axis=1)
    .sort_values(ascending=False)
    .index
    .tolist()
)

cluster_mapping = {
    cluster_rank[0]: "Premium",
    cluster_rank[1]: "Standard",
    cluster_rank[2]: "Basic",
}

df["qualityTier"] = df["cluster"].map(cluster_mapping)


os.makedirs("model", exist_ok=True)

joblib.dump(kmeans, "model/kmeans_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

with open("model/cluster_mapping.json", "w") as f:
    json.dump(cluster_mapping, f)


df.to_csv("research_centers_with_clusters.csv", index=False)


print("Model trained and saved")
print("\nCluster summary:")
print(df.groupby("qualityTier")[FEATURES].mean())
print("\nCluster summary by tier:")
print(
    df.groupby("qualityTier")[
        [
            "internalFacilitiesCount",
            "hospitals_10km",
            "pharmacies_10km",
            "facilityDiversity_10km",
            "facilityDensity_10km",
        ]
    ].mean()
)