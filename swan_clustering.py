"""
SWAN Menopause Clustering Project
Month 3: Clustering Analysis
=====================================
Runs K-Means and hierarchical clustering on symptom profiles,
finds optimal cluster count, profiles clusters by demographics,
and generates UMAP visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

DATA_PATH = "swan_baseline_clean.csv"

SYMPTOM_VARS = [
    "HOTFLAS0",
    "NITESWE0",
    "WAKEUP0",
    "DEPRESS0",
    "IRRITAB0",
    "FORGET0",
    "DIZZY0",
    "BODYPAI0",
    "VAGINDR0",
]

COVARIATE_VARS = [
    "AGE0",
    "RACE",
    "STATUS0",
    "BMI0",
    "PREMEVE0",
]

RACE_LABELS = {
    1: "Black",
    2: "Chinese",
    3: "Japanese",
    4: "Hispanic",
    5: "White"
}

STATUS_LABELS = {
    1: "Post (BSO)",
    2: "Natural Post",
    3: "Late Peri",
    4: "Early Peri",
    5: "Premenopausal",
    6: "Pregnant/BF",
    7: "Unknown (HT)",
    8: "Unknown (Hyst)"
}

colors = ["#E8C547", "#7EC8A4", "#A67FB5", "#E07B5A",
          "#5B8DD9", "#4CAF82", "#E8847A", "#C8A96E"]

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

print("=" * 50)
print("STEP 1: Loading Clean Data")
print("=" * 50)

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Get available variables
symptom_cols = [v for v in SYMPTOM_VARS if v in df.columns]
covariate_cols = [v for v in COVARIATE_VARS if v in df.columns]
print(f"Symptom variables: {len(symptom_cols)}")
print(f"Covariate variables: {len(covariate_cols)}")

# ─────────────────────────────────────────
# 2. PREPARE FEATURES
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 2: Preparing Features")
print("=" * 50)

# Impute any remaining missing values with median
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(df[symptom_cols])
X_df = pd.DataFrame(X, columns=symptom_cols)

# Scale features — important for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature matrix shape: {X_scaled.shape}")
print("Missing values after imputation:", np.isnan(X_scaled).sum())

# ─────────────────────────────────────────
# 3. FIND OPTIMAL CLUSTER COUNT
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 3: Finding Optimal Cluster Count")
print("=" * 50)

k_range = range(2, 9)
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.3f}")

# Best k by silhouette score
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nBest k by silhouette score: {best_k}")

# ── Figure 1: Elbow + Silhouette ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#0F0F0F")

for ax in [ax1, ax2]:
    ax.set_facecolor("#141414")
    ax.tick_params(colors="#6A6460")
    ax.spines["bottom"].set_color("#2A2A2A")
    ax.spines["left"].set_color("#2A2A2A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax1.plot(list(k_range), inertias, "o-", color="#E8C547", linewidth=2, markersize=7)
ax1.set_title("Elbow Method", color="#F5F0E8", fontsize=12)
ax1.set_xlabel("Number of Clusters (k)", color="#C8C0B8")
ax1.set_ylabel("Inertia", color="#C8C0B8")
ax1.axvline(x=best_k, color="#7EC8A4", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
ax1.legend(facecolor="#1A1A1A", labelcolor="#C8C0B8")

ax2.plot(list(k_range), silhouette_scores, "o-", color="#A67FB5", linewidth=2, markersize=7)
ax2.set_title("Silhouette Scores", color="#F5F0E8", fontsize=12)
ax2.set_xlabel("Number of Clusters (k)", color="#C8C0B8")
ax2.set_ylabel("Silhouette Score", color="#C8C0B8")
ax2.axvline(x=best_k, color="#7EC8A4", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
ax2.legend(facecolor="#1A1A1A", labelcolor="#C8C0B8")

fig.suptitle("Optimal Cluster Count — SWAN Baseline", color="#F5F0E8", fontsize=13)
plt.tight_layout()
plt.savefig("fig5_optimal_clusters.png", dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
print("  ✓ Saved: fig5_optimal_clusters.png")
plt.close()

# ─────────────────────────────────────────
# 4. FIT FINAL K-MEANS MODEL
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print(f"STEP 4: Fitting K-Means (k={best_k})")
print("=" * 50)

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["CLUSTER"] = kmeans.fit_predict(X_scaled)

# Cluster sizes
print("\nCluster sizes:")
print(df["CLUSTER"].value_counts().sort_index().to_string())

# ─────────────────────────────────────────
# 5. PROFILE CLUSTERS
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 5: Profiling Clusters")
print("=" * 50)

# Mean symptom scores per cluster
cluster_profiles = df.groupby("CLUSTER")[symptom_cols].mean().round(2)
print("\nMean symptom scores per cluster:")
print(cluster_profiles.to_string())

# ── Figure 2: Cluster Symptom Profiles Heatmap ──
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("#0F0F0F")
ax.set_facecolor("#141414")

sns.heatmap(
    cluster_profiles,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=ax,
    linewidths=0.5,
    linecolor="#0F0F0F",
    annot_kws={"size": 9},
)
ax.set_title("Mean Symptom Scores by Cluster", color="#F5F0E8", fontsize=13, pad=12)
ax.set_ylabel("Cluster", color="#C8C0B8")
ax.set_xlabel("Symptom", color="#C8C0B8")
ax.tick_params(colors="#C8C0B8")

plt.tight_layout()
plt.savefig("fig6_cluster_profiles.png", dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
print("  ✓ Saved: fig6_cluster_profiles.png")
plt.close()

# ── Figure 3: Cluster Profiles Radar-style Bar Chart ──
fig, axes = plt.subplots(1, best_k, figsize=(5 * best_k, 5), sharey=True)
fig.patch.set_facecolor("#0F0F0F")
if best_k == 1:
    axes = [axes]

fig.suptitle("Symptom Profile per Cluster", color="#F5F0E8", fontsize=13)

for i, ax in enumerate(axes):
    ax.set_facecolor("#141414")
    vals = cluster_profiles.loc[i]
    ax.barh(symptom_cols, vals, color=colors[i], edgecolor="#0F0F0F")
    n = (df["CLUSTER"] == i).sum()
    ax.set_title(f"Cluster {i}\n(n={n})", color="#F5F0E8", fontsize=10)
    ax.tick_params(colors="#6A6460")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2A2A2A")
    ax.spines["left"].set_color("#2A2A2A")

plt.tight_layout()
plt.savefig("fig7_cluster_bars.png", dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
print("  ✓ Saved: fig7_cluster_bars.png")
plt.close()

# ─────────────────────────────────────────
# 6. DEMOGRAPHIC BREAKDOWN BY CLUSTER
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 6: Demographic Breakdown by Cluster")
print("=" * 50)

# Age and BMI by cluster
demo_summary = df.groupby("CLUSTER")[["AGE0", "BMI0"]].mean().round(2) if "AGE0" in df.columns else None
if demo_summary is not None:
    print("\nMean Age and BMI per cluster:")
    print(demo_summary.to_string())

# Race distribution by cluster
if "RACE" in df.columns:
    df["RACE_LABEL"] = df["RACE"].map(RACE_LABELS)
    race_dist = df.groupby("CLUSTER")["RACE_LABEL"].value_counts(normalize=True).mul(100).round(1)
    print("\nRace distribution per cluster (%):")
    print(race_dist.to_string())

    # ── Figure 4: Race by Cluster ──
    race_counts = df.groupby(["CLUSTER", "RACE_LABEL"]).size().unstack(fill_value=0)
    race_pct = race_counts.div(race_counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#141414")

    race_pct.plot(kind="bar", ax=ax, color=colors[:len(race_pct.columns)],
                  edgecolor="#0F0F0F", width=0.75)
    ax.set_title("Race/Ethnicity Distribution by Cluster (%)", color="#F5F0E8", fontsize=13)
    ax.set_xlabel("Cluster", color="#C8C0B8")
    ax.set_ylabel("Percentage (%)", color="#C8C0B8")
    ax.tick_params(colors="#6A6460", axis="x", rotation=0)
    ax.legend(title="Race", bbox_to_anchor=(1.01, 1), loc="upper left",
              facecolor="#1A1A1A", labelcolor="#C8C0B8", title_fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2A2A2A")
    ax.spines["left"].set_color("#2A2A2A")

    plt.tight_layout()
    plt.savefig("fig8_race_by_cluster.png", dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    print("  ✓ Saved: fig8_race_by_cluster.png")
    plt.close()

# Menopausal status by cluster
if "STATUS0" in df.columns:
    df["STATUS_LABEL"] = df["STATUS0"].map(STATUS_LABELS)
    status_dist = df.groupby("CLUSTER")["STATUS_LABEL"].value_counts(normalize=True).mul(100).round(1)
    print("\nMenopausal status per cluster (%):")
    print(status_dist.to_string())

# ─────────────────────────────────────────
# 7. UMAP VISUALIZATION
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 7: UMAP Visualization")
print("=" * 50)

try:
    import umap
    reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#141414")

    for i in range(best_k):
        mask = df["CLUSTER"] == i
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colors[i],
            label=f"Cluster {i} (n={mask.sum()})",
            alpha=0.6,
            s=8,
            edgecolors="none"
        )

    ax.set_title("UMAP — Menopause Symptom Clusters", color="#F5F0E8", fontsize=13)
    ax.set_xlabel("UMAP 1", color="#C8C0B8")
    ax.set_ylabel("UMAP 2", color="#C8C0B8")
    ax.tick_params(colors="#6A6460")
    ax.legend(facecolor="#1A1A1A", labelcolor="#C8C0B8", markerscale=3)
    ax.spines["bottom"].set_color("#2A2A2A")
    ax.spines["left"].set_color("#2A2A2A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("fig9_umap.png", dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    print("  ✓ Saved: fig9_umap.png")
    plt.close()

except ImportError:
    print("  ⚠️  umap-learn not installed. Run: pip install umap-learn")

# ─────────────────────────────────────────
# 8. SAVE FINAL DATASET WITH CLUSTER LABELS
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 8: Saving Final Dataset")
print("=" * 50)

df.to_csv("swan_clustered.csv", index=False)
print("  ✓ Saved: swan_clustered.csv")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n{'=' * 50}")
print("Month 3 clustering complete.")
print(f"Optimal clusters found: {best_k}")
print("Next step: Month 4 writeup and poster abstract.")
print("=" * 50)
