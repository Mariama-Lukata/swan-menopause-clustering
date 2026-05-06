"""
SWAN Menopause Clustering Project
Month 2: Exploratory Data Analysis
=====================================
Replace the variable names in SYMPTOM_VARS and COVARIATE_VARS
with the exact column names from your codebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. CONFIGURATION — update these with your real variable names
# ─────────────────────────────────────────

DATA_PATH = "data/baseline-0001-Data.tsv"       # update to your actual file path
MISSING_CODES = [-9, -8, -1, 99, 999, -9.00] # update based on your codebook

# Replace these with your actual variable names from the codebook
SYMPTOM_VARS = [
    "HOTFLAS0",   # hot flash frequency/severity
    "NITESWE0",   # night sweats
    "WAKEUP0",   # sleep disturbance
    "DEPRESS0",
    "IRRITAB0",
    "FORGET0",      # depressed mood
    "DIZZY0",    # irritability
    "BODYPAI0",    # body pain
    "VAGINDR0",     # vaginal dryness
]

COVARIATE_VARS = [
    "AGE0",        # participant age
    "RACE",       # race/ethnicity
    "PREMEVE0",   # menopausal status
    "BMI0",  
    "STATUS0"      # body mass index
]

# ─────────────────────────────────────────
# 2. LOAD & CLEAN
# ─────────────────────────────────────────

print("=" * 50)
print("STEP 1: Loading Data")
print("=" * 50)

df = pd.read_csv(DATA_PATH, sep="\t", low_memory=False)
print(f"Raw dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Select only the variables you need
all_vars = SYMPTOM_VARS + COVARIATE_VARS
# Keep only columns that actually exist in the dataset
available = [v for v in all_vars if v in df.columns]
missing_cols = [v for v in all_vars if v not in df.columns]

if missing_cols:
    print(f"\n These variable names weren't found — check your codebook spelling:")
    for col in missing_cols:
        print(f"   - {col}")

df = df[available].copy()
print(f"\nWorking dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Replace missing value codes with NaN
print("\n" + "=" * 50)
print("STEP 2: Handling Missing Values")
print("=" * 50)

    
df = df.replace(MISSING_CODES, np.nan)
for var in SYMPTOM_VARS:
    if var in df.columns:
        df[var] = pd.to_numeric(df[var], errors="coerce")
        df[var] = df[var].replace(MISSING_CODES, np.nan)
for var in COVARIATE_VARS:
    if var in df.columns:
        df[var] = pd.to_numeric(df[var], errors="coerce")
        df[var] = df[var].replace(MISSING_CODES, np.nan)

        

# Missing value summary
missing_summary = df.isnull().sum()
missing_pct = (missing_summary / len(df) * 100).round(1)
missing_df = pd.DataFrame({
    "Missing Count": missing_summary,
    "Missing %": missing_pct
}).sort_values("Missing %", ascending=False)

print("\nMissing values per variable:")
print(missing_df[missing_df["Missing Count"] > 0].to_string())

# Drop rows missing more than half the symptom variables
symptom_available = [v for v in SYMPTOM_VARS if v in df.columns]
df_clean = df.dropna(subset=symptom_available, thresh=len(symptom_available) // 2)
print(f"\nRows after dropping incomplete symptom records: {len(df_clean)}")
print(f"Rows removed: {len(df) - len(df_clean)}")


# ─────────────────────────────────────────
# 3. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 3: Descriptive Statistics")
print("=" * 50)

print("\n── Symptom Variables ──")
print(df_clean[symptom_available].describe().round(2).to_string())

covariate_available = [v for v in COVARIATE_VARS if v in df_clean.columns]
print("\n── Covariate Variables ──")
print(df_clean[covariate_available].describe().round(2).to_string())

# Symptom prevalence (% who reported each symptom above 0)
print("\n── Symptom Prevalence (% reporting any symptom) ──")
for var in symptom_available:
    pct = (df_clean[var] > 0).sum() / df_clean[var].notna().sum() * 100
    print(f"  {var:<12}: {pct:.1f}%")

# ─────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 4: Generating Visualizations")
print("=" * 50)

# Set style
sns.set_theme(style="dark", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#0F0F0F",
    "axes.facecolor": "#141414",
    "axes.labelcolor": "#C8C0B8",
    "xtick.color": "#6A6460",
    "ytick.color": "#6A6460",
    "text.color": "#C8C0B8",
    "grid.color": "#1E1E1E",
})

# ── Figure 1: Symptom Score Distributions ──
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Symptom Score Distributions — SWAN Baseline", 
             fontsize=14, color="#F5F0E8", y=1.01)
axes = axes.flatten()

colors = ["#E8C547", "#7EC8A4", "#A67FB5", "#E07B5A",
          "#5B8DD9", "#4CAF82", "#E8847A", "#C8A96E"]

for i, var in enumerate(symptom_available[:8]):
    ax = axes[i]
    data = df_clean[var].dropna()
    ax.hist(data, bins=range(int(data.min()), int(data.max()) + 2),
            color=colors[i], alpha=0.85, edgecolor="#0F0F0F", linewidth=0.5)
    ax.set_title(var, fontsize=10, color="#F5F0E8")
    ax.set_xlabel("Score", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)

# Hide unused subplots
for j in range(len(symptom_available), 8):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("fig1_symptom_distributions.png", dpi=150, bbox_inches="tight",
            facecolor="#0F0F0F")
print("  ✓ Saved: fig1_symptom_distributions.png")
plt.close()

# ── Figure 2: Symptom Co-occurrence Heatmap ──
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("#0F0F0F")
ax.set_facecolor("#141414")

corr = df_clean[symptom_available].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=ax,
    linewidths=0.5,
    linecolor="#0F0F0F",
    annot_kws={"size": 9},
    vmin=0, vmax=1,
)
ax.set_title("Symptom Co-occurrence Correlation Matrix",
             fontsize=13, color="#F5F0E8", pad=15)
plt.tight_layout()
plt.savefig("fig2_symptom_correlation.png", dpi=150, bbox_inches="tight",
            facecolor="#0F0F0F")
print("  ✓ Saved: fig2_symptom_correlation.png")
plt.close()

# ── FIGURE 3: Symptoms by Menopausal Status ──
if "STATUS0" in df_clean.columns:
    meno_labels = {
        1: "Post (BSO)",
        2: "Natural Post",
        3: "Late Peri",
        4: "Early Peri",
        5: "Premenopausal",
        6: "Pregnant/Breastfeeding",
        7: "Unknown (HT Use)",
        8: "Unknown (Hysterectomy)"
    }
    df_plot = df_clean.copy()
    df_plot["STATUS0"] = df_plot["STATUS0"].map(meno_labels)

    # Find which statuses actually exist in the data
    existing_statuses = df_plot["STATUS0"].dropna().unique().tolist()
    print(f"Menopausal statuses found: {existing_statuses}")

    fig, axes = plt.subplots(1, len(symptom_available[:4]), figsize=(16, 5))
    fig.suptitle("Mean Symptom Score by Menopausal Status",
                 fontsize=13, color="#F5F0E8")

    for i, var in enumerate(symptom_available[:4]):
        ax = axes[i]
        means = df_plot.groupby("STATUS0")[var].mean()
        means.plot(kind="bar", ax=ax, color=colors[i], edgecolor="#0F0F0F")
        ax.set_title(var, fontsize=10, color="#F5F0E8")
        ax.set_ylabel("Mean Score", fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("fig3_symptoms_by_meno_status.png", dpi=150,
                bbox_inches="tight", facecolor="#0F0F0F")
    print("  ✓ Saved: fig3_symptoms_by_meno_status.png")
    plt.close()
else:
    print("  ⚠️  STATUS0 not found — skipping menopausal status plot")


        
# ── Figure 4: Symptoms by Race ──
if "RACE" in df_clean.columns:
    race_labels = {
        1: "Black", 2: "Chinese", 3: "Japanese",
        4: "Hispanic", 5: "White"
    }
    df_plot = df_clean.copy()
    df_plot["RACE"] = df_plot["RACE"].map(race_labels)

    # Mean symptom score by race
    race_means = df_plot.groupby("RACE")[symptom_available].mean()

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#141414")

    race_means.T.plot(kind="bar", ax=ax, color=colors[:len(race_means)],
                      width=0.75, edgecolor="#0F0F0F")
    ax.set_title("Mean Symptom Scores by Race/Ethnicity",
                 fontsize=13, color="#F5F0E8", pad=12)
    ax.set_xlabel("Symptom", fontsize=10)
    ax.set_ylabel("Mean Score", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Race", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig("fig4_symptoms_by_race.png", dpi=150,
                bbox_inches="tight", facecolor="#0F0F0F")
    print("  ✓ Saved: fig4_symptoms_by_race.png")
    plt.close()
else:
    print("  ⚠️  RACE not found — skipping race breakdown plot")

# ─────────────────────────────────────────
# 5. SAVE CLEAN DATASET FOR MONTH 3
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("STEP 5: Saving Clean Dataset")
print("=" * 50)

df_clean.to_csv("swan_baseline_clean.csv", index=False)
print(f"  ✓ Saved: swan_baseline_clean.csv")
print(f"  Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print(f"\n{'=' * 50}")
print("Month 2 EDA complete. Figures saved to your working directory.")
print("Next step: Month 3 clustering analysis.")
print("=" * 50)
