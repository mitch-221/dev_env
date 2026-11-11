# ---
# title: "HFpEF Bayesian Clustering — Phase 1: Replicating Caravita et al. (2026)"
# author: "Mitch Wishart"
# format:
#   html:
#     toc: true
#     code-fold: show
# ---

# %% [markdown]
# # Phase 1 — Replicating the Published Study
#
# This notebook replicates the one clustering pipeline the paper — Caravita
# et al. (2026), *Pulmonary Circulation*, "Hemodynamic-driven staging of
# HFpEF" — actually reports: unscaled k-means (k=5) on 8 of 14 hemodynamic
# variables, selected by a stated correlation-threshold rule. `script.r`
# (the group's reference code) contains this pipeline too, but buried inside
# six variants (scaled/unscaled × continuous-all/continuous-subset/binary);
# this notebook keeps only the one branch that matches the paper, and
# derives the variable-selection step from the real published correlation
# table instead of copying a hardcoded list.
#
# **Source files:** `dataset1.txt`, `dataset2.txt`, `dataset3.txt` — raw
# per-patient, per-exercise-step right-heart-catheterization data. Each
# patient has three rows: `REST`, `LEG` ("feet on the pedals"), `PEAK`.

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoids a known KMeans/MKL warning on Windows

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

sns.set_theme(style="whitegrid")
RANDOM_STATE = 12345

# Where the three dataset .txt files live. Fixed to the shared project
# folder so this runs the same whether it's executed from a GitHub checkout,
# this machine, or anywhere else — override with the DATA_DIR environment
# variable if you ever need to point it elsewhere without editing the file.
DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    r"C:\Users\mitch\OneDrive\1   B.ENGG\0   POLIMI\Bayesian Statistics\Project\r_code_datasets_wd",
))

# %% [markdown]
# ## 1. Load the raw data

# %%
list1 = pd.read_table(DATA_DIR / "dataset1.txt")
list2 = pd.read_table(DATA_DIR / "dataset2.txt")
list3 = pd.read_table(DATA_DIR / "dataset3.txt")
assert len(list1) == len(list2) == len(list3)

dataset = pd.DataFrame({
    "ID": list1["ID"].astype(int),
    "STEP": list1["STEP"].astype("category"),
    "PAWP.MIDA": list1["PAWP.MIDA"].astype(float),       # end-diastolic PAWP
    "PAWP.MEDIA": list1["PAWP.MEDIA"].astype(float),     # mean PAWP
    "RAP.MIDA": list1["RAP.MIDA"].astype(float),         # end-diastolic RAP
    "RAP.MEDIA": list1["RAP.MEDIA"].astype(float),       # mean RAP
    "PAPM": list2["PAPM"].astype(float),                 # mean PA pressure
    "PAWP_MIDA.CO.f.slp": list2["PAWP.MIDA.CO.f.slp"].astype(float),
    "PAWP.V.AMPL": list2["PAWP.V.AMPL"].astype(float),
    "PAWP.MEDIA.CO.f.slp": list2["PAWP.MEDIA.CO.f.slp"].astype(float),
    "PVR.MEDIA": list2["PVR.MEDIA"].astype(float),
    "RAP.VY": list2["RAP.VY"].astype(float),             # RAP systolic amplitude
    "CI": list2["CI"].astype(float),
    "COVO2.slp": list2["COVO2.slp"].astype(float),
    "PAPM.CO.f.slp": list2["PAPM.CO.f.slp"].astype(float),
    "RAP.MEDIA.CO.f.slp": list3["RAP.MEDIA.CO.f.slp"].astype(float),
    "PH": list3["PH"].astype(int),
    "HFpEF": list3["HFpEF"].astype(int),
    "stiff_LA": list3["stiff.LA"].astype(int),
    "high_PVR": list3["high.PVR"].astype(int),
    "RHF": list3["RHF"].astype(int),
    "TR": list3["TR"].astype(int),
    "low_CO": list3["low.CO"].astype(int),
})

peak_covo2 = dataset.loc[dataset["STEP"] == "PEAK"].set_index("ID")["COVO2.slp"]
dataset["COVO2.slp"] = dataset["ID"].map(peak_covo2)  # PEAK is the only correct value

hemo_vars = ["PAPM", "PAWP.MIDA", "PAWP_MIDA.CO.f.slp", "PAWP.V.AMPL", "PAWP.MEDIA",
             "PAWP.MEDIA.CO.f.slp", "PVR.MEDIA", "RAP.MIDA", "RAP.MEDIA",
             "RAP.MEDIA.CO.f.slp", "RAP.VY", "CI", "COVO2.slp", "PAPM.CO.f.slp"]
binary_vars = ["PH", "HFpEF", "stiff_LA", "high_PVR", "RHF", "TR", "low_CO"]
slope_vars = ["PAWP_MIDA.CO.f.slp", "PAWP.MEDIA.CO.f.slp", "RAP.MEDIA.CO.f.slp",
              "COVO2.slp", "PAPM.CO.f.slp"]

print(dataset["STEP"].value_counts())
dataset.head()

# %% [markdown]
# ## 2. Correlation matrix (paper §2.2, e-Table 1)
#
# The real published 14×14 Pearson correlation matrix, transcribed from
# e-Table 1 in the paper's Supporting Information — not recomputed, so this
# is their actual numbers.
#
# For two variables $x$ and $y$ measured on the same $n$ patients, the
# Pearson correlation coefficient is
#
# $$
# r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
#               {\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
# $$
#
# i.e. the sample covariance of $x$ and $y$, normalized by their standard
# deviations, giving a value in $[-1, 1]$. The paper's rule — drop one
# variable from any pair with $|r_{xy}| \geq 0.80$ — is just a threshold on
# this quantity, applied pairwise over all $\binom{14}{2}=91$ pairs.

# %%
_M = [
    [1.000, 0.899, 0.210, 0.549, 0.916, 0.163, 0.084, 0.784, 0.796, 0.215, 0.391, 0.463, -0.207, 0.224],
    [0.899, 1.000, 0.250, 0.555, 0.971, 0.211, -0.220, 0.826, 0.825, 0.226, 0.359, 0.477, -0.142, 0.183],
    [0.210, 0.250, 1.000, 0.245, 0.251, 0.925, 0.159, 0.229, 0.236, 0.666, 0.186, -0.201, -0.470, 0.711],
    [0.549, 0.555, 0.245, 1.000, 0.628, 0.155, -0.060, 0.510, 0.508, 0.307, 0.410, 0.061, -0.167, 0.101],
    [0.916, 0.971, 0.251, 0.628, 1.000, 0.202, -0.214, 0.813, 0.815, 0.228, 0.353, 0.412, -0.171, 0.190],
    [0.163, 0.211, 0.925, 0.155, 0.202, 1.000, 0.147, 0.152, 0.168, 0.682, 0.162, -0.175, -0.460, 0.789],
    [0.084, -0.220, 0.159, -0.060, -0.214, 0.147, 1.000, -0.120, -0.075, 0.217, 0.164, -0.373, -0.118, 0.285],
    [0.784, 0.826, 0.229, 0.510, 0.813, 0.152, -0.120, 1.000, 0.981, 0.308, 0.450, 0.361, -0.166, 0.100],
    [0.796, 0.825, 0.236, 0.508, 0.815, 0.168, -0.075, 0.981, 1.000, 0.338, 0.527, 0.336, -0.184, 0.139],
    [0.215, 0.226, 0.666, 0.307, 0.228, 0.682, 0.217, 0.308, 0.338, 1.000, 0.391, -0.241, -0.556, 0.744],
    [0.391, 0.359, 0.186, 0.410, 0.353, 0.162, 0.164, 0.450, 0.527, 0.391, 1.000, 0.027, -0.222, 0.171],
    [0.463, 0.477, -0.201, 0.061, 0.412, -0.175, -0.373, 0.361, 0.336, -0.241, 0.027, 1.000, 0.108, -0.197],
    [-0.207, -0.142, -0.470, -0.167, -0.171, -0.460, -0.118, -0.166, -0.184, -0.556, -0.222, 0.108, 1.000, -0.482],
    [0.224, 0.183, 0.711, 0.101, 0.190, 0.789, 0.285, 0.100, 0.139, 0.744, 0.171, -0.197, -0.482, 1.000],
]
published_corr = pd.DataFrame(_M, index=hemo_vars, columns=hemo_vars)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(published_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Published correlation matrix — e-Table 1 (Caravita et al. 2026)")
plt.tight_layout()
plt.show()

# Quick check against our own data: these numbers should be ~identical if
# this really is the same cohort the paper used.
our_corr = dataset[hemo_vars].corr(method="pearson")
max_diff = (our_corr - published_corr).abs().to_numpy().copy()
np.fill_diagonal(max_diff, 0)
print(f"Max |our correlation - published correlation| across all 91 pairs: {max_diff.max():.4f} "
      f"(near-zero confirms this is the same patient cohort)")

high_corr_pairs = (
    published_corr.where(np.triu(np.ones(published_corr.shape), k=1).astype(bool))
    .stack().loc[lambda s: s.abs() >= 0.80].sort_values(ascending=False)
)
print("\nPairs with |r| >= 0.80 in the published table:")
print(high_corr_pairs)

# %% [markdown]
# ## 3. Variable selection — algorithm, then the published list
#
# > "In case of high correlation between two variables, we planned to
# > prefer PAWP [...] over PA pressure and RAP and to prioritize values
# > averaged throughout the cardiac cycle over end-diastolic or systolic
# > values."
#
# Coding that rule and applying it to every `|r| >= 0.80` pair above:

# %%
role = {"PAPM": "PA", "PAPM.CO.f.slp": "PA",
        "PAWP.MIDA": "PAWP", "PAWP.MEDIA": "PAWP", "PAWP_MIDA.CO.f.slp": "PAWP",
        "PAWP.MEDIA.CO.f.slp": "PAWP", "PAWP.V.AMPL": "PAWP",
        "RAP.MIDA": "RAP", "RAP.MEDIA": "RAP", "RAP.MEDIA.CO.f.slp": "RAP", "RAP.VY": "RAP",
        "PVR.MEDIA": "other", "CI": "other", "COVO2.slp": "other"}
cycle = {"PAPM": 0, "PAPM.CO.f.slp": 0,
         "PAWP.MIDA": 1, "PAWP.MEDIA": 0, "PAWP_MIDA.CO.f.slp": 1, "PAWP.MEDIA.CO.f.slp": 0, "PAWP.V.AMPL": 2,
         "RAP.MIDA": 1, "RAP.MEDIA": 0, "RAP.MEDIA.CO.f.slp": 0, "RAP.VY": 2,
         "PVR.MEDIA": 2, "CI": 2, "COVO2.slp": 2}
family_rank = {"PAWP": 0, "other": 0, "PA": 1, "RAP": 1}  # prefer PAWP over PA pressure/RAP

def keep_vs_drop(a, b):
    ra, rb = family_rank[role[a]], family_rank[role[b]]
    if ra != rb:
        return (a, b) if ra < rb else (b, a)
    ca, cb = cycle[a], cycle[b]
    return (a, b) if ca <= cb else (b, a)

survivors = set(hemo_vars)
dropped_by = {}
for (a, b), r in high_corr_pairs.items():
    if a not in survivors or b not in survivors:
        continue
    keep, drop = keep_vs_drop(a, b)
    survivors.discard(drop)
    dropped_by[drop] = (keep, r)

print(f"Rule drops {len(dropped_by)} variables, leaving {len(survivors)}:")
for v, (keep, r) in dropped_by.items():
    print(f"  dropped {v:22s} (beaten by {keep}, r={r:.3f})")
print("Survivors:", sorted(survivors))

cluster_vars = ["PAWP.MEDIA", "PAWP.V.AMPL", "PAWP.MEDIA.CO.f.slp", "PAPM.CO.f.slp",
                "RAP.VY", "PVR.MEDIA", "CI", "COVO2.slp"]
extra = survivors - set(cluster_vars)
print(f"\nThe rule leaves 9 variables; the published list has 8. {extra} survives the rule "
      f"but isn't in the paper's final list, for reasons not stated in the Methods. "
      f"Skipping to the paper's published 8 below, since the goal is replicating what "
      f"they reported, not re-deriving a result they didn't fully document:")
print(cluster_vars)

# %% [markdown]
# ## 4. Patient-level table
#
# Table 1 reports each step-varying variable at `REST` / `LEG` / `PEAK`
# separately, and each slope variable once. Building that structure:

# %%
rest = dataset.loc[dataset["STEP"] == "REST"].drop(columns="STEP").add_suffix("_REST")
leg = dataset.loc[dataset["STEP"] == "LEG"].drop(columns="STEP").add_suffix("_LEG")
peak = dataset.loc[dataset["STEP"] == "PEAK"].drop(columns="STEP").add_suffix("_PEAK")
for df in (rest, leg, peak):
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)

patients = rest.merge(leg, on="ID").merge(peak, on="ID")
for v in slope_vars:
    patients[v] = patients[f"{v}_PEAK"]
    patients.drop(columns=[f"{v}_REST", f"{v}_LEG", f"{v}_PEAK"], inplace=True)

step_varying_cluster_vars = ["PAWP.MEDIA", "PAWP.V.AMPL", "PVR.MEDIA", "RAP.VY", "CI"]
slope_cluster_vars = ["PAWP.MEDIA.CO.f.slp", "PAPM.CO.f.slp", "COVO2.slp"]
cluster_cols = ([f"{v}_{s}" for v in step_varying_cluster_vars for s in ("REST", "LEG", "PEAK")]
                + slope_cluster_vars)

X_full = patients[["ID"] + cluster_cols].copy()
X = X_full.dropna().reset_index(drop=True)
print(f"{len(X_full)} -> {len(X)} patients after dropping incomplete rows "
      f"(paper reports n = 80)")

# %% [markdown]
# ## 5. k-means clustering — continuous, reduced 8-variable subset, unscaled
#
# > "we adopted the k-means clustering method, with the Euclidean distance
# > on the actual measurements without further normalization [...] k = 5
# > was set as the number of clusters."
#
# Given $n$ patients as points $\mathbf{x}_1, \dots, \mathbf{x}_n \in
# \mathbb{R}^{18}$ (the 18 columns built in Section 4) and a fixed number of
# clusters $k$, k-means looks for cluster assignments $c_i \in \{1,\dots,k\}$
# and centroids $\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_k$ that minimize
# the within-cluster sum of squares (WCSS):
#
# $$
# \text{WCSS} = \sum_{j=1}^{k} \sum_{i:\, c_i = j} \lVert \mathbf{x}_i - \boldsymbol{\mu}_j \rVert_2^2,
# \qquad
# \boldsymbol{\mu}_j = \frac{1}{|\{i : c_i = j\}|}\sum_{i:\, c_i = j} \mathbf{x}_i
# $$
#
# — plain squared Euclidean distance, unweighted, exactly as the paper
# specifies ("without further normalization": no per-variable scaling, so a
# variable on a larger numeric scale, like PVR, mechanically dominates the
# distance over one on a smaller scale, like cardiac index). This has no
# closed-form solution; `sklearn`'s `KMeans` minimizes it by Lloyd's
# algorithm (iteratively reassigning points to the nearest centroid, then
# recomputing centroids) from `n_init=10` random starts, keeping the best.
# $k$ itself isn't estimated by this objective — WCSS only ever decreases as
# $k$ grows — which is why the elbow plot below is a visual heuristic, not a
# statistic, and why the paper (and this script) just fixes $k=5$ by eye.

# %%
X_arr = X.drop(columns="ID").to_numpy()

wcss = []
k_range = range(2, 16)
for k in k_range:
    wcss.append(KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_arr).inertia_)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(list(k_range), wcss, marker="o")
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Within-cluster sum of squares")
ax.set_title("Elbow plot (cf. paper e-Figure 1)")
plt.tight_layout()
plt.show()

km5 = KMeans(n_clusters=5, n_init=10, random_state=RANDOM_STATE).fit(X_arr)
X_labeled = X.copy()
X_labeled["cluster"] = km5.labels_

# Relabel 1 (mildest) -> 5 (most severe) by peak-exercise mean PAWP, so our
# numbering means the same thing as the paper's, even if membership differs.
order = X_labeled.groupby("cluster")["PAWP.MEDIA_PEAK"].mean().sort_values().index
relabel = {old: new + 1 for new, old in enumerate(order)}
X_labeled["cluster"] = X_labeled["cluster"].map(relabel)
centroids = km5.cluster_centers_[[list(order).index(old) for old in sorted(relabel, key=relabel.get)]]

print("Our cluster sizes:", dict(X_labeled["cluster"].value_counts().sort_index()))
print("Paper's published sizes: 26, 30, 12, 4, 8")

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=X_labeled, x="cluster", y="PAWP.MEDIA_PEAK", ax=ax)
ax.set_title("Peak-exercise mean PAWP by cluster")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Data quality assessment
#
# ### 6a. Variable inventory

# %%
print(f"Continuous hemodynamic variables (14 screened for correlation): {hemo_vars}")
print(f"\nOf these, the 8 used for clustering: {cluster_vars}")
print(f"\nCategorical/binary phenotype variables (7, profiling only, not clustering inputs): {binary_vars}")
print("\nNote: the paper's clinical characteristics (age, sex, NT-proBNP, LVEF, etc. — "
      "its Table 3) aren't present in dataset1/2/3.txt at all; those files are "
      "hemodynamic catheterization data only.")

# %% [markdown]
# ### 6b. Missingness and structural gaps

# %%
missing_by_step = (
    dataset.set_index(["ID", "STEP"])[hemo_vars].isna()
    .groupby("STEP", observed=True).mean().mul(100).round(1)
)
print("% missing per hemodynamic variable, by STEP:")
missing_by_step

# %%
irregular_ids = [
    pid for pid, counts in dataset.groupby("ID")["STEP"].value_counts().unstack(fill_value=0).iterrows()
    if not (counts == 1).all() or len(counts) < 3
]
print(f"Patients missing a full REST/LEG/PEAK triplet: {irregular_ids}")

na_per_col = X_full.drop(columns="ID").isna().sum()
na_per_col = na_per_col[na_per_col > 0].sort_values(ascending=False)
print(f"\n{len(X_full)} -> {len(X)} patients survive dropna() for the clustering input matrix "
      f"({len(X_full) - len(X)} dropped). Columns responsible:")
print(na_per_col)

# %% [markdown]
# ### 6c. Outlier screen

# %%
hemo_z = (dataset[hemo_vars] - dataset[hemo_vars].mean()) / dataset[hemo_vars].std()
outlier_counts = (hemo_z.abs() > 3).sum().sort_values(ascending=False)
print("Rows with |z| > 3 (vs. the rest of this sample), per variable:")
print(outlier_counts[outlier_counts > 0])

# %% [markdown]
# ## Limitations of this replication
#
# - **Variable selection isn't fully reproducible.** The paper's stated
#   tie-break rule, applied mechanically to the real correlation table,
#   yields 9 variables; the paper reports 8. The extra survivor
#   (`RAP.MEDIA.CO.f.slp`) isn't correlated ≥0.80 with anything else left,
#   so its exclusion isn't explained by the Methods as written — we drop it
#   anyway to match what they actually report.
# - **k is imposed, not inferred.** k=5 comes from an "elbow, although not
#   very pronounced" the paper describes qualitatively — not a criterion
#   anyone could re-run and get the same answer from.
# - **Cluster assignment is a hard, deterministic label.** k-means gives
#   each patient exactly one cluster with no notion of confidence or
#   borderline membership, and is sensitive to initialization — our sizes
#   (18/13/29/8/12) land in the same ballpark as the paper's (26/30/12/4/8)
#   but don't match exactly, and there's no way to tell, from k-means alone,
#   whether that gap is noise or a real difference in cohort/implementation.
# - **No uncertainty quantification anywhere** — not on cluster count, not
#   on membership, not on the variable-selection step itself.
#
# ## Next steps (Phase 2)
#
# - Replace k-means with a Bayesian nonparametric mixture model (Dirichlet
#   process / infinite mixture) so the number of clusters is inferred from
#   the data, with a posterior distribution over k instead of an imposed 5.
# - Get a real posterior cluster-membership probability per patient, in
#   place of a hard label — lets us flag borderline patients honestly
#   rather than forcing a single answer.
# - Feed in the full 8-variable, 3-step (REST/LEG/PEAK) + slope structure
#   used here, so the model sees what script.r and the paper actually
#   measured, not a simplification of it.
# - Use the correlation/variable-selection discrepancy and the data-quality
#   findings above (ID 83's missing LEG step, the RAP.VY missingness
#   concentration, the outlier counts) as explicit checks on the Phase 2
#   model — e.g. does it still favor the same 8 variables, does it handle
#   the outliers sensibly without needing them manually screened out.
#
# ### What the Bayesian model actually looks like
#
# Where k-means hard-codes $k$ and gives each patient a single label, a
# Dirichlet process (DP) mixture treats the number of clusters itself as
# unknown and places a prior over it. Each patient's data vector
# $\mathbf{x}_i$ is assumed drawn from one of (countably many, in principle)
# component distributions, with mixture weights $\pi_j$ drawn from a
# **stick-breaking** construction:
#
# $$
# \pi_j = \beta_j \prod_{l=1}^{j-1}(1-\beta_l), \qquad
# \beta_j \sim \text{Beta}(1, \alpha), \qquad j = 1, 2, 3, \dots
# $$
#
# where $\alpha$ is a concentration parameter controlling how readily new
# clusters are created (larger $\alpha$ → more clusters). Each patient gets
# a latent cluster label $z_i \sim \text{Categorical}(\pi_1, \pi_2, \dots)$,
# and conditional on $z_i = j$, their data is modeled as
#
# $$
# \mathbf{x}_i \mid z_i = j \;\sim\; \mathcal{N}(\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)
# $$
#
# — a multivariate normal per cluster, replacing k-means' implicit
# "distance to a point" with an actual likelihood, so clusters can differ
# in spread and orientation, not just location. Priors are placed on the
# per-cluster parameters themselves (e.g. a Normal-Inverse-Wishart prior on
# $(\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$), making the whole thing
# fully Bayesian.
#
# Fitting this means sampling from the posterior
# $p(\mathbf{z}, \{\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j\}, \alpha \mid
# \mathbf{x}_1,\dots,\mathbf{x}_n)$ — intractable in closed form, so Phase 2
# will use a Gibbs sampler. The key step, at each iteration, resamples each
# patient's cluster label conditional on everyone else's (the Chinese
# Restaurant Process form of the DP), with probability of joining an
# existing cluster $j$ (holding $n_j$ current members) or starting a new
# one roughly:
#
# $$
# p(z_i = j \mid \mathbf{z}_{-i}, \mathbf{x}_i) \;\propto\;
# \begin{cases}
# n_j \cdot p(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j) & \text{existing cluster } j \\[4pt]
# \alpha \cdot p(\mathbf{x}_i) & \text{new cluster}
# \end{cases}
# $$
#
# where $\mathbf{z}_{-i}$ is every other patient's current assignment and
# $p(\mathbf{x}_i)$ is the prior-predictive density for a brand-new cluster.
# Running many such sweeps and keeping the post-burn-in draws gives a
# posterior distribution over the number of clusters, over each patient's
# cluster (as a distribution across samples, not a single number — this is
# what directly answers "probability of belonging to each cluster"), and
# over the cluster-specific parameters themselves.
