# ---
# title: "Bayesian Clustering for Data-Driven Clinical Staging in HFpEF"
# format:
#   html:
#     toc: true
#     code-fold: show
# ---

# %% [markdown]
# <style>
# pre, pre code, div.sourceCode, div.sourceCode pre, div.sourceCode code {
#   white-space: pre-wrap !important;
#   word-break: break-word !important;
#   overflow-x: visible !important;
# }
# </style>
#
# # Bayesian Clustering for Data-Driven Clinical Staging in HFpEF
#
# **Progress notes - replication & variable assessment &
# modelling strategy**
#
# *Mitch Wishart*

# %% [markdown]
# ::: {.callout-note}
# **Overview**
# We reproduced the original method (Caravita et al. (2026)) closely but not exactly
# (Phase 0), then went back to first principles on which variables and what model structure our own
# Bayesian nonparametric model should actually use, given $n=80$ patients (Phase 1).
# It was found that a full covariance matrix over the correlated continuous variables is numerically infeasible
# at this sample size - demonstrated below - which is the single biggest constraint shaping
# the whole model design (block-diagonal covariance instead of full or plain diagonal).
#
# This is the informal first look. We will convert to LaTeX for the presentation. This is also originally done in python.
# :::

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

sns.set_theme(style="whitegrid")
RANDOM_STATE = 12345
DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    r"C:\Users\mitch\OneDrive\1   B.ENGG\0   POLIMI\Bayesian Statistics\Project\r_code_datasets_wd",
))

# %% [markdown]
# # Phase 0 — Reproducing the Pipeline
#
# The raw files (`dataset1/2/3.txt`) concatenate column-wise into one long table: one row per
# exercise step per patient, `STEP` $\in$ {REST, LEG, PEAK}. The paper names 21 variables as clinically
# relevant (14 continuous + 7 binary), picked by domain judgement, then applies a redundancy rule
# (drop one of any pair correlated $|r|\geq0.80$) to get its published 8 continuous variables.

# %%
list1 = pd.read_table(DATA_DIR / "dataset1.txt")
list2 = pd.read_table(DATA_DIR / "dataset2.txt")
list3 = pd.read_table(DATA_DIR / "dataset3.txt")
assert len(list1) == len(list2) == len(list3)

dataset = pd.DataFrame({
    "ID": list1["ID"].astype(int),
    "STEP": list1["STEP"].astype("category"),
    "PAWP.MIDA": list1["PAWP.MIDA"].astype(float),
    "PAWP.MEDIA": list1["PAWP.MEDIA"].astype(float),
    "RAP.MIDA": list1["RAP.MIDA"].astype(float),
    "RAP.MEDIA": list1["RAP.MEDIA"].astype(float),
    "PAPM": list2["PAPM"].astype(float),
    "PAWP_MIDA.CO.f.slp": list2["PAWP.MIDA.CO.f.slp"].astype(float),
    "PAWP.V.AMPL": list2["PAWP.V.AMPL"].astype(float),
    "PAWP.MEDIA.CO.f.slp": list2["PAWP.MEDIA.CO.f.slp"].astype(float),
    "PVR.MEDIA": list2["PVR.MEDIA"].astype(float),
    "RAP.VY": list2["RAP.VY"].astype(float),
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

hemo_vars_14 = ["PAPM", "PAWP.MIDA", "PAWP_MIDA.CO.f.slp", "PAWP.V.AMPL", "PAWP.MEDIA",
                "PAWP.MEDIA.CO.f.slp", "PVR.MEDIA", "RAP.MIDA", "RAP.MEDIA",
                "RAP.MEDIA.CO.f.slp", "RAP.VY", "CI", "COVO2.slp", "PAPM.CO.f.slp"]
binary_vars_7 = ["PH", "HFpEF", "stiff_LA", "high_PVR", "RHF", "TR", "low_CO"]
print(f"n = {dataset['ID'].nunique()} patients, {len(dataset)} (patient, STEP) rows")
dataset.head()

# %% [markdown]
# ## Correlation matrix - basis for the redundancy screen
#
# Pooling all three STEP rows together reproduces the paper's own published correlation table
# almost exactly - e.g. PAPM$\leftrightarrow$PAWP.V.AMPL: published 0.549,
# our recomputation 0.549, mirroring how it was originally computed.

# %%
corr14 = dataset[hemo_vars_14].corr().round(3)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr14, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation matrix — paper's 14 clinically-relevant continuous variables")
plt.xticks(rotation=60, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

high_corr_pairs = (corr14.where(np.triu(np.ones(corr14.shape), k=1).astype(bool))
                   .stack().loc[lambda s: s.abs() >= 0.80].sort_values(ascending=False))
print(f"{len(high_corr_pairs)} pairs at |r|>=0.80")

# %% [markdown]
# ::: {.callout-note}
# Applying the paper's own rule to their own data gives **9** survivors, not the published **8** -
# RAP.MEDIA.CO.f.slp  has no correlation, |r|>0.80 (revisited properly in Phase 1).
# :::

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

survivors = set(hemo_vars_14)
dropped_by = {}
for (a, b), r in high_corr_pairs.items():
    if a not in survivors or b not in survivors:
        continue
    keep, drop = keep_vs_drop(a, b)
    survivors.discard(drop)
    dropped_by[drop] = (keep, r)

print(f"Rule drops {len(dropped_by)}, leaving {len(survivors)} survivors: {sorted(survivors)}")
paper_published_8 = ["PAWP.MEDIA", "PAWP.V.AMPL", "PAWP.MEDIA.CO.f.slp", "PAPM.CO.f.slp",
                      "RAP.VY", "PVR.MEDIA", "CI", "COVO2.slp"]
print(f"Paper's published 8: {sorted(paper_published_8)}")
print(f"Extra survivor the rule doesn't explain away: {sorted(survivors - set(paper_published_8))}")

# %% [markdown]
# ## $k$-means replication
#
# Pivoting to one row per patient (REST/LEG/PEAK become columns), dropping incomplete rows, gives
# $n=80$. Replicating the method using unscaled Euclidean $k$-means, $k=5$:

# %%
step_varying_8 = ["PAWP.MEDIA", "PAWP.V.AMPL", "RAP.VY", "PVR.MEDIA", "CI"]
slope_8 = ["PAWP.MEDIA.CO.f.slp", "PAPM.CO.f.slp", "COVO2.slp"]

rest = dataset.loc[dataset["STEP"] == "REST"].drop(columns="STEP").add_suffix("_REST")
leg = dataset.loc[dataset["STEP"] == "LEG"].drop(columns="STEP").add_suffix("_LEG")
peak = dataset.loc[dataset["STEP"] == "PEAK"].drop(columns="STEP").add_suffix("_PEAK")
for df in (rest, leg, peak):
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)
patients0 = rest.merge(leg, on="ID").merge(peak, on="ID")
for v in slope_8:
    patients0[v] = patients0[f"{v}_PEAK"]
cols0 = [f"{v}_{s}" for v in step_varying_8 for s in ("REST", "LEG", "PEAK")] + slope_8
patients0_full = patients0[["ID"] + cols0].dropna().reset_index(drop=True)
print(f"n = {len(patients0_full)} patients after pivot + dropna (paper: 80)")

Xk = patients0_full[cols0].to_numpy()
wss = [KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xk).inertia_ for k in range(2, 16)]

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(range(2, 16), wss, "o-", color="#2c6e91")
ax.axvline(5, color="red", ls="--", lw=1, label="paper's k=5")
ax.set_xlabel("k"); ax.set_ylabel("Within-cluster SS")
ax.set_title("k-means elbow (paper's 8 variables, unscaled)")
ax.legend()
plt.tight_layout()
plt.show()

km5 = KMeans(n_clusters=5, n_init=10, random_state=RANDOM_STATE).fit(Xk)
sizes5 = sorted(np.bincount(km5.labels_).tolist(), reverse=True)
paper_sizes = [26, 30, 12, 4, 8]
print(f"Paper (published) sizes: {sorted(paper_sizes, reverse=True)}")
print(f"Our replication sizes:   {sizes5}")

fig, ax = plt.subplots(figsize=(5.4, 3.6))
x = np.arange(5)
ax.bar(x - 0.18, sorted(paper_sizes, reverse=True), width=0.36, label="Paper (published)")
ax.bar(x + 0.18, sizes5, width=0.36, label="Our replication")
ax.set_xticks(x); ax.set_xticklabels([f"C{i+1}" for i in x])
ax.set_ylabel("Cluster size (n)")
ax.set_title("k=5 cluster sizes: paper vs. replication")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Same $n$, same variables, same unscaled Euclidean $k$-means - membership still differs. Most
# plausible cause: $k$-means' sensitivity to random initialization and to unscaled distance,
# neither of which we could fully pin down from the published method.
#
# **What Phase 0 established, and its limits.** Two of the paper's own limitations become concrete
# rather than assumed: unscaled Euclidean distance lets one variable dominate cluster assignment, and
# $k$-means gives zero uncertainty - a patient is in exactly one cluster.
# Also worth flagging: the paper's binary/categorical variables are not used anywhere
# in its actual clustering step - only for post-hoc profiling afterwards.The two-block mixed-data
# model will be an extension of the paper's method, not a replication.

# %% [markdown]
# # Phase 1 - Variable Assessment & Modelling Strategy
#
# Phase 1 investigates the **124** raw
# candidate columns across the three files and determines which our model should actually use.
# We considered what structure $n=80$ patients could actually support.
# %%
full = pd.concat([list1, list2, list3], axis=1)
candidate_cols = [c for c in full.columns if c not in ("ID", "STEP")]
binary_vars_7_raw = ["PH", "HFpEF", "stiff.LA", "high.PVR", "RHF", "TR", "low.CO"]  # raw-file names (dots)
print(f"{len(candidate_cols)} raw candidate columns, {full['ID'].nunique()} patients, {len(full)} rows")

# %% [markdown]
# ## Data structure & a concrete correction
#
# Continuous variables vary by STEP except 5 "slope" variables, PEAK-only by design.
# The 7 binary flags are not fixed patient labels either — their prevalence changes by exercise step,
# not a static diagnosis. Two further variables, `SESSO` and `RITMO`, are genuine patient-level
# constants rather than STEP-varying - but, unlike the slope variables, they turn out to be recorded
# only at REST in the raw files, checked directly below. (A third patient-level constant, `BA`,
# is picked up later in the categorical block.)

# %%
prevalence_by_step = full.groupby("STEP")[binary_vars_7_raw].mean().loc[["REST", "LEG", "PEAK"]]
print(prevalence_by_step.round(3))

fig, ax = plt.subplots(figsize=(7.5, 3.6))
prevalence_by_step.T.plot(kind="bar", ax=ax, color=["#8fb8de", "#f2b06d", "#d6604d"])
ax.set_ylabel("Prevalence"); ax.set_title("Binary-flag prevalence by exercise step")
ax.legend(title="STEP", loc="upper left", bbox_to_anchor=(1.01, 1))
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# HFpEF prevalence alone goes 40.4% (REST) $\to$ 88.8% (PEAK) - far too big a swing to be noise. These
# are exercise-dependent diagnostic classifications, not static labels.

# %%
for v in ["SESSO", "RITMO"]:
    by_step = full.groupby("STEP")[v].apply(lambda s: s.notna().sum())
    print(f"{v}: non-null rows by STEP -> {by_step.to_dict()}")

# %% [markdown]
# ## Why a blind correlation screen across all 124 columns fails

# %%
col_type = {}
for c in candidate_cols:
    numeric = pd.to_numeric(full[c], errors="coerce")
    if numeric.notna().sum() == 0:
        col_type[c] = "non_numeric"; continue
    n_unique = numeric.dropna().nunique()
    col_type[c] = "binary" if n_unique <= 2 else "continuous"

continuous_cols = [c for c, t in col_type.items() if t == "continuous"]
peak_mask = full["STEP"] == "PEAK"
slope_like = ["PAWP.MIDA.CO.f.slp", "PAWP.MEDIA.CO.f.slp", "RAP.MEDIA.CO.f.slp", "COVO2.slp", "PAPM.CO.f.slp"]
missing_pct = full[candidate_cols].isna().mean().mul(100)
for v in slope_like + binary_vars_7_raw:
    if v in missing_pct.index:
        missing_pct[v] = full.loc[peak_mask, v].isna().mean() * 100
missing_pct = missing_pct.round(1).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
missing_pct.head(20).plot(kind="barh", ax=ax, color="indianred")
ax.invert_yaxis()
ax.set_xlabel("% missing"); ax.set_title("Missingness — worst 20 of 124 candidate variables")
ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
plt.show()

usable_continuous = [c for c in continuous_cols if missing_pct.get(c, 100) < 15]
corr_all = full[usable_continuous].apply(pd.to_numeric, errors="coerce").corr()
high_corr_all = (corr_all.where(np.triu(np.ones(corr_all.shape), k=1).astype(bool))
                 .stack().loc[lambda s: s.abs() >= 0.80].sort_values(ascending=False))
survivors_naive = set(usable_continuous)
for (a, b), r in high_corr_all.items():
    if a not in survivors_naive or b not in survivors_naive:
        continue
    keep, drop = (a, b) if missing_pct[a] <= missing_pct[b] else (b, a)
    survivors_naive.discard(drop)

# "43 survivors" isn't 43 model dimensions: check whether each ACTUALLY VARIES by STEP per
# patient, not just whether it's non-null (some turn out to be the same value duplicated 3x).
real_dims = {}
for v in survivors_naive:
    sub = full[["ID", "STEP", v]].dropna(subset=[v])
    frac_constant = (sub.groupby("ID")[v].nunique() <= 1).mean()
    real_dims[v] = 1 if frac_constant > 0.9 else 3
d_naive_expanded = sum(real_dims.values())

print(f"Usable continuous (<15% missing): {len(usable_continuous)}")
print(f"Pairs at |r|>=0.80: {len(high_corr_all)}; naive survivors: {len(survivors_naive)}")
print(f"Real patient-level dimensionality once duplicated STEP columns are corrected: d = {d_naive_expanded}")
print(f"Even an EVEN 5-cluster split (80/5=16/cluster) is far short of the {d_naive_expanded+1} "
      f"needed just to invert a full covariance matrix at this d.")
print("Not usable at ANY covariance structure -> confirms falling back to the paper's "
      "clinically-reasoned 21 instead of a blind statistical screen.")

# %% [markdown]
# ## Falling back to the paper's variables
#
# The paper's 14 clinically-relevant candidates were taken as it was the only justifiable method
# to significantly reduce variables. Pure correlation screening and filtering by data availability left
# us with 43 continuous candidates - still too many. 6/8 of the paper's final variables passed the crude
# screening, indicating the other 2 were clinically-based decisions.

# %%
nine_survivors = sorted(survivors)  # computed in Phase 0 above, on the identical 14-variable pool
print(f"Mechanical rule on the 14: {nine_survivors}")
print(f"Paper's published 8:        {sorted(paper_published_8)}")
print(f"Extra survivor:              {sorted(set(nine_survivors) - set(paper_published_8))}")

# Hypothetical: what if the tie-break on the paper's own final 8 were domain-blind
# (missingness-based, against each variable's actual raw sibling in the dataset -- e.g.
# PAWP.MEDIA's alternative is PAWP.A, a column outside the named 14) instead of clinical?
# (Computed once against the raw missingness figures; recorded here rather than re-derived,
# since it compares against sibling columns outside the paper's own named 14-variable pool.)
tiebreak_results = {
    "PAWP.MEDIA": ("dropped", "PAWP.A", 0.959),
    "PAWP.V.AMPL": ("kept", None, None),
    "PAWP.MEDIA.CO.f.slp": ("kept", None, None),
    "PAPM.CO.f.slp": ("kept", None, None),
    "RAP.VY": ("dropped", "RAP.V.AMPL", 0.903),
    "PVR.MEDIA": ("kept", None, None),
    "CI": ("kept", None, None),
    "COVO2.slp": ("kept", None, None),
}
n_kept = sum(1 for v, (status, _, _) in tiebreak_results.items() if status == "kept")
for v, (status, beat_by, r) in tiebreak_results.items():
    line = f"  {v:22s} -> {status}"
    if beat_by:
        line += f" (beaten by {beat_by}, r={r:.3f})"
    print(line)
print(f"Blind tie-break would keep {n_kept}/8 of the published list unchanged.")
print(f"The other {8-n_kept}/8 (PAWP.MEDIA, RAP.VY) specifically needed clinical judgement.")
unchanged = {v for v, (status, _, _) in tiebreak_results.items() if status == "kept"}

# We decided to use the mechanically-derived 9, not the published 8.
# This makes the continuous block 19 patient-level
# dimensions (5 step-varying $\times$ 3 steps = 15, plus 4 slope-type $\times$ 1 = 4).
# :::
#
# ## Revised Binary/Categorical block
#
# `SESSO`, `RITMO`, `BA` are genuine, interpretable patient-level constants. `VO2.` (100%
# missing) and `INSP.EXP.AVG` (one distinct value in the whole dataset) stay excluded due to
# zero added information. No binary candidates beyond the paper's original 7
# exist anywhere in the raw files. **Categorical block:
# 10 variables.**

# %% [markdown]
# ## Full covariance matrix feasibility
#
# Patient sample-size limits the number of individuals per cluster and therefore limits parameter flexibility

# %%
def full_cov_params(d): return d * (d + 1) // 2 + d   # covariance + mean
def diag_cov_params(d): return 2 * d
def blockdiag_cov_params(n_blocks_3x3, n_slope_1d):
    return n_blocks_3x3 * (3 * 4 // 2 + 3) + n_slope_1d * 2

d_final_9 = 5 * 3 + 4   # our 9 variables, expanded across REST/LEG/PEAK
print(f"d = {d_final_9}")
print(f"full_cov_params({d_final_9}) = {full_cov_params(d_final_9)} params/cluster, "
      f"needs n_cluster >= {d_final_9+1} just to INVERT")
print(f"diag_cov_params({d_final_9}) = {diag_cov_params(d_final_9)} params/cluster, "
      f"needs only n_cluster >= 2")
print(f"blockdiag_cov_params = {blockdiag_cov_params(5, 4)} params/cluster, "
      f"needs only n_cluster >= 4 (per 3x3 block)")

kmeans_sizes_k5 = sizes5  # Phase 0's own k=5 clusters, computed above
print(f"\nPhase 0's own k=5 clusters: {kmeans_sizes_k5}")
print(f"Smallest cluster, n={min(kmeans_sizes_k5)} -- already below the {d_final_9+1} a full matrix "
      f"needs. A DP mixture that estimates k from the data routinely produces clusters of n=1-3.")

# Concrete demonstration: draw n=5 iid standard-normal points in d=19 dims, inspect their covariance.
rng = np.random.default_rng(0)
X_tiny = rng.normal(size=(5, d_final_9))
S = np.cov(X_tiny, rowvar=False)
rank, det, cond = np.linalg.matrix_rank(S), np.linalg.det(S), np.linalg.cond(S)
print("A Gaussian density divides by sqrt(det(Sigma)) - determinants close to zero makes that "
      "density numerically meaningless.")

# %% [markdown]
# In a fully Bayesian treatment with a proper
# Normal-Inverse-Wishart prior, the posterior will still be produced. But
# at these sample sizes the posterior is then almost entirely prior-driven. A full $19\times19$ covariance matrix
# is not usable for handling correlated continuous variables at this sample size.

# %%
comparison = pd.DataFrame({
    "Full (d=19)": [full_cov_params(d_final_9), d_final_9 + 1, "yes (if feasible)", "no"],
    "Diagonal (d=19)": [diag_cov_params(d_final_9), 2, "no", "yes"],
    "Block-diagonal (adopted)": [blockdiag_cov_params(5, 4), "4 (per 3x3 block)", "yes, within each variable", "yes"],
}, index=["Params / cluster", "Min. cluster size needed", "Models REST/LEG/PEAK correlation?", "Feasible at n=80?"])
comparison

# %% [markdown]
# Diagonal methods (every dimension independent) is numerically safe but fails to acknowledge
# REST/LEG/PEAK readings are three repeated measurements of the same quantity, not independent readings.
#
# Block-diagonal was found as a potential solution. Each of the 5 step-varying variables gets its own
# $3\times3$ covariance block - a multivariate Gaussian over just that one variable's
# REST/LEG/PEAK triplet via a Normal-Inverse-Wishart (NIW) conjugate update. Different variables'
# blocks stay independent of each other and the full $19\times19$ matrix is not required.

# %%
from scipy.stats import invwishart

p_blk = 3
m0_blk = np.zeros(p_blk)
kappa0_blk = 1.0
nu0_blk = p_blk + 2
Psi0_blk = np.eye(p_blk) * (nu0_blk - p_blk - 1)

def sample_niw_posterior(Y, rng):
    """NIW posterior draw for a 3-dim block, given Y: (n_h, 3) -- this cluster's data."""
    nj = len(Y)
    if nj == 0:
        Sigma = invwishart.rvs(df=nu0_blk, scale=Psi0_blk, random_state=rng)
        mu = rng.multivariate_normal(m0_blk, Sigma / kappa0_blk)
        return mu, Sigma
    ybar = Y.mean(axis=0)
    diff_y = Y - ybar
    S_ = diff_y.T @ diff_y
    kappa_n = kappa0_blk + nj
    nu_n = nu0_blk + nj
    m_n = (kappa0_blk * m0_blk + nj * ybar) / kappa_n
    d0 = (ybar - m0_blk).reshape(-1, 1)
    Psi_n = Psi0_blk + S_ + (kappa0_blk * nj / kappa_n) * (d0 @ d0.T)
    Sigma = invwishart.rvs(df=nu_n, scale=Psi_n, random_state=rng)
    mu = rng.multivariate_normal(m_n, Sigma / kappa_n)
    return mu, Sigma

# sanity check: draw from a tiny synthetic cluster of 6 patients, 3-dim block
demo_rng = np.random.default_rng(1)
Y_demo = demo_rng.normal(loc=[0, 0.5, 1.0], size=(6, 3))
mu_demo, Sigma_demo = sample_niw_posterior(Y_demo, demo_rng)
print("NIW block posterior draw, n=6:\n  mu =", mu_demo.round(2), "\n  Sigma =\n", Sigma_demo.round(2))

# %% [markdown]
# ## Categorical kernel
#
# The 3 patient-level constants keep independent Beta-Bernoulli updates per cluster — the working form
# of the Hamming-distance kernel the brief suggests. The 7 exercise-dependent flags are not
# collapsed to a single value. Instead, each flag's full
# (REST, LEG, PEAK) triplet is one of 8 joint categories, fit with a Dirichlet-Multinomial - the
# direct categorical counterpart of the continuous block's $3\times3$ NIW treatment.

# %%
n_cat = 8
alpha0_dir = np.ones(n_cat)

def sample_dirichlet_posterior(cat_col, rng):
    """Dirichlet posterior over the 8 joint REST/LEG/PEAK categories for this cluster/flag."""
    counts = np.bincount(cat_col, minlength=n_cat) if len(cat_col) else np.zeros(n_cat)
    return rng.dirichlet(alpha0_dir + counts)

demo_cats = demo_rng.integers(0, 8, size=10)
theta_demo = sample_dirichlet_posterior(demo_cats, demo_rng)
print("Dirichlet-Multinomial posterior draw over the 8 REST/LEG/PEAK patterns:\n", theta_demo.round(3))

# %% [markdown]
# ## Final variable set & the Phase 2 modelling plan
#
# Four variable blocks survive Phase 1, each with a different likelihood/update rule below.

# %%
final_vars = pd.DataFrame([
    ["Continuous, step-varying (NIW 3x3)", "PAWP.MEDIA, PAWP.V.AMPL, PVR.MEDIA, RAP.VY, CI", 15],
    ["Continuous, slope-type (NIG, 1-D)", "PAWP.MEDIA.CO.f.slp, PAPM.CO.f.slp, COVO2.slp, RAP.MEDIA.CO.f.slp", 4],
    ["Categorical, exercise-dependent (Dirichlet-Mult.)", "PH, HFpEF, stiff.LA, high.PVR, RHF, TR, low.CO", 7],
    ["Categorical, patient-constant (Beta-Bernoulli)", "SESSO, RITMO, BA", 3],
], columns=["Block", "Variables", "Dims"])
final_vars

# %% [markdown]
# **The Phase 2 modelling plan:**
#
# **1. Specify the model.** One shared Dirichlet Process partition (stick-breaking, truncated at
#    $H=15$ components) assigns every one of the 80 patients to exactly one latent cluster; that same
#    label drives two block likelihoods jointly, not two clusterings glued together afterwards:
#    - *Continuous (19 dims).* 5 step-varying variables (15 dims) each get a $3\times3$
#      Normal-Inverse-Wishart (NIW) prior per cluster; 4 slope-type variables (4 dims) each get a
#      1-D Normal-Inverse-Gamma (NIG) prior per cluster — the block-diagonal covariance from the
#      feasibility argument above, never the full $19\times19$ matrix.
#    - *Categorical (10 dims).* 7 exercise-dependent flags each get a Dirichlet-Multinomial prior
#      per cluster over their 8 joint REST/LEG/PEAK categories; 3 patient-constant variables
#      (`SESSO`, `RITMO`, `BA`) each get a Beta-Bernoulli prior per cluster.
# **2. Initialize.** Fix each block's prior hyperparameters and $H=15$; give every patient a starting
#    label, either at random or warm-started from Phase 0's $k$-means partition.
# **3. Run the Gibbs sampler.** The joint posterior over labels and cluster parameters has no closed
#    form, so MCMC approximates it: alternate, for thousands of iterations, (a) resampling every
#    patient's label, proportional to the combined likelihood of their data across both blocks under
#    each cluster's current parameters, and (b) resampling each cluster's parameters from their
#    conjugate posterior given only the patients currently assigned there.
# **4. Check convergence.** Inspect trace plots and run several chains from different starting
#    labels; only once they agree and look stationary are the draws trustworthy as posterior samples
#    — discard the early "burn-in" iterations before that point.
# **5. Resolve label-switching.** Cluster labels are arbitrary across iterations (cluster "3" in one
#    draw needn't be the same group as cluster "3" in another), so the kept draws can't just be
#    averaged directly. Build a patient-by-patient co-clustering matrix (how often two patients
#    landed in the same cluster) and extract one representative partition from it.
# **6. Report.** Summarize each final cluster's block parameters — means, covariances, category
#    probabilities — as posterior point estimates with credible intervals: the uncertainty-aware
#    counterpart of Phase 0's single hard $k$-means partition.

# %% [markdown]
# ## Running sensitivity-analysis log

# %%
sensitivity_log = pd.DataFrame([
    ["Which STEP for the 7 binary flags", "Vary by STEP for most patients (HFpEF 40%->89%)",
     "Resolved via joint Dirichlet-Multinomial, not an assumption"],
    ["Domain-blind screen of all 124 columns", f"{len(survivors_naive)} survivors -> d={d_naive_expanded}, exceeds even an even cluster split",
     "Confirms clinically-reasoned 21 as the right start"],
    ["Clinical vs. blind tie-break on the paper's 8", f"{len(unchanged)}/8 unchanged either way; {8-len(unchanged)}/8 need clinical judgement",
     "Quantifies how much of their 8 is 'statistics' vs. 'clinical'"],
    ["9 vs. published 8 continuous variables", "Their own stated rule yields 9, not 8, undocumented why",
     "Use the 9 the method actually produces"],
    ["Full vs. diagonal vs. block-diagonal covariance", "Full is singular at realistic cluster sizes; diagonal discards real correlation",
     "Block-diagonal is the only feasible, information-preserving option"],
    ["PEAK-only vs. joint REST/LEG/PEAK for the 7 flags", "PEAK-only needed an unstated assumption",
     "Dirichlet-Multinomial lets the data decide"],
], columns=["Decision", "Finding", "Implication"])
sensitivity_log
