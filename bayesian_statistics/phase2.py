# ---
# title: "HFpEF Bayesian Clustering — Phase 2: Two-Block DP Mixture, Block-Diagonal Covariance"
# author: "Mitch Wishart"
# format:
#   html:
#     toc: true
#     code-fold: show
# ---

# %% [markdown]
# # Phase 2 — Executing the Modelling Strategy
#
# Executes Phase 1 §9's strategy, revised: **9 continuous variables** (the
# mechanically-derived set, not the paper's published 8 — Phase 1's
# sensitivity note explains why), **block-diagonal covariance** for the
# continuous kernel (not pure diagonal — see "The model" below), and a
# **10-variable categorical/binary block** (the original 7 PEAK-step flags
# plus `SESSO`, `RITMO`, `BA`, recoded to 0/1).
#
# One latent cluster label per patient, two blocks, one shared posterior —
# a patient's clinical stage is one thing, expressed through continuous
# hemodynamics and binary/categorical phenotype at once.

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import invwishart, multivariate_normal

sns.set_theme(style="whitegrid")
RNG = np.random.default_rng(12345)

DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    r"C:\Users\mitch\OneDrive\1   B.ENGG\0   POLIMI\Bayesian Statistics\Project\r_code_datasets_wd",
))

# %% [markdown]
# ## Load data and build both blocks
#
# Continuous: 9 variables — 5 step-varying (`PAWP.MEDIA`, `PAWP.V.AMPL`,
# `PVR.MEDIA`, `RAP.VY`, `CI`) get REST/LEG/PEAK columns; 4 slope-type
# (`PAWP.MEDIA.CO.f.slp`, `PAPM.CO.f.slp`, `COVO2.slp`,
# `RAP.MEDIA.CO.f.slp`) get one column each. 19 dims total.
#
# Categorical/binary: 7 flags at PEAK step (Phase 1 §0 — these are
# exercise-dependent diagnostic classifications, not patient-level
# constants) plus `SESSO`/`RITMO`/`BA`, which genuinely are patient-level
# constants (confirmed in Phase 1, 0/89 patients vary across STEP), recoded
# to 0/1.

# %%
list1 = pd.read_table(DATA_DIR / "dataset1.txt")
list2 = pd.read_table(DATA_DIR / "dataset2.txt")
list3 = pd.read_table(DATA_DIR / "dataset3.txt")

dataset = pd.DataFrame({
    "ID": list1["ID"].astype(int),
    "STEP": list1["STEP"].astype("category"),
    "SESSO": list1["SESSO"],
    "RITMO": list1["RITMO"],
    "PAWP.MEDIA": list1["PAWP.MEDIA"].astype(float),
    "PAWP.V.AMPL": list2["PAWP.V.AMPL"].astype(float),
    "PAWP.MEDIA.CO.f.slp": list2["PAWP.MEDIA.CO.f.slp"].astype(float),
    "PVR.MEDIA": list2["PVR.MEDIA"].astype(float),
    "RAP.VY": list2["RAP.VY"].astype(float),
    "CI": list2["CI"].astype(float),
    "COVO2.slp": list2["COVO2.slp"].astype(float),
    "PAPM.CO.f.slp": list2["PAPM.CO.f.slp"].astype(float),
    "BA": list2["BA"],
    "RAP.MEDIA.CO.f.slp": list3["RAP.MEDIA.CO.f.slp"].astype(float),
    "PH": list3["PH"].astype(float),
    "HFpEF": list3["HFpEF"].astype(float),
    "stiff_LA": list3["stiff.LA"].astype(float),
    "high_PVR": list3["high.PVR"].astype(float),
    "RHF": list3["RHF"].astype(float),
    "TR": list3["TR"].astype(float),
    "low_CO": list3["low.CO"].astype(float),
})

peak_covo2 = dataset.loc[dataset["STEP"] == "PEAK"].set_index("ID")["COVO2.slp"]
dataset["COVO2.slp"] = dataset["ID"].map(peak_covo2)

# Recode the newly-added categorical variables to 0/1.
dataset["SESSO"] = (dataset["SESSO"] == "M").astype(float)      # 1 = male
dataset["RITMO"] = (dataset["RITMO"] == "FA").astype(float)     # 1 = atrial fibrillation
dataset["BA"] = (dataset["BA"] == "y").astype(float)

step_varying = ["PAWP.MEDIA", "PAWP.V.AMPL", "PVR.MEDIA", "RAP.VY", "CI"]
slope_vars = ["PAWP.MEDIA.CO.f.slp", "PAPM.CO.f.slp", "COVO2.slp", "RAP.MEDIA.CO.f.slp"]
peak_binary_vars = ["PH", "HFpEF", "stiff_LA", "high_PVR", "RHF", "TR", "low_CO"]
constant_binary_vars = ["SESSO", "RITMO", "BA"]
binary_vars = peak_binary_vars + constant_binary_vars

rest = dataset.loc[dataset["STEP"] == "REST"].drop(columns="STEP").add_suffix("_REST")
leg = dataset.loc[dataset["STEP"] == "LEG"].drop(columns="STEP").add_suffix("_LEG")
peak = dataset.loc[dataset["STEP"] == "PEAK"].drop(columns="STEP").add_suffix("_PEAK")
for df in (rest, leg, peak):
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)

patients = rest.merge(leg, on="ID").merge(peak, on="ID")
for v in slope_vars + constant_binary_vars:
    patients[v] = patients[f"{v}_PEAK"]  # slope: genuinely single-valued; constants: any step is identical
for v in peak_binary_vars:
    patients[v] = patients[f"{v}_PEAK"]  # exercise-dependent flags: PEAK specifically (Phase 1 §0)
patients.drop(columns=[c for c in patients.columns
                        if any(c.startswith(f"{v}_") for v in slope_vars + binary_vars)],
              inplace=True)

cluster_cols = [f"{v}_{s}" for v in step_varying for s in ("REST", "LEG", "PEAK")] + slope_vars
patients_full = patients[["ID"] + cluster_cols + binary_vars].dropna().reset_index(drop=True)
patient_ids = patients_full["ID"].to_numpy()

n = len(patients_full)
d_cont = len(cluster_cols)
d_bin = len(binary_vars)
print(f"n = {n} patients, continuous block d = {d_cont} dims, binary/categorical block d = {d_bin} dims")
print(f"Continuous columns: {cluster_cols}")
print(f"Binary/categorical columns: {binary_vars}")

X_raw = patients_full[cluster_cols].to_numpy()
col_mean, col_sd = X_raw.mean(axis=0), X_raw.std(axis=0)
X = (X_raw - col_mean) / col_sd
B = patients_full[binary_vars].to_numpy()

# Index bookkeeping: which columns of X belong to which block.
block_idx = {v: [cluster_cols.index(f"{v}_{s}") for s in ("REST", "LEG", "PEAK")] for v in step_varying}
slope_idx = [cluster_cols.index(v) for v in slope_vars]

# %% [markdown]
# ## The model
#
# $$
# r_i \mid \mathbf{w} \sim \text{Categorical}(w_1,\dots,w_H), \qquad
# w_h = v_h \prod_{\ell<h}(1-v_\ell), \qquad v_h \sim \text{Beta}(1, M)
# $$
#
# **Continuous likelihood — block-diagonal, not pure diagonal:**
#
# $$
# y_i \mid r_i=h \;\sim\; \Big[\prod_{v \in \text{step-varying}} \mathcal{N}_3(\mathbf{y}_{i,v}; \boldsymbol\mu_{h,v}, \boldsymbol\Sigma_{h,v})\Big]
# \times \Big[\prod_{j \in \text{slope}} \mathcal{N}(y_{ij}; \mu_{hj}, \sigma_{hj}^2)\Big]
# $$
#
# Each step-varying variable gets its own genuine 3-dimensional Gaussian
# (real 3x3 covariance $\boldsymbol\Sigma_{h,v}$, correlating that
# variable's own REST/LEG/PEAK readings) via a Normal-Inverse-Wishart (NIW)
# conjugate update — the matrix generalization of the Normal-Inverse-Gamma
# update already used for the slope variables. Different variables' blocks
# stay independent of each other (full 19x19 is infeasible — Phase 1 §6:
# determinant ~1e-224 at realistic cluster sizes); each 3x3 block only
# needs n_cluster>=4 to be well-posed, which every cluster we've produced
# clears easily.
#
# **Categorical/binary likelihood — independent Bernoulli per dimension,
# Beta-Bernoulli conjugate, across all 10 variables.** This is the brief's
# suggested Hamming-distance kernel in its working form (Phase 1 §9).

# %%
H = 15

# NIW hyperparameters for each 3-dim block (shared prior across h and v).
p_blk = 3
m0_blk = np.zeros(p_blk)
kappa0_blk = 1.0
nu0_blk = p_blk + 2                      # = 5, smallest integer giving a proper, weakly informative prior
Psi0_blk = np.eye(p_blk) * (nu0_blk - p_blk - 1)   # = I, so E[Sigma] = I (matches standardized scale)

# NIG hyperparameters for the 4 slope (1-D) dimensions.
m0, kappa0, a0, b0 = 0.0, 1.0, 2.0, 1.0

# Beta-Bernoulli hyperparameters for the 10 categorical dimensions.
a_bin, b_bin = 1.0, 1.0
EPS = 1e-9

ga_a, ga_b = 1.0, 1.0   # Gamma hyperprior on M


def sample_niw_posterior(Y):
    """NIW posterior draw for a 3-dim block, given Y: (n_h, 3)."""
    nj = len(Y)
    if nj == 0:
        Sigma = invwishart.rvs(df=nu0_blk, scale=Psi0_blk, random_state=RNG)
        mu = RNG.multivariate_normal(m0_blk, Sigma / kappa0_blk)
        return mu, Sigma
    ybar = Y.mean(axis=0)
    diff_y = Y - ybar
    S = diff_y.T @ diff_y
    kappa_n = kappa0_blk + nj
    nu_n = nu0_blk + nj
    m_n = (kappa0_blk * m0_blk + nj * ybar) / kappa_n
    d0 = (ybar - m0_blk).reshape(-1, 1)
    Psi_n = Psi0_blk + S + (kappa0_blk * nj / kappa_n) * (d0 @ d0.T)
    Sigma = invwishart.rvs(df=nu_n, scale=Psi_n, random_state=RNG)
    mu = RNG.multivariate_normal(m_n, Sigma / kappa_n)
    return mu, Sigma


def sample_nig_posterior(y_col):
    """NIG posterior draw for one 1-D slope dimension."""
    nj = len(y_col)
    if nj == 0:
        sigma2 = stats.invgamma.rvs(a0, scale=b0, random_state=RNG)
        mu = RNG.normal(m0, np.sqrt(sigma2 / kappa0))
        return mu, sigma2
    ybar = y_col.mean()
    ss = ((y_col - ybar) ** 2).sum()
    kappa_n = kappa0 + nj
    a_n = a0 + nj / 2
    b_n = b0 + 0.5 * ss + (kappa0 * nj * ybar ** 2) / (2 * kappa_n)
    m_n = (kappa0 * m0 + nj * ybar) / kappa_n
    sigma2 = stats.invgamma.rvs(a_n, scale=b_n, random_state=RNG)
    mu = RNG.normal(m_n, np.sqrt(sigma2 / kappa_n))
    return mu, sigma2


def sample_beta_bernoulli_posterior(b_col):
    nj = len(b_col)
    n1 = b_col.sum() if nj else 0
    return RNG.beta(a_bin + n1, b_bin + (nj - n1))


# %% [markdown]
# ## Joint Gibbs sampler

# %%
n_iter, burn_in = 2000, 700

r = RNG.integers(0, 5, size=n)
M = 1.0
mu_blk = {v: np.zeros((H, 3)) for v in step_varying}
Sigma_blk = {v: np.tile(np.eye(3), (H, 1, 1)) for v in step_varying}
mu_slope = np.zeros((H, len(slope_vars)))
sigma2_slope = np.ones((H, len(slope_vars)))
p_bin = np.full((H, d_bin), 0.5)

n_occupied_trace = np.zeros(n_iter, dtype=int)
r_samples = np.zeros((n_iter - burn_in, n), dtype=int)

for it in range(n_iter):
    members_mask = [r == h for h in range(H)]

    # (a) continuous block parameters
    for h in range(H):
        X_h = X[members_mask[h]]
        for v in step_varying:
            mu_blk[v][h], Sigma_blk[v][h] = sample_niw_posterior(X_h[:, block_idx[v]])
        for j, sidx in enumerate(slope_idx):
            mu_slope[h, j], sigma2_slope[h, j] = sample_nig_posterior(X_h[:, sidx])

    # (b) categorical block parameters
    for h in range(H):
        B_h = B[members_mask[h]]
        for l in range(d_bin):
            p_bin[h, l] = sample_beta_bernoulli_posterior(B_h[:, l])

    # (c) stick-breaking weights
    counts = np.bincount(r, minlength=H)
    Bc = np.array([counts[h + 1:].sum() for h in range(H)])
    v_stick = np.ones(H)
    v_stick[:-1] = RNG.beta(1 + counts[:-1], M + Bc[:-1])
    w = v_stick * np.concatenate(([1.0], np.cumprod(1 - v_stick[:-1])))
    w = np.clip(w, 1e-300, None)
    w /= w.sum()
    log_w = np.log(w)

    # (d) joint reassignment: block-diagonal continuous log-lik + slope log-lik + binary log-lik
    log_lik = np.tile(log_w, (n, 1))
    for v in step_varying:
        Y_v = X[:, block_idx[v]]
        for h in range(H):
            log_lik[:, h] += multivariate_normal.logpdf(Y_v, mean=mu_blk[v][h], cov=Sigma_blk[v][h])
    for j, sidx in enumerate(slope_idx):
        diff2 = (X[:, sidx:sidx + 1] - mu_slope[:, j][None, :]) ** 2
        log_lik += -0.5 * (np.log(2 * np.pi * sigma2_slope[:, j][None, :]) + diff2 / sigma2_slope[:, j][None, :])
    p_safe = np.clip(p_bin, EPS, 1 - EPS)
    log_lik += (B[:, None, :] * np.log(p_safe)[None, :, :]
                + (1 - B[:, None, :]) * np.log(1 - p_safe)[None, :, :]).sum(axis=2)

    log_lik -= log_lik.max(axis=1, keepdims=True)
    probs = np.exp(log_lik)
    probs /= probs.sum(axis=1, keepdims=True)
    r = np.array([RNG.choice(H, p=probs[i]) for i in range(n)])

    # (e) update M (Escobar & West 1995)
    k = len(np.unique(r))
    phi = RNG.beta(M + 1, n)
    rate = ga_b - np.log(phi)
    pi_mix = (ga_a + k - 1) / (n * rate)
    pi_mix = pi_mix / (1 + pi_mix)
    M = RNG.gamma(ga_a + k, 1 / rate) if RNG.random() < pi_mix else RNG.gamma(ga_a + k - 1, 1 / rate)

    n_occupied_trace[it] = k
    if it >= burn_in:
        r_samples[it - burn_in] = r

print(f"Ran {n_iter} sweeps ({burn_in} burn-in). "
      f"Post-burn-in occupied-cluster range: {n_occupied_trace[burn_in:].min()}"
      f"-{n_occupied_trace[burn_in:].max()}")

# %% [markdown]
# ## Convergence check

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(n_occupied_trace, lw=0.7)
axes[0].axvline(burn_in, color="red", ls="--", label="burn-in cutoff")
axes[0].set_xlabel("Sweep")
axes[0].set_ylabel("# occupied clusters")
axes[0].set_title("Trace: number of occupied clusters")
axes[0].legend()

post_k = n_occupied_trace[burn_in:]
axes[1].hist(post_k, bins=np.arange(post_k.min(), post_k.max() + 2) - 0.5,
              color="steelblue", edgecolor="white")
axes[1].axvline(5, color="red", ls="--", label="paper's k=5")
axes[1].set_xlabel("# occupied clusters")
axes[1].set_title("Posterior distribution over k (post burn-in)")
axes[1].legend()
plt.tight_layout()
plt.show()

print("Posterior P(k) (post burn-in):")
print(pd.Series(post_k).value_counts(normalize=True).sort_index().round(3))

# %% [markdown]
# ## Point-estimate partition — Dahl's (2005) least-squares method

# %%
co_cluster = np.zeros((n, n))
for s in range(r_samples.shape[0]):
    co_cluster += (r_samples[s][:, None] == r_samples[s][None, :])
co_cluster /= r_samples.shape[0]

best_sse, best_idx = np.inf, 0
for s in range(r_samples.shape[0]):
    same = (r_samples[s][:, None] == r_samples[s][None, :]).astype(float)
    sse = ((same - co_cluster) ** 2).sum()
    if sse < best_sse:
        best_sse, best_idx = sse, s

reference_r = r_samples[best_idx]
_, reference_r = np.unique(reference_r, return_inverse=True)
k_final = reference_r.max() + 1
print(f"Dahl's point estimate: {k_final} clusters, sizes {np.bincount(reference_r)}")

fig, ax = plt.subplots(figsize=(7, 6))
order = np.argsort(reference_r)
sns.heatmap(co_cluster[np.ix_(order, order)], ax=ax, cmap="viridis", cbar_kws={"label": "P(same cluster)"})
ax.set_title("Posterior co-clustering matrix (patients ordered by final cluster)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Per-patient posterior cluster-membership probability

# %%
def match_to_reference(sweep_r, reference_r, k_final):
    mapping = {}
    for h in np.unique(sweep_r):
        overlap = [(reference_r[sweep_r == h] == kk).sum() for kk in range(k_final)]
        mapping[h] = int(np.argmax(overlap))
    return np.array([mapping[h] for h in sweep_r])

membership_counts = np.zeros((n, k_final))
for s in range(r_samples.shape[0]):
    relabeled = match_to_reference(r_samples[s], reference_r, k_final)
    membership_counts[np.arange(n), relabeled] += 1
membership_probs = membership_counts / r_samples.shape[0]

prob_df = pd.DataFrame(membership_probs, columns=[f"cluster_{kk}" for kk in range(k_final)])
prob_df.insert(0, "ID", patient_ids)
prob_df.insert(1, "point_estimate_cluster", reference_r)

margin = (membership_probs.max(axis=1) - np.sort(membership_probs, axis=1)[:, -2]
          if k_final > 1 else membership_probs.max(axis=1))
confident_idx, ambiguous_idx = margin.argmax(), margin.argmin()
print("Most confident patient (example):")
print(prob_df.iloc[[confident_idx]].round(3))
print("\nMost ambiguous patient (example):")
print(prob_df.iloc[[ambiguous_idx]].round(3))

# %% [markdown]
# ## Cluster profiles — both blocks

# %%
profile_df = patients_full.copy()
profile_df["cluster"] = reference_r
order_by_severity = profile_df.groupby("cluster")["PAWP.MEDIA_PEAK"].mean().sort_values().index
severity_map = {old: new for new, old in enumerate(order_by_severity, start=1)}
profile_df["cluster"] = profile_df["cluster"].map(severity_map)

print("Cluster sizes:", dict(profile_df["cluster"].value_counts().sort_index()))
print("Phase 0 (k-means, k=5 imposed) sizes: 18/13/29/8/12")
print("Paper's published sizes: 26/30/12/4/8")

cont_profile_cols = [c for c in cluster_cols if c.endswith("_PEAK")] + slope_vars
print("\nContinuous block means by cluster (PEAK / slope values):")
print(profile_df.groupby("cluster")[cont_profile_cols].mean().round(2))

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=profile_df, x="cluster", y="PAWP.MEDIA_PEAK", ax=ax)
ax.set_title("Peak-exercise mean PAWP by cluster")
plt.tight_layout()
plt.show()

print("\nCategorical/binary block prevalence by cluster:")
print(profile_df.groupby("cluster")[binary_vars].mean().round(3))

fig, ax = plt.subplots(figsize=(10, 4))
profile_df.groupby("cluster")[binary_vars].mean().T.plot(kind="bar", ax=ax)
ax.set_ylabel("Prevalence")
ax.set_title("Binary/categorical phenotype prevalence by cluster")
ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - Block-diagonal covariance replaces pure diagonal: each step-varying
#   variable's REST/LEG/PEAK triplet now has a genuine 3x3 covariance,
#   correlating with itself (a real multivariate Gaussian), while staying
#   independent of other variables' blocks (full 19x19 remains infeasible).
# - Continuous block uses the mechanically-derived 9 variables (Phase 1's
#   revised decision), not the paper's published 8.
# - Categorical/binary block expanded to 10 variables, adding `SESSO`,
#   `RITMO`, `BA` (confirmed patient-level constants, unlike the original
#   7 PEAK-step-dependent flags).
# - Compare the posterior P(k) here against the earlier pure-diagonal,
#   8-variable, 7-flag version: if block-diagonal covariance is doing what
#   it's meant to, the posterior should concentrate closer to the paper's
#   k=5 than the pure-diagonal model's 8-9.
# - Still outstanding: the PAWP.A/RAP.V.AMPL sensitivity re-run, and a
#   predictive example for a new, out-of-sample patient (required-analysis
#   item 8) — neither built yet.
