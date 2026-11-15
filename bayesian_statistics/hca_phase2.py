# ---
# title: "HFpEF Bayesian Clustering — Phase 2: Dirichlet Process Mixture Model"
# author: "Mitch Wishart"
# format:
#   html:
#     toc: true
#     code-fold: show
# ---

# %% [markdown]
# # Phase 2 — A Bayesian Nonparametric Clustering Model
#
# Phase 1 replicated the paper's k-means pipeline and flagged its limitations:
# k was imposed rather than inferred, cluster membership was a hard
# deterministic label, unscaled Euclidean distance let large-magnitude
# variables dominate, and nothing in the pipeline carried any notion of
# uncertainty. This notebook replaces that pipeline with a **Dirichlet
# process (DP) mixture model**, following the framework and posterior
# simulation methods in Müller, Quintana, Jara & Hanson, *Bayesian
# Nonparametric Data Analysis* (2015), Ch. 1–2.
#
# **Model structure, and why each piece was chosen:**
#
# - **Truncated stick-breaking DP** (Ishwaran & James 2001; book §2.4.6,
#   Algorithm 4), truncated at $H$ components well above the paper's $k=5$.
#   This is the practical, implementable version of an infinite DP mixture —
#   simpler to code correctly than a full Chinese-Restaurant-Process sampler,
#   with controllable approximation error as $H$ grows.
# - **Diagonal Gaussian kernel per cluster** (independent Normal-Inverse-Gamma
#   per dimension), rather than a full 18×18 covariance matrix. With only
#   ~80 patients split across up to $H$ clusters, many clusters will have
#   single-digit membership — far too little data to estimate a full
#   covariance matrix for 18 dimensions without it being degenerate. Treating
#   dimensions as conditionally independent within a cluster is the standard,
#   numerically stable fallback at this sample size; it's a deliberate
#   simplification, not an oversight, and is worth flagging in the report.
# - **Inputs standardized (z-scored) before clustering.** This directly
#   resolves the Phase 1 limitation where unscaled Euclidean distance let
#   a large-magnitude variable like PVR dominate: the model's own learned
#   per-dimension variance would eventually compensate for scale, but
#   standardizing first avoids forcing it to do that work with very little
#   data per cluster.
# - **Concentration parameter $M$ given a Gamma(1,1) hyperprior**, updated
#   each sweep via the Escobar & West (1995) auxiliary-variable scheme (book
#   §2.4.2, Algorithm 2) — so the model's appetite for opening new clusters
#   is learned from the data rather than hand-tuned.
# - **Dahl's (2005) least-squares method** for a single point-estimate
#   partition from the posterior, and a **posterior co-clustering matrix**
#   to sidestep label-switching when reporting results — label identity
#   ("cluster 3") is not meaningful across MCMC sweeps, but "these two
#   patients were grouped together" is.
# - **Per-patient cluster probabilities** are now a genuine posterior
#   quantity: after fixing a reference partition (via Dahl's method), each
#   post-burn-in sweep's clusters are matched back to the reference clusters
#   (by maximum overlap) and the empirical frequency of each patient landing
#   in each reference cluster, across sweeps, *is* the posterior membership
#   probability — not the ad hoc distance heuristic used for illustration in
#   Phase 1.

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
RNG = np.random.default_rng(12345)

DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    r"C:\Users\mitch\OneDrive\1   B.ENGG\0   POLIMI\Bayesian Statistics\Project\r_code_datasets_wd",
))

# %% [markdown]
# ## 1. Load data and build the same input matrix as Phase 1
#
# Same 8 clustering variables, same REST/LEG/PEAK + slope structure, same
# 80-patient sample — so Phase 2's results are directly comparable to
# Phase 1's, rather than being a different problem in disguise.

# %%
list1 = pd.read_table(DATA_DIR / "dataset1.txt")
list2 = pd.read_table(DATA_DIR / "dataset2.txt")
list3 = pd.read_table(DATA_DIR / "dataset3.txt")

dataset = pd.DataFrame({
    "ID": list1["ID"].astype(int),
    "STEP": list1["STEP"].astype("category"),
    "PAWP.MEDIA": list1["PAWP.MEDIA"].astype(float),
    "PAWP.V.AMPL": list2["PAWP.V.AMPL"].astype(float),
    "PAWP.MEDIA.CO.f.slp": list2["PAWP.MEDIA.CO.f.slp"].astype(float),
    "PVR.MEDIA": list2["PVR.MEDIA"].astype(float),
    "RAP.VY": list2["RAP.VY"].astype(float),
    "CI": list2["CI"].astype(float),
    "COVO2.slp": list2["COVO2.slp"].astype(float),
    "PAPM.CO.f.slp": list2["PAPM.CO.f.slp"].astype(float),
})

peak_covo2 = dataset.loc[dataset["STEP"] == "PEAK"].set_index("ID")["COVO2.slp"]
dataset["COVO2.slp"] = dataset["ID"].map(peak_covo2)

step_varying = ["PAWP.MEDIA", "PAWP.V.AMPL", "PVR.MEDIA", "RAP.VY", "CI"]
slope_vars = ["PAWP.MEDIA.CO.f.slp", "PAPM.CO.f.slp", "COVO2.slp"]

rest = dataset.loc[dataset["STEP"] == "REST"].drop(columns="STEP").add_suffix("_REST")
leg = dataset.loc[dataset["STEP"] == "LEG"].drop(columns="STEP").add_suffix("_LEG")
peak = dataset.loc[dataset["STEP"] == "PEAK"].drop(columns="STEP").add_suffix("_PEAK")
for df in (rest, leg, peak):
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)

patients = rest.merge(leg, on="ID").merge(peak, on="ID")
for v in slope_vars:
    patients[v] = patients[f"{v}_PEAK"]
    patients.drop(columns=[f"{v}_REST", f"{v}_LEG", f"{v}_PEAK"], inplace=True)

cluster_cols = [f"{v}_{s}" for v in step_varying for s in ("REST", "LEG", "PEAK")] + slope_vars
X_df = patients[["ID"] + cluster_cols].dropna().reset_index(drop=True)
patient_ids = X_df["ID"].to_numpy()
n, d = len(X_df), len(cluster_cols)
print(f"n = {n} patients, d = {d} dimensions: {cluster_cols}")

# Standardize — see the model-structure note above.
X_raw = X_df[cluster_cols].to_numpy()
col_mean, col_sd = X_raw.mean(axis=0), X_raw.std(axis=0)
X = (X_raw - col_mean) / col_sd

# %% [markdown]
# ## 2. The model
#
# $$
# y_i \mid r_i = h \;\sim\; \prod_{j=1}^{d} \mathcal{N}(\mu_{hj}, \sigma_{hj}^2),
# \qquad
# r_i \mid \mathbf{w} \sim \text{Categorical}(w_1,\dots,w_H)
# $$
#
# $$
# w_h = v_h \prod_{\ell<h}(1-v_\ell), \qquad v_h \sim \text{Beta}(1, M), \quad h<H, \quad v_H = 1
# $$
#
# with an independent Normal-Inverse-Gamma base measure on each dimension's
# $(\mu_{hj}, \sigma_{hj}^2)$, and $M \sim \text{Gamma}(1,1)$.
#
# Because the inputs are standardized, the prior is centered at the
# standardized scale's natural values: prior mean 0, prior $E[\sigma^2]=1$.

# %%
H = 15  # truncation level, comfortably above the paper's k=5

# NIG base measure hyperparameters (per dimension, shared across h)
m0 = np.zeros(d)
kappa0 = 1.0
a0 = 2.0
b0 = 1.0  # E[sigma^2] = b0/(a0-1) = 1, matching the standardized scale

ga_a, ga_b = 1.0, 1.0  # Gamma(a,b) hyperprior on M (shape-rate)


def sample_nig_posterior(y_col):
    """Draw (mu, sigma2) from the NIG posterior given a 1-D sample y_col.
    Empty cluster (y_col has length 0) draws straight from the prior."""
    nj = len(y_col)
    if nj == 0:
        sigma2 = stats.invgamma.rvs(a0, scale=b0, random_state=RNG)
        mu = RNG.normal(0.0, np.sqrt(sigma2 / kappa0))
        return mu, sigma2
    ybar = y_col.mean()
    ss = ((y_col - ybar) ** 2).sum()
    kappa_n = kappa0 + nj
    a_n = a0 + nj / 2
    b_n = b0 + 0.5 * ss + (kappa0 * nj * ybar ** 2) / (2 * kappa_n)
    m_n = (kappa0 * 0.0 + nj * ybar) / kappa_n
    sigma2 = stats.invgamma.rvs(a_n, scale=b_n, random_state=RNG)
    mu = RNG.normal(m_n, np.sqrt(sigma2 / kappa_n))
    return mu, sigma2


# %% [markdown]
# ## 3. Gibbs sampler (truncated DP, book Algorithm 4 + Algorithm 2's M-update)

# %%
n_iter, burn_in = 3000, 1000

r = RNG.integers(0, 5, size=n)  # arbitrary warm start
M = 1.0
mu = np.zeros((H, d))
sigma2 = np.ones((H, d))

n_occupied_trace = np.zeros(n_iter, dtype=int)
M_trace = np.zeros(n_iter)
r_samples = np.zeros((n_iter - burn_in, n), dtype=int)

for it in range(n_iter):
    # --- (a) cluster parameters: mu_h, sigma2_h for each of the H slots ---
    for h in range(H):
        members = X[r == h]
        for j in range(d):
            mu[h, j], sigma2[h, j] = sample_nig_posterior(members[:, j] if len(members) else np.array([]))

    # --- (b) stick-breaking weights ---
    counts = np.bincount(r, minlength=H)
    A = counts
    B = np.array([counts[h + 1:].sum() for h in range(H)])
    v = np.ones(H)
    v[:-1] = RNG.beta(1 + A[:-1], M + B[:-1])
    w = v * np.concatenate(([1.0], np.cumprod(1 - v[:-1])))
    w = np.clip(w, 1e-300, None)
    w /= w.sum()

    # --- (c) reassign each patient ---
    log_w = np.log(w)
    # log N(y | mu, sigma2) summed over dims, vectorized over (i, h)
    diff2 = (X[:, None, :] - mu[None, :, :]) ** 2
    log_lik = -0.5 * (np.log(2 * np.pi * sigma2[None, :, :]) + diff2 / sigma2[None, :, :])
    log_post = log_w[None, :] + log_lik.sum(axis=2)
    log_post -= log_post.max(axis=1, keepdims=True)
    probs = np.exp(log_post)
    probs /= probs.sum(axis=1, keepdims=True)
    r = np.array([RNG.choice(H, p=probs[i]) for i in range(n)])

    # --- (d) update M (Escobar & West 1995 auxiliary variable scheme) ---
    k = len(np.unique(r))
    phi = RNG.beta(M + 1, n)
    rate = ga_b - np.log(phi)
    pi_mix = (ga_a + k - 1) / (n * rate)
    pi_mix = pi_mix / (1 + pi_mix)
    if RNG.random() < pi_mix:
        M = RNG.gamma(ga_a + k, 1 / rate)
    else:
        M = RNG.gamma(ga_a + k - 1, 1 / rate)

    n_occupied_trace[it] = k
    M_trace[it] = M
    if it >= burn_in:
        r_samples[it - burn_in] = r

print(f"Ran {n_iter} sweeps ({burn_in} burn-in). "
      f"Post-burn-in occupied-cluster range: {n_occupied_trace[burn_in:].min()}"
      f"-{n_occupied_trace[burn_in:].max()}")

# %% [markdown]
# ## 4. Convergence check
#
# The number of occupied clusters per sweep is the simplest and most
# relevant diagnostic here — unlike k-means, this model doesn't fix k, so
# seeing this trace settle into a stable range (rather than drifting or
# oscillating wildly) is the basic check that the sampler is behaving.

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
# ## 5. Point-estimate partition — Dahl's (2005) least-squares method
#
# Cluster *labels* aren't identifiable across sweeps (label switching), but
# whether two patients are in the same cluster together *is* well-defined
# and comparable across sweeps. The posterior co-clustering matrix
# $\hat{C}_{ij}$ = fraction of sweeps where patients $i,j$ share a cluster
# is the label-invariant summary; Dahl's method then picks, among the
# stored sweeps, the single partition whose own co-clustering matrix is
# closest (in squared error) to $\hat{C}$ — i.e. the most "typical" sweep.

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
# Relabel densely (0, 1, 2, ...) in order of first appearance, for readability.
_, reference_r = np.unique(reference_r, return_inverse=True)
k_final = reference_r.max() + 1
print(f"Dahl's point estimate: {k_final} clusters, sizes {np.bincount(reference_r)}")

fig, ax = plt.subplots(figsize=(7, 6))
order = np.argsort(reference_r)
sns.heatmap(co_cluster[np.ix_(order, order)], ax=ax, cmap="viridis", cbar_kws={"label": "P(same cluster)"})
ax.set_title("Posterior co-clustering matrix (patients ordered by final cluster)")
ax.set_xlabel("Patient (reordered)")
ax.set_ylabel("Patient (reordered)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Per-patient cluster-membership probability
#
# Now a genuine posterior quantity, not a heuristic: match each post-burn-in
# sweep's clusters onto Dahl's reference clusters by maximum overlap, then
# the empirical frequency each patient lands in each reference cluster,
# across all sweeps, is their posterior membership probability.

# %%
def match_to_reference(sweep_r, reference_r, k_final):
    mapping = {}
    for h in np.unique(sweep_r):
        overlap = [(reference_r[sweep_r == h] == k).sum() for k in range(k_final)]
        mapping[h] = int(np.argmax(overlap))
    return np.array([mapping[h] for h in sweep_r])

membership_counts = np.zeros((n, k_final))
for s in range(r_samples.shape[0]):
    relabeled = match_to_reference(r_samples[s], reference_r, k_final)
    membership_counts[np.arange(n), relabeled] += 1
membership_probs = membership_counts / r_samples.shape[0]

prob_df = pd.DataFrame(membership_probs, columns=[f"cluster_{k}" for k in range(k_final)])
prob_df.insert(0, "ID", patient_ids)
prob_df.insert(1, "point_estimate_cluster", reference_r)

margin = membership_probs.max(axis=1) - np.sort(membership_probs, axis=1)[:, -2] if k_final > 1 else membership_probs.max(axis=1)
confident_idx, ambiguous_idx = margin.argmax(), margin.argmin()
print("Most confident patient (example):")
print(prob_df.iloc[[confident_idx]].round(3))
print("\nMost ambiguous patient (example):")
print(prob_df.iloc[[ambiguous_idx]].round(3))

# %% [markdown]
# ## 7. Cluster profiles and comparison with Phase 1 / the paper

# %%
profile_df = X_df.copy()
profile_df["cluster"] = reference_r
order_by_severity = profile_df.groupby("cluster")["PAWP.MEDIA_PEAK"].mean().sort_values().index
severity_map = {old: new for new, old in enumerate(order_by_severity, start=1)}
profile_df["cluster"] = profile_df["cluster"].map(severity_map)
reference_r_relabeled = profile_df["cluster"].to_numpy()

print("Phase 2 (DP mixture, Dahl point estimate) cluster sizes:",
      dict(profile_df["cluster"].value_counts().sort_index()))
print("Phase 1 (k-means, k=5 imposed) cluster sizes: 18/13/29/8/12")
print("Paper's published cluster sizes: 26/30/12/4/8")

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=profile_df, x="cluster", y="PAWP.MEDIA_PEAK", ax=ax)
ax.set_title("Peak-exercise mean PAWP by Phase 2 cluster")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - The model infers $k$ rather than fixing it — the posterior over k (§4)
#   is the direct answer to Phase 1's "k=5 is imposed, not inferred"
#   limitation, and can be compared directly against the paper's informal
#   elbow-plot choice.
# - Per-patient cluster probabilities (§6) are now a real posterior
#   quantity derived from repeated MCMC sweeps, not the inverse-distance
#   heuristic used for illustration in Phase 1.
# - The diagonal-covariance simplification (§ model structure) and the
#   truncation level $H=15$ are both deliberate, stated modeling choices —
#   worth a sentence each in the report, since they're the two places this
#   implementation trades some fidelity for tractability given the dataset's
#   size (n=80, d=18).
# - The posterior favors around 8–9 clusters, noticeably more than the
#   paper's k=5. The most likely driver is the diagonal-covariance choice
#   itself: REST/LEG/PEAK copies of the same variable are strongly
#   correlated in reality, but the model treats them as independent
#   dimensions, so noise in any one of them can split patients into extra
#   clusters that a shared-covariance model would have merged. Worth
#   reporting as an honest finding rather than tuning the model until it
#   lands on 5 — and worth testing directly if time allows, e.g. by
#   reducing to one step per variable, or revisiting full covariance.
# - Next: compare this model's point-estimate partition against the
#   paper's reported clusters profile-by-profile (Kruskal-Wallis per
#   variable, as the paper did), and consider whether the full-covariance
#   version is worth revisiting if time allows.
