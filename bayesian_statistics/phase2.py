# ---
# title: "HFpEF Bayesian Clustering — Phase 2: Two-Block Dirichlet Process Mixture Model"
# author: "Mitch Wishart"
# format:
#   html:
#     toc: true
#     code-fold: show
# ---

# %% [markdown]
# # Phase 2 — Executing the Modelling Strategy
#
# This notebook executes the strategy defined in Phase 1 §9, step by step.
# Every decision below was already justified in Phase 1 with numbers, not
# intuition — this notebook's job is to build the model, not re-argue for
# it; each step below references back to where its justification lives.
#
# **One latent cluster label per patient, two blocks, one shared posterior:**
# a patient's clinical stage is one thing, expressed through both their
# continuous hemodynamics and their binary phenotype at once — not two
# separately-fit models compared afterwards.

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
# ## Load data and build both blocks
#
# Continuous block: the paper's 8 variables (Phase 1 §5, §8), standardized.
# Binary block: the paper's 7 flags, taken at **PEAK step specifically**
# — Phase 1 §0 found these are exercise-dependent diagnostic classifications,
# not fixed patient labels (e.g. HFpEF prevalence goes from ~40% at REST to
# ~89% at PEAK), so using any other row, or an assumed patient-level
# constant, would silently change the categorical block's whole model.

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

step_varying = ["PAWP.MEDIA", "PAWP.V.AMPL", "PVR.MEDIA", "RAP.VY", "CI"]
slope_vars = ["PAWP.MEDIA.CO.f.slp", "PAPM.CO.f.slp", "COVO2.slp"]
binary_vars = ["PH", "HFpEF", "stiff_LA", "high_PVR", "RHF", "TR", "low_CO"]

rest = dataset.loc[dataset["STEP"] == "REST"].drop(columns="STEP").add_suffix("_REST")
leg = dataset.loc[dataset["STEP"] == "LEG"].drop(columns="STEP").add_suffix("_LEG")
peak = dataset.loc[dataset["STEP"] == "PEAK"].drop(columns="STEP").add_suffix("_PEAK")
for df in (rest, leg, peak):
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)

patients = rest.merge(leg, on="ID").merge(peak, on="ID")
for v in slope_vars:
    patients[v] = patients[f"{v}_PEAK"]
    patients.drop(columns=[f"{v}_REST", f"{v}_LEG", f"{v}_PEAK"], inplace=True)
for v in binary_vars:
    patients[v] = patients[f"{v}_PEAK"]
    patients.drop(columns=[f"{v}_REST", f"{v}_LEG", f"{v}_PEAK"], inplace=True)

cluster_cols = [f"{v}_{s}" for v in step_varying for s in ("REST", "LEG", "PEAK")] + slope_vars
patients_full = patients[["ID"] + cluster_cols + binary_vars].dropna().reset_index(drop=True)
patient_ids = patients_full["ID"].to_numpy()

n = len(patients_full)
d_cont = len(cluster_cols)
d_bin = len(binary_vars)
print(f"n = {n} patients, continuous block d = {d_cont} dims, binary block d = {d_bin} dims")
print(f"Continuous columns: {cluster_cols}")
print(f"Binary columns (PEAK step): {binary_vars}")

X_raw = patients_full[cluster_cols].to_numpy()
col_mean, col_sd = X_raw.mean(axis=0), X_raw.std(axis=0)
X = (X_raw - col_mean) / col_sd                      # Step 1: standardize continuous block
B = patients_full[binary_vars].to_numpy()            # binary block, already 0/1

# %% [markdown]
# ## The model
#
# Shared cluster label, two conditionally-independent blocks given that
# label:
#
# $$
# y_i \mid r_i = h \;\sim\; \prod_{j=1}^{d_{\text{cont}}} \mathcal{N}(\mu_{hj}, \sigma_{hj}^2),
# \qquad
# b_i \mid r_i = h \;\sim\; \prod_{\ell=1}^{d_{\text{bin}}} \text{Bernoulli}(p_{h\ell})
# $$
#
# $$
# r_i \mid \mathbf{w} \sim \text{Categorical}(w_1,\dots,w_H), \qquad
# w_h = v_h \prod_{\ell<h}(1-v_\ell), \qquad v_h \sim \text{Beta}(1, M)
# $$
#
# **Continuous kernel — diagonal Gaussian / NIG per dimension.** Phase 1 §6
# showed this isn't the easier option among several, it's close to the only
# numerically well-posed one at these cluster sizes: a full covariance
# matrix needs n_cluster >= d+1 just to be invertible, and every cluster
# Phase 0 actually produced fails that test at d=18, let alone the
# unfiltered d=32+.
#
# **Binary kernel — independent Bernoulli per dimension, Beta-Bernoulli
# conjugate.** This is the brief's suggested Hamming-distance kernel in
# its working form: a patient's 7-flag vector is most probable under
# whichever cluster's characteristic pattern $(p_{h,1},\dots,p_{h,7})$ it's
# closest to in Hamming distance, because the Bernoulli log-likelihood
# $\sum_\ell [b_{i\ell}\log p_{h\ell} + (1-b_{i\ell})\log(1-p_{h\ell})]$ is
# maximized exactly when $b_i$ matches $p_h$'s rounded pattern — the
# independence-per-dimension form isn't a simplification bolted on
# separately from the Hamming-kernel idea, it's the same idea written as a
# likelihood instead of a distance.
#
# **Why both blocks share one $r_i$, not two separate partitions.** The
# brief's "clinical stage" is a property of the patient, not of a block —
# if the two blocks had separate cluster labels, "stage 3" would mean
# different things depending on which block you asked, which isn't a
# coherent clinical staging system. The joint likelihood below enforces one
# label driving both.

# %%
H = 15  # truncation level (Phase 1 §9 / Phase 0 prototype: well above paper's k=5)

m0 = np.zeros(d_cont)
kappa0 = 1.0
a0, b0 = 2.0, 1.0           # E[sigma^2] = b0/(a0-1) = 1, matching standardized scale
a_bin, b_bin = 1.0, 1.0     # Beta(1,1) = uniform prior on each p_h,l
ga_a, ga_b = 1.0, 1.0       # Gamma(a,b) hyperprior on M (shape-rate)


def sample_nig_posterior(y_col):
    """Draw (mu, sigma2) from the NIG posterior given a 1-D sample y_col."""
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


def sample_beta_bernoulli_posterior(b_col):
    """Draw p from the Beta-Bernoulli posterior given a 1-D 0/1 sample b_col."""
    nj = len(b_col)
    n1 = b_col.sum() if nj else 0
    n0 = nj - n1
    return RNG.beta(a_bin + n1, b_bin + n0)


# %% [markdown]
# ## Joint Gibbs sampler (Phase 1 §9, Step 5)

# %%
n_iter, burn_in = 3000, 1000
EPS = 1e-9  # clips p away from exactly 0/1 for safe log()

r = RNG.integers(0, 5, size=n)
M = 1.0
mu = np.zeros((H, d_cont))
sigma2 = np.ones((H, d_cont))
p_bin = np.full((H, d_bin), 0.5)

n_occupied_trace = np.zeros(n_iter, dtype=int)
M_trace = np.zeros(n_iter)
r_samples = np.zeros((n_iter - burn_in, n), dtype=int)

for it in range(n_iter):
    # (a) continuous cluster parameters
    for h in range(H):
        members = X[r == h]
        for j in range(d_cont):
            mu[h, j], sigma2[h, j] = sample_nig_posterior(members[:, j] if len(members) else np.array([]))

    # (b) binary cluster parameters
    for h in range(H):
        members_b = B[r == h]
        for l in range(d_bin):
            p_bin[h, l] = sample_beta_bernoulli_posterior(members_b[:, l] if len(members_b) else np.array([]))

    # (c) stick-breaking weights
    counts = np.bincount(r, minlength=H)
    A = counts
    Bc = np.array([counts[h + 1:].sum() for h in range(H)])
    v = np.ones(H)
    v[:-1] = RNG.beta(1 + A[:-1], M + Bc[:-1])
    w = v * np.concatenate(([1.0], np.cumprod(1 - v[:-1])))
    w = np.clip(w, 1e-300, None)
    w /= w.sum()

    # (d) joint reassignment: continuous log-lik + binary log-lik
    log_w = np.log(w)
    diff2 = (X[:, None, :] - mu[None, :, :]) ** 2
    log_lik_cont = -0.5 * (np.log(2 * np.pi * sigma2[None, :, :]) + diff2 / sigma2[None, :, :])

    p_safe = np.clip(p_bin, EPS, 1 - EPS)
    log_lik_bin = (B[:, None, :] * np.log(p_safe)[None, :, :]
                   + (1 - B[:, None, :]) * np.log(1 - p_safe)[None, :, :])

    log_post = log_w[None, :] + log_lik_cont.sum(axis=2) + log_lik_bin.sum(axis=2)
    log_post -= log_post.max(axis=1, keepdims=True)
    probs = np.exp(log_post)
    probs /= probs.sum(axis=1, keepdims=True)
    r = np.array([RNG.choice(H, p=probs[i]) for i in range(n)])

    # (e) update M (Escobar & West 1995 auxiliary variable scheme)
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
# ## Cluster profiles — both blocks (required-analysis items 4 and 5)

# %%
profile_df = patients_full.copy()
profile_df["cluster"] = reference_r
order_by_severity = profile_df.groupby("cluster")["PAWP.MEDIA_PEAK"].mean().sort_values().index
severity_map = {old: new for new, old in enumerate(order_by_severity, start=1)}
profile_df["cluster"] = profile_df["cluster"].map(severity_map)

print("Cluster sizes:", dict(profile_df["cluster"].value_counts().sort_index()))
print("Phase 0 (k-means, k=5 imposed) sizes: 18/13/29/8/12")
print("Paper's published sizes: 26/30/12/4/8")

# Continuous block: mean of each variable (PEAK value where step-varying) per cluster
cont_profile_cols = [c for c in cluster_cols if c.endswith("_PEAK")] + slope_vars
print("\nContinuous block means by cluster (PEAK / slope values):")
print(profile_df.groupby("cluster")[cont_profile_cols].mean().round(2))

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=profile_df, x="cluster", y="PAWP.MEDIA_PEAK", ax=ax)
ax.set_title("Peak-exercise mean PAWP by cluster")
plt.tight_layout()
plt.show()

# Binary block: prevalence of each flag per cluster
print("\nBinary block prevalence by cluster:")
print(profile_df.groupby("cluster")[binary_vars].mean().round(3))

fig, ax = plt.subplots(figsize=(9, 4))
profile_df.groupby("cluster")[binary_vars].mean().T.plot(kind="bar", ax=ax)
ax.set_ylabel("Prevalence")
ax.set_title("Binary phenotype prevalence by cluster")
ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# - This is the two-block model the brief actually asks for — not the
#   continuous-only prototype from the Phase 0 pipeline, and not two
#   separately-fit models — one shared partition $r_i$, built and justified
#   step by step in Phase 1.
# - Every structural choice here (diagonal covariance, Beta-Bernoulli
#   binary kernel, truncated stick-breaking DP, which 8+7 variables, which
#   STEP to use for the binary flags) has its justification filed as a
#   Phase 1 sensitivity note, not asserted here without support.
# - Still outstanding, per Phase 1 §9 Step 7: the PAWP.A/RAP.V.AMPL
#   sensitivity re-run, and a predictive example assigning a new,
#   out-of-sample patient to these clusters — both required-analysis items,
#   not yet built.
