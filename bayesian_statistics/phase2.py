# ---
# title: "Bayesian Clustering for Data-Driven Clinical Staging in HFpEF — Phase 2"
# format:
#   html:
#     toc: true
#     code-fold: true
#     code-tools: true
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
# # Phase 2 — Bayesian Nonparametric Clustering Model
#
# **Mitch Wishart**
#
# Standalone Phase 2: the model is specified, built, and run end-to-end on the real 80 patients,
# with a 4-chain convergence check (R-hat + Adjusted Rand Index between chains). Self-contained —
# doesn't depend on the Phase 0/Phase 1 notebook, just the three raw data files.

# %%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import invwishart, multivariate_normal
from sklearn.metrics import adjusted_rand_score

sns.set_theme(style="whitegrid")
RANDOM_STATE = 12345
DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    r"C:\Users\mitch\OneDrive\1   B.ENGG\0   POLIMI\Bayesian Statistics\Project\r_code_datasets_wd",
))

list1 = pd.read_table(DATA_DIR / "dataset1.txt")
list2 = pd.read_table(DATA_DIR / "dataset2.txt")
list3 = pd.read_table(DATA_DIR / "dataset3.txt")
assert len(list1) == len(list2) == len(list3)

# %% [markdown]
# ## The model
#
# One shared Dirichlet Process partition
# (stick-breaking, truncated at $H=15$) assigns every patient a single latent cluster label; that
# one label drives both blocks jointly, not two separate clusterings glued together afterwards.
#
# $$
# r_i \mid \mathbf{w} \sim \text{Categorical}(w_1,\dots,w_H), \qquad
# w_h = v_h \prod_{\ell<h}(1-v_\ell), \qquad v_h \sim \text{Beta}(1, M)
# $$
#
# Continuous block: each step-varying variable keeps its own $3\times3$ NIW-conjugate Gaussian over
# its REST/LEG/PEAK triplet; each slope variable keeps a 1-D NIG-conjugate Gaussian — block-diagonal,
# never the full $19\times19$. Categorical block: the 7 exercise-dependent flags keep their
# 8-category Dirichlet-Multinomial, fit over the full joint REST/LEG/PEAK pattern rather than one
# collapsed STEP; `SESSO`/`RITMO`/`BA` keep independent Beta-Bernoulli updates.

# %% [markdown]
# ## Build the two blocks over all 80 patients

# %%
dataset2 = pd.DataFrame({
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
peak_covo2_2 = dataset2.loc[dataset2["STEP"] == "PEAK"].set_index("ID")["COVO2.slp"]
dataset2["COVO2.slp"] = dataset2["ID"].map(peak_covo2_2)
dataset2["SESSO"] = (dataset2["SESSO"] == "M").astype(float)      # 1 = male
dataset2["RITMO"] = (dataset2["RITMO"] == "FA").astype(float)     # 1 = atrial fibrillation
dataset2["BA"] = (dataset2["BA"] == "y").astype(float)

step_varying = ["PAWP.MEDIA", "PAWP.V.AMPL", "PVR.MEDIA", "RAP.VY", "CI"]
slope_vars = ["PAWP.MEDIA.CO.f.slp", "PAPM.CO.f.slp", "COVO2.slp", "RAP.MEDIA.CO.f.slp"]
peak_binary_vars = ["PH", "HFpEF", "stiff_LA", "high_PVR", "RHF", "TR", "low_CO"]
constant_binary_vars = ["SESSO", "RITMO", "BA"]

rest2 = dataset2.loc[dataset2["STEP"] == "REST"].drop(columns="STEP").add_suffix("_REST")
leg2 = dataset2.loc[dataset2["STEP"] == "LEG"].drop(columns="STEP").add_suffix("_LEG")
peak2 = dataset2.loc[dataset2["STEP"] == "PEAK"].drop(columns="STEP").add_suffix("_PEAK")
for df in (rest2, leg2, peak2):
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)

patients2 = rest2.merge(leg2, on="ID").merge(peak2, on="ID")
for v in slope_vars:
    patients2[v] = patients2[f"{v}_PEAK"]  # single-valued; any STEP column is identical
# SESSO/RITMO are only recorded on the REST row of the raw files; BA is populated at every STEP.
patients2["SESSO"] = patients2["SESSO_REST"]
patients2["RITMO"] = patients2["RITMO_REST"]
patients2["BA"] = patients2["BA_PEAK"]

# Each of the 7 exercise-dependent flags: encode the full (REST, LEG, PEAK) triplet as one of
# 8 joint patterns (4*REST + 2*LEG + PEAK), fit below via Dirichlet-Multinomial - no "which STEP
# matters" assumption needed.
for v in peak_binary_vars:
    patients2[f"{v}_cat"] = 4 * patients2[f"{v}_REST"] + 2 * patients2[f"{v}_LEG"] + patients2[f"{v}_PEAK"]

cat_cols = [f"{v}_cat" for v in peak_binary_vars]
cluster_cols = [f"{v}_{s}" for v in step_varying for s in ("REST", "LEG", "PEAK")] + slope_vars
patients2_full = patients2[["ID"] + cluster_cols + cat_cols + constant_binary_vars].dropna().reset_index(drop=True)
patients2_full[cat_cols] = patients2_full[cat_cols].astype(int)
patient_ids = patients2_full["ID"].to_numpy()

# Decoded REST/LEG/PEAK 0/1 values, kept purely for readable profiling later.
for v in peak_binary_vars:
    patients2_full[f"{v}_REST_dec"] = (patients2_full[f"{v}_cat"] // 4) % 2
    patients2_full[f"{v}_LEG_dec"] = (patients2_full[f"{v}_cat"] // 2) % 2
    patients2_full[f"{v}_PEAK_dec"] = patients2_full[f"{v}_cat"] % 2

n = len(patients2_full)
X_raw = patients2_full[cluster_cols].to_numpy()
col_mean, col_sd = X_raw.mean(axis=0), X_raw.std(axis=0)
X = (X_raw - col_mean) / col_sd
B_cat = patients2_full[cat_cols].to_numpy(dtype=int)        # (n, 7), values in {0,...,7}
B_simple = patients2_full[constant_binary_vars].to_numpy()  # (n, 3)
block_idx = {v: [cluster_cols.index(f"{v}_{s}") for s in ("REST", "LEG", "PEAK")] for v in step_varying}
slope_idx = [cluster_cols.index(v) for v in slope_vars]

print(f"n = {n} patients, continuous d = {len(cluster_cols)}, categorical: "
      f"{len(peak_binary_vars)} Dirichlet-Multinomial flags + {len(constant_binary_vars)} Beta-Bernoulli flags")

# %% [markdown]
# ## Hyperparameters and conjugate updates

# %%
H = 15
RNG = np.random.default_rng(RANDOM_STATE)

# NIW hyperparameters, shared across clusters/variables, for each 3-dim continuous block.
p_blk = 3
m0_blk = np.zeros(p_blk)
kappa0_blk = 1.0
nu0_blk = p_blk + 2                                # = 5, smallest proper weakly-informative prior
Psi0_blk = np.eye(p_blk) * (nu0_blk - p_blk - 1)    # = I, so E[Sigma] = I on the standardized scale

# NIG hyperparameters for the 4 slope (1-D) dimensions.
m0, kappa0, a0, b0 = 0.0, 1.0, 2.0, 1.0

# Beta-Bernoulli hyperparameters for SESSO/RITMO/BA.
a_bin, b_bin = 1.0, 1.0

# Symmetric Dirichlet (alpha0=1) over the 8 joint REST/LEG/PEAK patterns, for the 7 flags.
n_cat2 = 8
alpha0_dir2 = np.ones(n_cat2)

EPS = 1e-9
ga_a, ga_b = 1.0, 1.0  # Gamma hyperprior on the DP concentration M


def niw_posterior(Y, rng):
    """NIW posterior draw for one 3-dim block, given Y: (n_h, 3)."""
    nj = len(Y)
    if nj == 0:
        Sigma = invwishart.rvs(df=nu0_blk, scale=Psi0_blk, random_state=rng)
        mu = rng.multivariate_normal(m0_blk, Sigma / kappa0_blk)
        return mu, Sigma
    ybar = Y.mean(axis=0)
    diff_y = Y - ybar
    S = diff_y.T @ diff_y
    kappa_n = kappa0_blk + nj
    nu_n = nu0_blk + nj
    m_n = (kappa0_blk * m0_blk + nj * ybar) / kappa_n
    d0 = (ybar - m0_blk).reshape(-1, 1)
    Psi_n = Psi0_blk + S + (kappa0_blk * nj / kappa_n) * (d0 @ d0.T)
    Sigma = invwishart.rvs(df=nu_n, scale=Psi_n, random_state=rng)
    mu = rng.multivariate_normal(m_n, Sigma / kappa_n)
    return mu, Sigma


def nig_posterior(y_col, rng):
    """NIG posterior draw for one 1-D slope dimension."""
    nj = len(y_col)
    if nj == 0:
        sigma2 = stats.invgamma.rvs(a0, scale=b0, random_state=rng)
        mu = rng.normal(m0, np.sqrt(sigma2 / kappa0))
        return mu, sigma2
    ybar = y_col.mean()
    ss = ((y_col - ybar) ** 2).sum()
    kappa_n = kappa0 + nj
    a_n = a0 + nj / 2
    b_n = b0 + 0.5 * ss + (kappa0 * nj * ybar ** 2) / (2 * kappa_n)
    m_n = (kappa0 * m0 + nj * ybar) / kappa_n
    sigma2 = stats.invgamma.rvs(a_n, scale=b_n, random_state=rng)
    mu = rng.normal(m_n, np.sqrt(sigma2 / kappa_n))
    return mu, sigma2


def beta_bernoulli_posterior(b_col, rng):
    nj = len(b_col)
    n1 = b_col.sum() if nj else 0
    return rng.beta(a_bin + n1, b_bin + (nj - n1))


def dirichlet_posterior(cat_col, rng):
    """Dirichlet posterior over the 8 joint REST/LEG/PEAK categories for one cluster/flag."""
    counts = np.bincount(cat_col, minlength=n_cat2) if len(cat_col) else np.zeros(n_cat2)
    return rng.dirichlet(alpha0_dir2 + counts)

# %% [markdown]
# ## Run the Gibbs sampler — as a reusable chain
#
# Alternate, each sweep: (a) resample every cluster's parameters from its conjugate posterior given
# only its currently-assigned patients, then the stick-breaking weights; (b) resample every patient's
# label, proportional to the combined likelihood of their data across both blocks under each
# cluster's current parameters; (c) update the DP concentration $M$ (Escobar & West, 1995).
#
# Written as a function of a single `seed` so the exact same sampler can be re-run as several
# **independent chains** below, each starting from its own random initial labels — that's what a
# multi-chain convergence check (R-hat) actually needs: not one long run, but several separate runs
# that either land on the same answer or don't.

# %%
n_iter, burn_in = 2000, 700


def run_gibbs_chain(seed, n_iter=n_iter, burn_in=burn_in):
    rng = np.random.default_rng(seed)
    r = rng.integers(0, 5, size=n)
    M = 1.0
    mu_blk = {v: np.zeros((H, 3)) for v in step_varying}
    Sigma_blk = {v: np.tile(np.eye(3), (H, 1, 1)) for v in step_varying}
    mu_slope = np.zeros((H, len(slope_vars)))
    sigma2_slope = np.ones((H, len(slope_vars)))
    theta_dir = {v: np.full((H, n_cat2), 1.0 / n_cat2) for v in peak_binary_vars}
    p_bin = np.full((H, len(constant_binary_vars)), 0.5)

    n_occupied_trace = np.zeros(n_iter, dtype=int)
    loglik_trace = np.zeros(n_iter)
    r_samples = np.zeros((n_iter - burn_in, n), dtype=int)

    for it in range(n_iter):
        members_mask = [r == h for h in range(H)]

        # (a) continuous block parameters
        for h in range(H):
            X_h = X[members_mask[h]]
            for v in step_varying:
                mu_blk[v][h], Sigma_blk[v][h] = niw_posterior(X_h[:, block_idx[v]], rng)
            for j, sidx in enumerate(slope_idx):
                mu_slope[h, j], sigma2_slope[h, j] = nig_posterior(X_h[:, sidx], rng)

        # categorical block parameters
        for h in range(H):
            Bc_h = B_cat[members_mask[h]]
            Bs_h = B_simple[members_mask[h]]
            for j, v in enumerate(peak_binary_vars):
                theta_dir[v][h] = dirichlet_posterior(Bc_h[:, j], rng)
            for l in range(len(constant_binary_vars)):
                p_bin[h, l] = beta_bernoulli_posterior(Bs_h[:, l], rng)

        # stick-breaking weights
        counts = np.bincount(r, minlength=H)
        Bc = np.array([counts[h + 1:].sum() for h in range(H)])
        v_stick = np.ones(H)
        v_stick[:-1] = rng.beta(1 + counts[:-1], M + Bc[:-1])
        w = v_stick * np.concatenate(([1.0], np.cumprod(1 - v_stick[:-1])))
        w = np.clip(w, 1e-300, None)
        w /= w.sum()
        log_w = np.log(w)

        # (b) joint reassignment: block-diagonal continuous log-lik + slope log-lik + binary log-lik
        log_lik = np.tile(log_w, (n, 1))
        for v in step_varying:
            Y_v = X[:, block_idx[v]]
            for h in range(H):
                log_lik[:, h] += multivariate_normal.logpdf(Y_v, mean=mu_blk[v][h], cov=Sigma_blk[v][h])
        for j, sidx in enumerate(slope_idx):
            diff2 = (X[:, sidx:sidx + 1] - mu_slope[:, j][None, :]) ** 2
            log_lik += -0.5 * (np.log(2 * np.pi * sigma2_slope[:, j][None, :]) + diff2 / sigma2_slope[:, j][None, :])
        for j, v in enumerate(peak_binary_vars):
            theta_log = np.log(np.clip(theta_dir[v], EPS, 1.0))
            log_lik += theta_log[:, B_cat[:, j]].T
        p_safe = np.clip(p_bin, EPS, 1 - EPS)
        log_lik += (B_simple[:, None, :] * np.log(p_safe)[None, :, :]
                    + (1 - B_simple[:, None, :]) * np.log(1 - p_safe)[None, :, :]).sum(axis=2)

        log_lik_raw = log_lik.copy()  # kept un-shifted, for a label-invariant fit diagnostic below
        log_lik -= log_lik.max(axis=1, keepdims=True)
        probs = np.exp(log_lik)
        probs /= probs.sum(axis=1, keepdims=True)
        r = np.array([rng.choice(H, p=probs[i]) for i in range(n)])

        # (c) update M
        k = len(np.unique(r))
        phi = rng.beta(M + 1, n)
        rate = ga_b - np.log(phi)
        pi_mix = (ga_a + k - 1) / (n * rate)
        pi_mix = pi_mix / (1 + pi_mix)
        M = rng.gamma(ga_a + k, 1 / rate) if rng.random() < pi_mix else rng.gamma(ga_a + k - 1, 1 / rate)

        n_occupied_trace[it] = k
        # Complete-data log-likelihood of this sweep's draw — label-invariant (doesn't care which
        # number a cluster is called), used below for the multi-chain R-hat check.
        loglik_trace[it] = log_lik_raw[np.arange(n), r].sum()
        if it >= burn_in:
            r_samples[it - burn_in] = r

    return n_occupied_trace, loglik_trace, r_samples


n_occupied_trace, loglik_trace, r_samples = run_gibbs_chain(RANDOM_STATE)
print(f"Chain 0 (seed={RANDOM_STATE}): ran {n_iter} sweeps ({burn_in} burn-in). "
      f"Post-burn-in occupied-cluster range: "
      f"{n_occupied_trace[burn_in:].min()}-{n_occupied_trace[burn_in:].max()}")

# %% [markdown]
# ## Checking convergence

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
# ## Resolving label-switching — Dahl's (2005) least-squares point estimate
#
# Cluster labels are arbitrary across sweeps, so the kept draws can't just be averaged directly.
# Build the patient-by-patient co-clustering matrix (how often two patients landed in the same
# cluster) and take the single sweep closest to it in squared error as the representative partition.

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
# ## Multi-chain convergence check — R-hat
#
# Everything above is one chain: one random starting assignment, one run. On its own, that can't
# tell us whether the sampler actually explored the real posterior or just settled somewhere and
# stayed — there's nothing to compare it against. So run 3 more **independent** chains, each from
# its own random starting labels, and check whether they agree.
#
# R-hat (Gelman & Rubin, 1992) is exactly that comparison, formalized: the ratio of between-chain
# variance to within-chain variance for some monitored quantity. $\hat{R}\approx 1$ means the chains
# agree; a large $\hat{R}$ means they don't.
#
# The one thing to get right for a mixture model: never compute R-hat directly on a cluster-indexed
# parameter like $\mu_3$. Cluster "3" in chain 1 and cluster "3" in chain 2 can be completely
# different patient groups — nothing forces consistent numbering across *independent* chains (this
# is the same label-switching problem Dahl's method solves *within* one chain, just showing up
# *across* chains here instead). So R-hat below is computed only on two **label-invariant**
# summaries that don't care what a cluster happens to be numbered: (1) the number of occupied
# clusters per sweep, (2) the complete-data log-likelihood per sweep. Separately, we also check
# whether the chains agree on the actual *partition* (who clusters with whom) via the Adjusted Rand
# Index between each chain's own Dahl point estimate — R-hat alone can look fine while the chains
# still disagree on membership, so this is the complementary check.

# %%
from sklearn.metrics import adjusted_rand_score

chain_seeds = [RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2, RANDOM_STATE + 3]
chains = [(n_occupied_trace, loglik_trace, r_samples)]  # chain 0, already run above
for seed in chain_seeds[1:]:
    chains.append(run_gibbs_chain(seed))
    print(f"Chain (seed={seed}): done.")


def gelman_rubin_rhat(draws_by_chain):
    """Classic Gelman & Rubin R-hat for a list of m equal-length 1-D post-burn-in draws."""
    draws = np.array(draws_by_chain)  # (m, n_draws)
    m, n_draws = draws.shape
    chain_means = draws.mean(axis=1)
    chain_vars = draws.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n_draws * chain_means.var(ddof=1)
    var_hat = ((n_draws - 1) / n_draws) * W + B / n_draws
    return np.sqrt(var_hat / W)


post_occupied = [occ[burn_in:] for occ, _, _ in chains]
post_loglik = [ll[burn_in:] for _, ll, _ in chains]
rhat_k = gelman_rubin_rhat(post_occupied)
rhat_loglik = gelman_rubin_rhat(post_loglik)
print(f"R-hat, # occupied clusters: {rhat_k:.3f}")
print(f"R-hat, complete-data log-likelihood: {rhat_loglik:.3f}")
print("(Rule of thumb: R-hat < 1.01-1.05 is considered converged.)")

fig, ax = plt.subplots(figsize=(10, 4))
for idx, (occ, _, _) in enumerate(chains):
    ax.plot(occ, lw=0.7, label=f"chain {idx} (seed={chain_seeds[idx]})")
ax.axvline(burn_in, color="red", ls="--", label="burn-in cutoff")
ax.set_xlabel("Sweep")
ax.set_ylabel("# occupied clusters")
ax.set_title("4 independent chains — number of occupied clusters per sweep")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# Does each chain not just agree on HOW MANY clusters, but on WHICH patients group together?
chain_partitions = []
for occ, ll, r_s in chains:
    co = np.zeros((n, n))
    for s in range(r_s.shape[0]):
        co += (r_s[s][:, None] == r_s[s][None, :])
    co /= r_s.shape[0]
    best_sse_c, best_idx_c = np.inf, 0
    for s in range(r_s.shape[0]):
        same = (r_s[s][:, None] == r_s[s][None, :]).astype(float)
        sse_c = ((same - co) ** 2).sum()
        if sse_c < best_sse_c:
            best_sse_c, best_idx_c = sse_c, s
    chain_partitions.append(r_s[best_idx_c])

print("\nPer-chain Dahl point estimate:")
for idx, part in enumerate(chain_partitions):
    _, relab = np.unique(part, return_inverse=True)
    print(f"  chain {idx}: {relab.max()+1} clusters, sizes {np.bincount(relab)}")

print("\nPairwise Adjusted Rand Index between chains' point-estimate partitions "
      "(1.0 = identical partition, 0.0 = no better than random):")
ari_table = pd.DataFrame(
    [[adjusted_rand_score(chain_partitions[i], chain_partitions[j]) for j in range(4)] for i in range(4)],
    index=[f"chain {i}" for i in range(4)], columns=[f"chain {j}" for j in range(4)],
)
print(ari_table.round(3))

# %% [markdown]
# ## Reporting — cluster profiles, both blocks

# %%
profile_df = patients2_full.copy()
profile_df["cluster"] = reference_r
order_by_severity = profile_df.groupby("cluster")["PAWP.MEDIA_PEAK"].mean().sort_values().index
severity_map = {old: new for new, old in enumerate(order_by_severity, start=1)}
profile_df["cluster"] = profile_df["cluster"].map(severity_map)

print("Our cluster sizes:", dict(profile_df["cluster"].value_counts().sort_index()))
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

print("\nPatient-constant prevalence by cluster (SESSO/RITMO/BA):")
print(profile_df.groupby("cluster")[constant_binary_vars].mean().round(3))

binary_profile_cols = [f"{v}_PEAK_dec" for v in peak_binary_vars] + constant_binary_vars
fig, ax = plt.subplots(figsize=(10, 4))
plot_df = profile_df.groupby("cluster")[binary_profile_cols].mean()
plot_df.columns = peak_binary_vars + constant_binary_vars
plot_df.T.plot(kind="bar", ax=ax)
ax.set_ylabel("Prevalence")
ax.set_title("Binary/categorical phenotype prevalence by cluster\n"
             "(7 flags shown at PEAK for readability; the model fit the full REST/LEG/PEAK pattern)")
ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Where this leaves us:** the posterior favours a single value of $k$ with high probability
# (shown above), compared against both Phase 0's imposed $k$-means partition and the paper's
# published sizes. 4 independent chains agree on both how many clusters there are (R-hat above) and
# on who clusters with whom (the pairwise Adjusted Rand Index table) - the two convergence checks
# the single-chain version couldn't make on its own. Next for the write-up: a short discussion of
# what distinguishes each cluster clinically (the profile tables above), the posterior credible
# intervals per cluster parameter (currently point draws from one representative sweep per chain,
# not yet summarized with uncertainty), and the predictive check for a new, out-of-sample patient.
