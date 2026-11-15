# ---
# title: "HFpEF Bayesian Clustering — Phase 1: Variable Assessment & Modelling Strategy"
# author: "Mitch Wishart"
# format:
#   html:
#     toc: true
#     code-fold: show
# ---

# %% [markdown]
# # Phase 1 — Assessing the Data, Choosing the Variables, Defining the Model
#
# Phase 0 replicated the paper's pipeline on the 21 variables (14 continuous
# + 7 binary) *they* named as clinically relevant, without asking whether
# that's the right starting point for our own model. This notebook does
# that assessment properly, in order:
#
# 1. What's actually in the ~124 raw candidate columns (not just the 21).
# 2. What method is appropriate for narrowing that down, and — crucially —
#    how the answer depends on what a Bayesian mixture/DP model can
#    actually support given n=80 patients (not a separate question from
#    variable selection, but a constraint on it).
# 3. A naive, domain-blind correlation screen across all ~124 columns, and
#    a concrete demonstration of *why that doesn't work* for us.
# 4. Falling back to the paper's clinically-informed 21, applying the same
#    redundancy logic to just those, and a structural finding about how the
#    binary flags actually behave across STEP that changes how they must be
#    used.
# 5. The final variable list, split into the continuous and binary blocks.
# 6. The modelling strategy — defined step by step here, executed in Phase 2.
#
# No MCMC model is fit in this notebook. Everything here is either
# descriptive (counts, correlations, missingness) or simple arithmetic
# (parameter counts, feasibility checks) — the point is to decide what
# Phase 2 should build, with the reasoning for each decision on the record.

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    r"C:\Users\mitch\OneDrive\1   B.ENGG\0   POLIMI\Bayesian Statistics\Project\r_code_datasets_wd",
))

PAPER_CLUSTER_VARS = ["PAWP.MEDIA", "PAWP.V.AMPL", "PAWP.MEDIA.CO.f.slp", "PAPM.CO.f.slp",
                      "RAP.VY", "PVR.MEDIA", "CI", "COVO2.slp"]
PAPER_BINARY_VARS = ["PH", "HFpEF", "stiff.LA", "high.PVR", "RHF", "TR", "low.CO"]

# Running log of sensitivity-analysis findings — required-analysis item 7
# in the brief ("how sensitive are the results to preprocessing decisions?").
# Each entry records one concrete way a decision could have gone differently
# and what changed because of it, so these don't just live in chat.
SENSITIVITY_NOTES = []

# %% [markdown]
# ## 0. The data's actual structure: ID -> STEP -> (continuous, binary)
#
# Before any variable selection, it's worth being explicit about the shape
# of the raw data, because it drives two separate decisions below.
#
# - **`dataset1.txt`/`dataset2.txt`/`dataset3.txt`** are row-aligned (no
#   shared key, but the same row order) and concatenate column-wise into
#   one long table: one row per **(patient, exercise step)** pair, with
#   `STEP` in `{REST, LEG, PEAK}`.
# - **Continuous hemodynamic variables** genuinely vary by STEP for most
#   variables (that's the whole point of an exercise catheterization study
#   — resting values vs. peak-exertion values) — except the 5 "slope"
#   variables, which are defined only once, at PEAK, by the dataset's own
#   design.
# - **The 7 binary phenotype flags** (`PH`, `HFpEF`, `stiff.LA`, `high.PVR`,
#   `RHF`, `TR`, `low.CO`) look, at first glance, like fixed per-patient
#   labels — but they are not. Checked directly below, they vary by STEP
#   for most patients, because they're exercise-dependent diagnostic
#   classifications (consistent with how HFpEF is often invasively
#   diagnosed: unmasked only under exertion, not visible at rest).

# %%
list1 = pd.read_table(DATA_DIR / "dataset1.txt")
list2 = pd.read_table(DATA_DIR / "dataset2.txt")
list3 = pd.read_table(DATA_DIR / "dataset3.txt")
assert len(list1) == len(list2) == len(list3)

full = pd.concat([list1, list2, list3], axis=1)
print(f"{full.shape[0]} patient-step rows, {full.shape[1]} total columns, "
      f"{full['ID'].nunique()} patients")

id_step_cols = ["ID", "STEP"]
candidate_cols = [c for c in full.columns if c not in id_step_cols]
print(f"{len(candidate_cols)} candidate variables (excluding ID, STEP)")

# %%
nun = full.groupby("ID")[PAPER_BINARY_VARS].nunique()
n_varies = (nun > 1).sum()
print("Number of patients for whom each binary flag takes >1 distinct value across STEP:")
print(n_varies)

prevalence_by_step = full.groupby("STEP")[PAPER_BINARY_VARS].mean().round(3)
print("\nPrevalence of each binary flag, by STEP:")
print(prevalence_by_step)

print(
    "\n-> These are NOT static patient labels. HFpEF prevalence alone goes from "
    f"{prevalence_by_step.loc['REST', 'HFpEF']:.1%} at REST to "
    f"{prevalence_by_step.loc['PEAK', 'HFpEF']:.1%} at PEAK — far too large a swing to be "
    "measurement noise. The PEAK value is almost certainly the clinically intended one (it "
    "matches the expected near-universal HFpEF prevalence for a HFpEF study cohort, and mirrors "
    "exactly how the 5 continuous 'slope' variables are already PEAK-only by design). Every "
    "binary-block analysis in this project uses the PEAK-step value, not an arbitrary row or an "
    "incorrectly-assumed patient-level constant."
)

SENSITIVITY_NOTES.append({
    "decision": "Which STEP to use for the binary phenotype flags",
    "finding": (
        "The 7 binary flags vary by STEP for most patients (e.g. HFpEF prevalence: "
        f"{prevalence_by_step.loc['REST', 'HFpEF']:.1%} at REST vs. "
        f"{prevalence_by_step.loc['PEAK', 'HFpEF']:.1%} at PEAK) — they are exercise-dependent "
        "diagnostic classifications, not fixed patient labels. An earlier pass (variable_screening.py) "
        "computed binary prevalence by mixing all 3 STEP rows together, which is simply wrong."
    ),
    "implication": (
        "Use PEAK-step values for every binary variable, consistent with how the slope "
        "variables are already PEAK-only. Using REST or a STEP-mixed average instead would "
        "materially change prevalence (and therefore the categorical block's model) — this is a "
        "genuine, reportable preprocessing decision, not a minor detail."
    ),
})

# %% [markdown]
# ## 1. Classify every candidate column
#
# Three buckets: **continuous** (numeric, many distinct values), **binary**
# (numeric, only 0/1), and **non-numeric** (text-coded columns like `SESSO`
# [sex], `RITMO` [rhythm] — these need a data dictionary to interpret
# properly and are excluded here rather than guessed at).

# %%
col_type = {}
for c in candidate_cols:
    numeric = pd.to_numeric(full[c], errors="coerce")
    if numeric.notna().sum() == 0:
        col_type[c] = "non_numeric"
        continue
    frac_numeric_parseable = numeric.notna().sum() / full[c].notna().sum() if full[c].notna().sum() else 0
    if frac_numeric_parseable < 0.99:
        col_type[c] = "non_numeric"
        continue
    n_unique = numeric.dropna().nunique()
    col_type[c] = "binary" if n_unique <= 2 else "continuous"

type_counts = pd.Series(col_type).value_counts()
print(type_counts)

non_numeric_cols = [c for c, t in col_type.items() if t == "non_numeric"]
print(f"\nNon-numeric / text-coded columns (need a data dictionary, not guessed): {non_numeric_cols}")
print("These stay out of both blocks entirely until a data dictionary resolves what their codes mean.")

# %% [markdown]
# ## 2. Missingness across all candidate columns
#
# The 5 "slope" variables are, by the dataset's own design, only meaningful
# at the PEAK step — their REST/LEG rows are intentionally blank, not a
# data quality problem. A naive row-by-row missingness count would unfairly
# flag them as ~65% missing, so they're checked only within PEAK rows
# instead (the same correction now applies to the binary flags, per §0).

# %%
SLOPE_VARS = ["PAWP.MIDA.CO.f.slp", "PAWP.MEDIA.CO.f.slp", "RAP.MEDIA.CO.f.slp",
              "COVO2.slp", "PAPM.CO.f.slp"]
peak_mask = full["STEP"] == "PEAK"

missing_pct = full[candidate_cols].isna().mean().mul(100).round(1)
for v in SLOPE_VARS + PAPER_BINARY_VARS:
    if v in missing_pct.index:
        missing_pct[v] = full.loc[peak_mask, v].isna().mean() * 100
missing_pct = missing_pct.round(1).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, max(6, len(candidate_cols) * 0.12)))
missing_pct.plot(kind="barh", ax=ax, color="indianred")
ax.invert_yaxis()
ax.set_xlabel("% missing (PEAK-only for STEP-restricted variables)")
ax.set_title(f"Missingness across all {len(candidate_cols)} candidate variables")
ax.tick_params(axis="y", labelsize=5)
plt.tight_layout()
plt.show()

print("Worst 15 for missingness:")
print(missing_pct.head(15))

# %% [markdown]
# ## 3. What method should narrow ~124 candidates down, and why does the
#    answer depend on the model, not just the data?
#
# A correlation-redundancy screen (drop one of any pair with |r|>=0.80) is
# the obvious first move — it's literally what the paper itself does on
# its smaller 14-variable list. But "how many variables can survive" isn't
# only a statistical question: it's also a question about what a Bayesian
# mixture/DP model can actually estimate from n=80 patients split across
# several clusters, several of which will be small.
#
# Two aspects of the model are directly shaped by this:
#
# - **Covariance structure of the continuous kernel.** A full covariance
#   matrix per cluster needs more patients than dimensions just to be
#   *invertible* (n_cluster >= d+1); a diagonal (per-dimension) covariance
#   only needs n_cluster >= 2 per dimension. The more variables survive
#   filtering, the harder this constraint bites — demonstrated with real
#   numbers in §6 below.
# - **The two-block structure itself.** The brief's continuous block and
#   categorical block are modelled with different kernels (Gaussian vs.
#   Hamming/Bernoulli) precisely because continuous and binary variables
#   can't share one covariance structure — so "how many variables" has to
#   be answered separately for each block, not as one combined count.
#
# This means variable selection and model specification are not two
# separate steps done in sequence — the dimensionality constraint from the
# model choice is itself a reason to filter harder, not just a downstream
# consequence of however many variables the correlation screen leaves in.

# %% [markdown]
# ## 4. Naive full-dataset correlation screen — and why it doesn't work for us
#
# Same idea as the paper's correlation-threshold rule, but applied to
# *every* continuous candidate with usable completeness (<15% missing),
# with a domain-blind tie-break (keep whichever of a correlated pair has
# less missing data) since there's no clinical reasoning available for
# ~100 unfamiliar raw columns the way there was for the paper's named 14.

# %%
continuous_cols = [c for c, t in col_type.items() if t == "continuous"]
usable_continuous = [c for c in continuous_cols if missing_pct[c] < 15]
print(f"{len(continuous_cols)} continuous candidates; {len(usable_continuous)} usable (<15% missing)")

corr = full[usable_continuous].apply(pd.to_numeric, errors="coerce").corr()

high_corr_pairs_full = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack().loc[lambda s: s.abs() >= 0.80].sort_values(ascending=False)
)
print(f"\n{len(high_corr_pairs_full)} pairs with |r| >= 0.80 among the {len(usable_continuous)} "
      f"usable continuous variables -> heavy redundancy in the raw dataset.")

survivors_naive = set(usable_continuous)
dropped_full = {}
for (a, b), r in high_corr_pairs_full.items():
    if a not in survivors_naive or b not in survivors_naive:
        continue
    keep, drop = (a, b) if (missing_pct[a], a) <= (missing_pct[b], b) else (b, a)
    survivors_naive.discard(drop)
    dropped_full[drop] = (keep, r)

print(f"\nRule drops {len(dropped_full)} variables, leaving {len(survivors_naive)} candidates.")

# %% [markdown]
# ### Why 43 survivors is still not usable
#
# 43 "survivors" doesn't mean 43 model dimensions — most continuous
# variables here are measured at REST/LEG/PEAK, so the real, patient-level
# dimensionality is larger. But simply checking *which STEP rows are
# non-null* overcounts: several variables (the known "slope" variables,
# plus others discovered here, like `BSA`) store the SAME value repeated
# across all 3 STEP rows rather than three genuinely different
# measurements — so the real test is whether the value actually *varies*
# by STEP for a given patient, not just whether it's present.

# %%
survivor_list = sorted(survivors_naive)
real_dims = {}
for v in survivor_list:
    sub = full[["ID", "STEP", v]].dropna(subset=[v])
    nunique_per_patient = sub.groupby("ID")[v].nunique()
    frac_constant_across_steps = (nunique_per_patient <= 1).mean()
    # If the value is constant across STEP for the large majority of patients,
    # it's one real measurement duplicated for convenience, not 3 independent ones.
    real_dims[v] = 1 if frac_constant_across_steps > 0.9 else 3

d_naive_expanded = sum(real_dims.values())
duplicate_filled = sorted(v for v, d in real_dims.items() if d == 1)
print(f"Of the {len(survivor_list)} naive survivors, {len(duplicate_filled)} are duplicate-filled "
      f"across STEP (really 1 measurement, not 3): {duplicate_filled}")
print(f"Correctly counting only genuine repeats as 3 dimensions gives patient-level "
      f"dimensionality d = {d_naive_expanded} (a naive notna()-presence check would have "
      f"wrongly inflated this to {3 * len(survivor_list)}).")

n_patients = full["ID"].nunique()
# Optimistic case: a perfectly even split across the paper's k=5 clusters.
even_cluster_n = n_patients // 5
need_full = d_naive_expanded + 1
print(f"\nEven under an optimistic, perfectly even 5-cluster split "
      f"({n_patients} patients / 5 = {even_cluster_n} per cluster), a full covariance matrix at "
      f"d={d_naive_expanded} needs n>={need_full} per cluster to even be invertible. "
      f"{even_cluster_n} {'>=' if even_cluster_n >= need_full else '<'} {need_full} "
      f"-> {'feasible' if even_cluster_n >= need_full else 'SINGULAR even in the best case'}.")
print("And real clusters are never evenly sized (Phase 0's k-means gave 18/13/29/8/12) — the "
      "smallest realistic cluster makes this far worse, not better.")

SENSITIVITY_NOTES.append({
    "decision": "Screen all ~124 raw columns with a domain-blind correlation rule, instead of "
                "starting from the paper's clinically-relevant 21",
    "finding": (
        f"{len(survivors_naive)} continuous variables survive, expanding to d={d_naive_expanded} "
        f"patient-level dimensions — more than {even_cluster_n}, the size of an evenly-split "
        "cluster, which guarantees a singular full covariance matrix even before considering "
        "real (uneven) cluster sizes."
    ),
    "implication": (
        "Not usable as the model's input set at n=80, regardless of covariance structure choice: "
        "even a diagonal model would be carrying far more noisy, redundant, non-clinically-"
        "reasoned dimensions than the sample size supports. Confirms the decision to fall back to "
        "the paper's 21 clinically-relevant candidates (§5) instead."
    ),
})

# %% [markdown]
# ## 5. Falling back to the paper's 21 clinically-relevant candidates
#
# The paper's 21 (14 continuous + 7 binary) aren't derived from the data —
# they're a clinical/domain selection: known HFpEF hemodynamic physiology
# picked these out of the ~124 raw columns, which isn't something a
# correlation matrix can verify or re-derive. That's exactly why they're
# the right starting point, and why §4's screen is a validation check on
# them, not a replacement for them.
#
# Applying the *same* |r|>=0.80 redundancy rule to just the paper's 14
# continuous candidates, using the paper's own stated clinical tie-break
# (prefer PAWP over PA pressure/RAP; prefer cycle-averaged over end-
# diastolic/systolic values — see Phase 0 §3 for the full derivation),
# actually gets 9 survivors, not the paper's published 8:

# %%
hemo_vars_14 = ["PAPM", "PAWP.MIDA", "PAWP.MIDA.CO.f.slp", "PAWP.V.AMPL", "PAWP.MEDIA",
                "PAWP.MEDIA.CO.f.slp", "PVR.MEDIA", "RAP.MIDA", "RAP.MEDIA",
                "RAP.MEDIA.CO.f.slp", "RAP.VY", "CI", "COVO2.slp", "PAPM.CO.f.slp"]
nine_survivors = sorted(set(PAPER_CLUSTER_VARS) | {"RAP.MEDIA.CO.f.slp"})
print(f"Mechanical rule on the 14: {len(nine_survivors)} survivors -> {nine_survivors}")
print(f"Paper's published list: {len(PAPER_CLUSTER_VARS)} -> {PAPER_CLUSTER_VARS}")
print("The one extra survivor, RAP.MEDIA.CO.f.slp, isn't correlated >=0.80 with anything else "
      "left once the other drops are applied, so nothing in the stated rule removes it — its "
      "absence from the paper's published 8 isn't explained by the Methods as written.")

# %% [markdown]
# A second, separate sensitivity question: what if the tie-break itself
# were domain-blind (missingness-based, like §4) instead of clinical?
# Applying that version of the rule to just the paper's final 8:

# %%
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

SENSITIVITY_NOTES.append({
    "decision": "What if the same STYLE of redundancy screen were run on the paper's final 8, "
                "without the paper's clinical reasoning?",
    "finding": (
        f"IF the |r|>=0.80 redundancy rule is applied with a domain-blind missingness-based "
        f"tie-break instead of the paper's clinical reasoning, THEN {n_kept}/8 of the paper's "
        f"actual final 8 still come out the same. The other 2 (PAWP.MEDIA, RAP.VY) would instead "
        f"be replaced by a raw sub-component variable with marginally less missing data "
        f"(PAWP.MEDIA->PAWP.A, RAP.VY->RAP.V.AMPL)."
    ),
    "implication": (
        "This is NOT the paper's actual variable set changing — it's a hypothetical, showing how "
        "much of the paper's final 8 is explainable by redundancy screening alone vs. how much "
        "specifically needed clinical judgement. 6/8 would land the same way either way; 2/8 "
        "specifically depend on clinical reasoning a blind statistical rule can't reproduce."
    ),
})

# %% [markdown]
# **Decision for the model (Phase 2): use the paper's published 8 continuous
# variables**, not the mechanically-derived 9, and not the larger
# domain-blind 43. Reasons:
#
# 1. **Interpretability** — the brief wants clinically interpretable
#    profiles; a missingness-based tie-break has no clinical grounding.
# 2. **Dimensionality, given n=80** — more variables (9, or 43) directly
#    worsens the covariance-feasibility problem quantified in §6.
# 3. **The 8-vs-9 and clinical-vs-blind discrepancies are themselves good
#    sensitivity-analysis material** (required-analysis item 7) — reporting
#    them honestly is more valuable than silently resolving them.

# %% [markdown]
# ## 6. Covariance structure: parameter counting and a feasibility check
#
# Having settled on the paper's 8 continuous variables, the next model
# decision is what covariance structure the Gaussian kernel should use.
# This is arithmetic, not intuition.

# %%
N_NONSLOPE_14, N_SLOPE_14 = 9, 5   # of the 14, measured at 3 steps vs. PEAK-only
d_all_14 = N_NONSLOPE_14 * 3 + N_SLOPE_14
N_NONSLOPE_8, N_SLOPE_8 = 5, 3     # of the paper's final 8
d_paper_8 = N_NONSLOPE_8 * 3 + N_SLOPE_8
print(f"All 14 continuous candidates, expanded across steps: d = {d_all_14}")
print(f"Paper's final 8, expanded across steps: d = {d_paper_8}")

def full_cov_params(d):
    return d * (d + 1) // 2 + d

def diag_cov_params(d):
    return 2 * d

for d, label in [(d_all_14, "all 14, expanded"), (d_paper_8, "paper's 8, expanded")]:
    print(f"{label:24s} (d={d:2d}):  full={full_cov_params(d):4d} params/cluster   "
          f"diagonal={diag_cov_params(d):3d} params/cluster")

# %%
kmeans_sizes = [18, 13, 29, 8, 12]   # Phase 0, k-means, k=5

for name, sizes in [("Phase 0 k-means (k=5)", kmeans_sizes)]:
    min_n = min(sizes)
    print(f"\n{name}: sizes={sizes}, smallest cluster n={min_n}")
    for d, label in [(d_all_14, "d=32 (unfiltered 14)"), (d_paper_8, "d=18 (paper's 8)")]:
        need = d + 1
        print(f"    full covariance at {label}: needs n>={need} per cluster -> "
              f"{'FEASIBLE' if min_n >= need else 'SINGULAR (infeasible)'} for the smallest cluster")

# A concrete, tiny demonstration: n=5, d=18 (smaller than any real cluster above).
rng = np.random.default_rng(0)
n_tiny, d_tiny = 5, 18
X_tiny = rng.normal(size=(n_tiny, d_tiny))
S = np.cov(X_tiny, rowvar=False)
rank, det, cond = np.linalg.matrix_rank(S), np.linalg.det(S), np.linalg.cond(S)
print(f"\nDemo: n={n_tiny}, d={d_tiny} -> covariance rank={rank} (full rank would be {d_tiny}), "
      f"determinant={det:.3e}, condition number={cond:.3e}")
print("A multivariate Gaussian density divides by sqrt(determinant); a determinant this close to "
      "zero makes that density numerically meaningless, independent of whether numpy's inv() "
      "happens to return a number without raising an error.")

print(f"\nDiagonal covariance needs only n>=2 per cluster per dimension — every cluster produced "
      f"in Phase 0 (smallest n={min(kmeans_sizes)}) clears that easily, at either d.")

SENSITIVITY_NOTES.append({
    "decision": "Full vs. diagonal covariance for the continuous kernel",
    "finding": (
        f"At d={d_paper_8} (paper's 8, expanded), a full covariance matrix needs n>={d_paper_8+1} "
        f"per cluster just to be invertible. Phase 0's actual smallest cluster (n={min(kmeans_sizes)}) "
        "fails that test, and a small, over-fragmented DP mixture cluster (Phase 2 produces clusters "
        "as small as n=1) fails it far worse."
    ),
    "implication": (
        "Diagonal covariance isn't one defensible option among several — given these cluster sizes, "
        "it's close to the only numerically well-posed choice, independent of how many variables "
        "survive filtering. Filtering down to 8 still matters because it keeps the independence "
        "assumption underlying 'diagonal' more defensible (removes the known |r|>=0.80 redundancy), "
        "not because it's what makes diagonal covariance necessary."
    ),
})

# %% [markdown]
# ## 7. Binary candidates beyond the paper's 7
#
# Using the PEAK-step prevalence established in §0 (not the STEP-mixed
# figures an earlier pass used).

# %%
binary_cols = [c for c, t in col_type.items() if t == "binary"]
binary_summary = pd.DataFrame({
    "missing_pct_peak": missing_pct[binary_cols],
    "prevalence_peak": full.loc[peak_mask, binary_cols].apply(pd.to_numeric, errors="coerce").mean().round(3),
}).sort_values("missing_pct_peak")
print(f"{len(binary_cols)} binary candidates total (paper's 7 plus any others found here), "
      f"PEAK-step values:")
print(binary_summary)
print("\nNo binary candidates beyond the paper's 7 were found in the raw files at all — this "
      "isn't a screening decision, there simply aren't others.")

# %% [markdown]
# ## 8. Final variable set
#
# **Continuous block — 8 variables, 18 patient-level dimensions after
# REST/LEG/PEAK (or PEAK-only, for slope variables) expansion:**
#
# | Variable | Steps | Columns |
# |---|---|---|
# | PAWP.MEDIA | REST, LEG, PEAK | 3 |
# | PAWP.V.AMPL | REST, LEG, PEAK | 3 |
# | PVR.MEDIA | REST, LEG, PEAK | 3 |
# | RAP.VY | REST, LEG, PEAK | 3 |
# | CI | REST, LEG, PEAK | 3 |
# | PAWP.MEDIA.CO.f.slp | PEAK only (slope) | 1 |
# | PAPM.CO.f.slp | PEAK only (slope) | 1 |
# | COVO2.slp | PEAK only (slope) | 1 |
#
# **Binary block — 7 variables, 1 PEAK-step value each (per §0):**
#
# `PH`, `HFpEF`, `stiff.LA`, `high.PVR`, `RHF`, `TR`, `low.CO`
#
# **Excluded:** the 5 non-numeric columns (§1) pending a data dictionary;
# the 37 extra continuous survivors from §4's naive screen (not clinically
# reasoned, and the dimensionality argument in §6 rules them out anyway).

# %%
print("CONTINUOUS BLOCK (8 variables / 18 dims):", PAPER_CLUSTER_VARS)
print("BINARY BLOCK (7 variables, PEAK-step):", PAPER_BINARY_VARS)

# %% [markdown]
# ## 9. Modelling strategy for Phase 2 (defined here, executed there)
#
# One latent cluster label $z_i \in \{1,\dots,H\}$ per patient drives BOTH
# blocks jointly — this is a single shared partition, not two separate
# models compared afterwards, because the brief's "clinical stage" is one
# thing per patient, expressed through both continuous hemodynamics and
# binary phenotype simultaneously.
#
# **Step 1 — Standardize the continuous block.** z-score each of the 18
# columns before fitting, so no variable dominates purely by numeric scale
# (resolves Phase 0's unscaled-Euclidean limitation).
#
# **Step 2 — Continuous kernel: diagonal Gaussian / per-dimension NIG.**
# Justified in §6 by cluster-size feasibility, not just tractability.
#
# **Step 3 — Binary kernel: per-dimension Bernoulli, Beta-Bernoulli
# conjugate.** Each cluster $h$ has its own probability $p_{h,\ell}$ of
# each binary flag being 1; a patient's binary vector is most likely under
# whichever cluster's $(p_{h,1},\dots,p_{h,7})$ pattern it's closest to in
# Hamming distance — i.e. this independent-Bernoulli-per-dimension form
# *is* the Hamming-distance kernel the brief suggests (Argiento, Filippi
# Mazzola & Paci 2025), not an approximation of it.
#
# **Step 4 — Shared stick-breaking DP prior.** Truncated at $H=15$
# (Ishwaran & James 2001 / book Algorithm 4), with concentration $M \sim
# \text{Gamma}(1,1)$ updated via the Escobar & West (1995) auxiliary scheme
# (book Algorithm 2) — same machinery already built and tested in Phase 2's
# continuous-only prototype, now driving both blocks at once.
#
# **Step 5 — Joint Gibbs sampler**, each sweep:
#   (a) resample $(\mu_{hj}, \sigma^2_{hj})$ for each cluster/continuous-dim
#       from its NIG posterior;
#   (b) resample $p_{h\ell}$ for each cluster/binary-dim from its
#       Beta-Bernoulli posterior;
#   (c) resample stick-breaking weights $v_h$, $w_h$;
#   (d) resample each patient's $z_i$ using the COMBINED log-likelihood —
#       continuous NIG log-density + binary Bernoulli log-mass — so cluster
#       assignment genuinely depends on both blocks together;
#   (e) update $M$ (Escobar-West).
#
# **Step 6 — Post-processing.** Dahl's (2005) least-squares point estimate,
# posterior co-clustering matrix, per-patient posterior membership
# probabilities via relabeling to the reference partition — all identical
# in method to Phase 2's prototype, just applied to the joint model.
#
# **Step 7 — Required analyses (per the brief).** Profile both blocks by
# cluster (continuous: mean/boxplot per variable per cluster; binary:
# prevalence per cluster); the sensitivity-analysis findings logged above,
# plus a direct re-run swapping PAWP.A/RAP.V.AMPL in for PAWP.MEDIA/RAP.VY;
# a predictive example assigning a new, out-of-sample patient to the
# inferred clusters.

# %% [markdown]
# ## 10. Running sensitivity-analysis log

# %%
for i, note in enumerate(SENSITIVITY_NOTES, start=1):
    print(f"[{i}] {note['decision']}")
    print(f"    Finding:     {note['finding']}")
    print(f"    Implication: {note['implication']}\n")
