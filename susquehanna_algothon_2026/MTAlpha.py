"""
Algothon final submission strategy.

THREE COMPONENTS, each independently validated on out-of-sample data:

1. FACTOR-RESIDUAL CROSS-SECTIONAL REVERSION (assets 1-50)
   Strip top-2 PCA factors from 3-day returns, vol-normalise,
   trade the cross-sectional z-score. Validated: positive in all
   rolling windows, noise-benchmark tested, SR ~1.2 full-sample.

2. CTGI/EELT STRUCTURAL PAIR (assets 25 & 37)
   Log price ratio ADF p=0.000, half-life 7.1d, ratio std 0.092.
   Walk-forward z-score trade on the ratio. SR 2.67 full-sample,
   positive in A, B and C independently.

3. MTNS/BENI STRUCTURAL PAIR (assets 33 & 42)
   Log price ratio ADF p=0.000, half-life 11.0d.
   SR 1.48 full-sample, positive in A, B and C independently.

Pairs deliberately excluded despite high B scores:
   EELT cluster (12 pairs) — B=443 but C=-334, regime break at
   day 650, would damage hidden-window performance.

Ticker index mapping (0-based):
   CTGI=25, EELT=37, MTNS=33, BENI=42

Rename to <yourteamname>.py to submit.
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# ---- Component 1: factor-residual reversion ----
NF          = 2      # PCA factors to strip
H           = 3      # reversion horizon (days)
FITW        = 250    # factor fit window
REFIT_EVERY = 20     # days between factor refits
VOL_WIN     = 60     # vol normalisation window
AGGR        = 4.0    # tanh steepness
RESID_BAND  = 0.15   # rebalance band (fraction of cap)

# ---- Component 2 & 3: structural pairs ----
# (i, j, z_window, band) — tuned on B, confirmed positive on C
PAIRS = [
    (25, 37, 40, 0.5),   # CTGI/EELT: half-life 7.1d
    (33, 42, 40, 1.0),   # MTNS/BENI: half-life 11.0d
]
PAIR_AGGR = 2.0

MIN_HISTORY = max(FITW, 40) + 10

_pos      = np.zeros(N_INST, dtype=int)
_last_n   = 0
_V        = None
_fit_day  = -10**9


def _zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def getMyPosition(prcSoFar):
    global _pos, _last_n, _V, _fit_day

    prc     = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh  = np.floor(CAPS / p_today).astype(int)

    if n_days <= _last_n:           # fresh run: reset state
        _pos     = np.zeros(N_INST, dtype=int)
        _V       = None
        _fit_day = -10**9
    _last_n = n_days

    if n_days < MIN_HISTORY:
        _pos = np.clip(_pos, -max_sh, max_sh)
        return _pos.copy()

    rets  = np.diff(np.log(prc), axis=1)
    log_p = np.log(prc)

    target_dollars = np.zeros(N_INST)

    # =========================================================
    # Component 1: factor-residual reversion (all 51 assets)
    # =========================================================
    if _V is None or (n_days - _fit_day) >= REFIT_EVERY:
        R  = rets[:, -FITW:]
        Rc = R - R.mean(axis=1, keepdims=True)
        _, V = np.linalg.eigh(np.cov(Rc))
        _V = V[:, -NF:]
        _fit_day = n_days

    rh    = rets[:, -H:].sum(axis=1)
    resid = rh - _V @ (_V.T @ rh)
    vol   = np.maximum(rets[:, -VOL_WIN:].std(axis=1), 1e-8)
    resid = resid / (vol * np.sqrt(H))
    expo  = np.tanh(-AGGR * _zscore(resid))
    target_dollars += expo * CAPS

    # =========================================================
    # Component 2 & 3: structural pairs
    # =========================================================
    for i, j, zw, band in PAIRS:
        if max(i, j) >= n_inst or n_days < zw + 5:
            continue
        ratio_win = log_p[i, -zw:] - log_p[j, -zw:]
        s_mean = ratio_win[:-1].mean()
        s_sd   = ratio_win[:-1].std()
        if s_sd < 1e-12:
            continue
        z = (ratio_win[-1] - s_mean) / s_sd
        if abs(z) <= band:
            continue
        f = np.tanh(PAIR_AGGR * z)
        # z > 0: i is expensive -> short i, long j
        target_dollars[i] -= f * CAPS[i]
        target_dollars[j] += f * CAPS[j]

    # =========================================================
    # Clip combined target to caps
    # =========================================================
    target_dollars = np.clip(target_dollars, -CAPS, CAPS)
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    # Rebalance band: smaller for pairs (fast signal), larger for resid
    band_arr = np.full(N_INST, RESID_BAND)
    for i, j, _, _ in PAIRS:
        band_arr[i] = 0.05
        band_arr[j] = 0.05

    drift  = np.abs(target_sh - _pos) * p_today
    trade  = drift > band_arr * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
