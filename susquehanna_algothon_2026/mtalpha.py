"""
Algothon submission strategy: factor-residual cross-sectional reversion.

Signal: each asset's trailing 3-day return, with the top-2 PCA factors
(fit on trailing 250d, refit every 20d) projected out and the residual
normalized by the asset's own volatility. Assets whose residual is
high relative to the cross-section are shorted; low are bought.
Exposure = tanh(-2 * z) of the cross-sectional z-score, sized to the
per-asset dollar caps, executed as integer shares with a 15%-of-cap
rebalance band and daily self-clipping to that day's cap ceiling.

Validated via walk-forward segment testing with noise benchmarks;
alternative structures examined and rejected on evidence: pairwise
cointegration (indistinguishable from noise), per-asset drift (absent),
seasonality (absent), ALGO index arbitrage (relationship real,
OOS R^2 0.97, but the spread is a random walk - no convergence).
"""

import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

NF = 2             # PCA factors stripped
H = 3              # reversion horizon (days)
FITW = 250         # factor fit window
REFIT_EVERY = 20   # days between factor refits
VOL_WIN = 60       # per-asset vol window for normalization
AGGR = 2.0         # signal -> exposure steepness
REBAL_BAND = 0.15  # trade only if target moves > 15% of cap
MIN_HISTORY = FITW + 10

_pos = np.zeros(N_INST, dtype=int)
_last_n = 0
_V = None
_fit_day = -10**9


def _zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def getMyPosition(prcSoFar):
    global _pos, _last_n, _V, _fit_day

    prc = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh = np.floor(CAPS / p_today).astype(int)

    if n_days <= _last_n:           # history restarted: reset state
        _pos = np.zeros(N_INST, dtype=int)
        _V = None
        _fit_day = -10**9
    _last_n = n_days

    if n_days < MIN_HISTORY:
        _pos = np.clip(_pos, -max_sh, max_sh)
        return _pos.copy()

    rets = np.diff(np.log(prc), axis=1)

    if _V is None or (n_days - _fit_day) >= REFIT_EVERY:
        R = rets[:, -FITW:]
        Rc = R - R.mean(axis=1, keepdims=True)
        _, V = np.linalg.eigh(np.cov(Rc))
        _V = V[:, -NF:]
        _fit_day = n_days

    rh = rets[:, -H:].sum(axis=1)
    resid = rh - _V @ (_V.T @ rh)
    vol = np.maximum(rets[:, -VOL_WIN:].std(axis=1), 1e-8)
    resid = resid / (vol * np.sqrt(H))

    expo = np.tanh(-AGGR * _zscore(resid))
    target = np.round(expo * CAPS / p_today).astype(int)
    target = np.clip(target, -max_sh, max_sh)

    drift = np.abs(target - _pos) * p_today
    trade = drift > REBAL_BAND * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
