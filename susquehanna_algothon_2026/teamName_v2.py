"""
Algothon strategy: factor-residual reversion + adaptive market lead-lag.

Component 1 - FACTOR-RESIDUAL CROSS-SECTIONAL REVERSION (stable core)
  Fit top-NF PCA factors on trailing FITW days of returns (refit every
  20 days). Compute each asset's trailing H-day return residual to the
  factors; short the relative winners, long the relative losers.
  Diagnostics showed this positive in every 100-day window of the data.

Component 2 - ADAPTIVE MARKET LEAD-LAG (regime-aware overlay)
  The equal-weight market's own AR(1) drifted from negative (reversal
  regime) to positive (momentum regime) across the dataset. Rather than
  hardcode a direction, measure trailing AR(1) and trade yesterday's
  market move in the measured direction, scaled by its significance.
  Gated off when insignificant. Self-corrects if the regime flips.

Positions: expo in [-1,1] per asset -> dollars = expo * cap -> integer
shares, with a rebalance band to control commissions, self-clipped to
each day's cap.
"""

import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# ---- Component 1: factor-residual reversion ----
NF = 2            # number of PCA factors to strip
H = 3             # reversion horizon (days)
FITW = 250        # factor fit window
REFIT_EVERY = 20  # days between factor refits
W_RESID = 0.75    # weight of component in [-1,1] exposure

# ---- Component 2: adaptive market lead-lag ----
AR_WINDOW = 150   # trailing window for market AR(1) estimate
AR_TMIN = 1.0     # |t-stat| gate before the overlay activates
W_MKT = 0.6       # weight of component

# ---- Execution ----
REBAL_BAND = 0.08  # min |target-current| as fraction of cap to trade
MIN_HISTORY = FITW + 10

_pos = np.zeros(N_INST, dtype=int)
_last_n = 0
_V = None
_last_fit = -10**9


def _zscore_xs(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def getMyPosition(prcSoFar):
    global _pos, _last_n, _V, _last_fit

    prc = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh = np.floor(CAPS / p_today).astype(int)

    if n_days <= _last_n:          # fresh backtest in same process
        _pos = np.zeros(N_INST, dtype=int)
        _V = None
        _last_fit = -10**9
    _last_n = n_days

    if n_days < MIN_HISTORY:
        _pos = np.clip(_pos, -max_sh, max_sh)
        return _pos.copy()

    rets = np.diff(np.log(prc), axis=1)

    # ---- Component 1: factor-residual reversion ----
    if _V is None or (n_days - _last_fit) >= REFIT_EVERY:
        R = rets[:, -FITW:]
        Rc = R - R.mean(axis=1, keepdims=True)
        _, V = np.linalg.eigh(np.cov(Rc))
        _V = V[:, -NF:]
        _last_fit = n_days
    rh = rets[:, -H:].sum(axis=1)
    resid = rh - _V @ (_V.T @ rh)
    expo_resid = np.tanh(-_zscore_xs(resid))

    # ---- Component 2: adaptive market lead-lag ----
    mkt = rets.mean(axis=0)
    w = mkt[-AR_WINDOW:]
    ac = np.corrcoef(w[:-1], w[1:])[0, 1]
    t_ac = ac * np.sqrt(AR_WINDOW)
    if np.isfinite(t_ac) and abs(t_ac) > AR_TMIN:
        mvol = max(w.std(), 1e-8)
        s = np.tanh(np.sign(ac) * min(abs(t_ac) / 2.0, 1.5) *
                    mkt[-1] / mvol)
        expo_mkt = np.full(n_inst, s)
    else:
        expo_mkt = np.zeros(n_inst)

    # ---- Combine, size, execute ----
    expo = np.clip(W_RESID * expo_resid + W_MKT * expo_mkt, -1, 1)
    target_sh = np.round(expo * CAPS / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    drift = np.abs(target_sh - _pos) * p_today
    trade = drift > REBAL_BAND * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
