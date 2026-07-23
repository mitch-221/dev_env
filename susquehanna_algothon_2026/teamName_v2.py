"""
Algothon strategy v3: ALGO index arbitrage + factor-residual reversion.

COMPONENT 1 - ALGO INDEX ARBITRAGE (primary)
  ALGO is (empirically, OOS R^2 ~0.97) a linear index of the other 50
  assets. We estimate replication weights by rolling OLS of ALGO
  returns on basket returns, form the spread
      s_t = log P_algo - sum_i w_i log P_i,
  z-score it over a short trailing window, and trade the spread:
  short ALGO / long basket when ALGO is rich, reverse when cheap.
  Market exposure cancels; PnL is the mispricing converging. Sized to
  ALGO's full $100k cap (0.2bp commission makes turnover ~free).

COMPONENT 2 - FACTOR-RESIDUAL REVERSION (basket assets)
  Cross-sectional 3-day reversion on the 50 constituents with top-NF
  PCA factors stripped and per-asset vol normalization. Runs in the
  cap headroom left after the arb's hedge legs, so the hedge is never
  distorted. ALGO itself is owned exclusively by the arb.

Execution: combined dollar targets -> integer shares -> daily
self-clipping to caps. Small rebalance band for cost control.
"""

import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# ---- Component 1: index arb ----
ARB_WFIT = 250        # rolling OLS window (returns) for weights
ARB_REFIT = 5         # days between weight refits
ARB_ZW = 5            # z-score window for the spread
ARB_SCALE = 100_000.0 # ALGO leg size at full signal
ARB_RIDGE = 1e-6

# ---- Component 2: factor-resid reversion ----
NF = 2
H = 3
FITW = 250
REFIT_EVERY = 20
VOL_NORM = True
VOL_WIN = 60
AGGR = 2.0
W_RESID = 1.0         # 0 disables the resid component

# ---- Execution ----
REBAL_BAND = 0.02     # fraction of cap; arb wants frequent updates
MIN_HISTORY = max(ARB_WFIT + ARB_ZW, FITW) + 10

_pos = np.zeros(N_INST, dtype=int)
_last_n = 0
_w = None
_w_fit_day = -10**9
_V = None
_V_fit_day = -10**9


def _zscore_xs(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def _ols(X, y):
    XtX = X.T @ X + ARB_RIDGE * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def getMyPosition(prcSoFar):
    global _pos, _last_n, _w, _w_fit_day, _V, _V_fit_day

    prc = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh = np.floor(CAPS / p_today).astype(int)

    if n_days <= _last_n:            # fresh run in same process
        _pos = np.zeros(N_INST, dtype=int)
        _w = None
        _V = None
        _w_fit_day = _V_fit_day = -10**9
    _last_n = n_days

    if n_days < MIN_HISTORY:
        _pos = np.clip(_pos, -max_sh, max_sh)
        return _pos.copy()

    log_p = np.log(prc)
    rets = np.diff(log_p, axis=1)

    # ================= Component 1: index arb =================
    if _w is None or (n_days - _w_fit_day) >= ARB_REFIT:
        X = rets[1:, -ARB_WFIT:].T
        y = rets[0, -ARB_WFIT:]
        _w = _ols(X, y)
        _w_fit_day = n_days

    spread = log_p[0, -(ARB_ZW + 1):] - _w @ log_p[1:, -(ARB_ZW + 1):]
    s_win, s_now = spread[:-1], spread[-1]
    s_sd = s_win.std()
    if s_sd > 1e-12:
        z = (s_now - s_win.mean()) / s_sd
        f = np.tanh(z)
    else:
        f = 0.0
    arb_dollars = np.zeros(n_inst)
    arb_dollars[0] = -f * ARB_SCALE
    arb_dollars[1:] = f * ARB_SCALE * _w
    arb_dollars = np.clip(arb_dollars, -CAPS, CAPS)

    # ============ Component 2: factor-resid reversion ============
    resid_dollars = np.zeros(n_inst)
    if W_RESID > 0:
        if _V is None or (n_days - _V_fit_day) >= REFIT_EVERY:
            R = rets[:, -FITW:]
            Rc = R - R.mean(axis=1, keepdims=True)
            _, V = np.linalg.eigh(np.cov(Rc))
            _V = V[:, -NF:]
            _V_fit_day = n_days
        rh = rets[:, -H:].sum(axis=1)
        resid = rh - _V @ (_V.T @ rh)
        if VOL_NORM:
            vol = np.maximum(rets[:, -VOL_WIN:].std(axis=1), 1e-8)
            resid = resid / (vol * np.sqrt(H))
        expo = np.tanh(-AGGR * _zscore_xs(resid)) * W_RESID
        # resid trades only the basket, in headroom left by the hedge;
        # ALGO belongs to the arb.
        headroom = np.maximum(CAPS - np.abs(arb_dollars), 0.0)
        headroom[0] = 0.0
        resid_dollars = expo * headroom

    # ================= Combine & execute =================
    target_dollars = arb_dollars + resid_dollars
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    drift = np.abs(target_sh - _pos) * p_today
    trade = drift > REBAL_BAND * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
