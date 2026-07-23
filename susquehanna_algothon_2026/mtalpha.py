"""
ALGO price-level spread trader.

ALGO is an index of the other 50 assets (OOS R^2 = 0.97).
We estimate replication weights by rolling OLS of ALGO returns on
the basket, build the log-price-level spread:

    spread_t = log(P_algo_t) - w @ log(P_basket_t)

Then z-score the spread over a rolling window and only enter a
position when the spread is in the EXTREME tails -- i.e. the
absolute spread z-score is above the 75th percentile of its own
recent rolling distribution. Trade ALGO only (no basket hedge).

Rename to <yourteamname>.py to submit.
"""

import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

WFIT      = 150    # rolling OLS window for weights (return space)
REFIT     = 5      # days between weight refits
ZW        = 60     # window for spread z-score
PCTILE    = 75     # only trade when |z| > this percentile of recent |z|s
AGGR      = 2.0    # tanh steepness
MIN_HISTORY = WFIT + ZW + 5

_pos     = np.zeros(N_INST, dtype=int)
_last_n  = 0
_w       = None
_fit_day = -10**9


def _ols(X, y, ridge=1e-6):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def getMyPosition(prcSoFar):
    global _pos, _last_n, _w, _fit_day

    prc     = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh  = np.floor(CAPS / p_today).astype(int)

    if n_days <= _last_n:           # fresh run in same process
        _pos     = np.zeros(N_INST, dtype=int)
        _w       = None
        _fit_day = -10**9
    _last_n = n_days

    if n_days < MIN_HISTORY:
        _pos = np.clip(_pos, -max_sh, max_sh)
        return _pos.copy()

    rets = np.diff(np.log(prc), axis=1)

    # refit replication weights from returns
    if _w is None or (n_days - _fit_day) >= REFIT:
        X    = rets[1:, -WFIT:].T
        y    = rets[0,  -WFIT:]
        _w   = _ols(X, y)
        _fit_day = n_days

    # price-level spread over rolling ZW window
    log_p  = np.log(prc)
    spread = log_p[0, -ZW:] - (_w @ log_p[1:, -ZW:])

    # demean and z-score the spread within the window
    s_mean = spread[:-1].mean()
    s_sd   = spread[:-1].std()
    if s_sd < 1e-12:
        _pos = np.zeros(N_INST, dtype=int)
        return _pos.copy()

    z_series = (spread[:-1] - s_mean) / s_sd   # z-scores of recent spread
    z_now    = (spread[-1]  - s_mean) / s_sd    # today's z-score

    # 75th percentile of |z| over the window -> threshold
    threshold = np.percentile(np.abs(z_series), PCTILE)

    if np.abs(z_now) > threshold:
        # ALGO rich (z_now > 0) -> short ALGO; cheap -> long ALGO
        algo_shares = int(np.round(
            -np.tanh(AGGR * z_now) * CAPS[0] / p_today[0]
        ))
        algo_shares = int(np.clip(algo_shares, -max_sh[0], max_sh[0]))
    else:
        algo_shares = 0

    new_pos    = np.zeros(N_INST, dtype=int)
    new_pos[0] = algo_shares

    _pos = new_pos
    return new_pos.copy()
    
