"""
MTAlpha Algothon strategy: 11 structural pairs + factor-residual reversion.

COMPONENT 1 - 11 COINTEGRATED PAIRS
  All pairs confirmed with ratio ADF p<0.01 AND positive out-of-sample
  Sharpe in both segments B[500-650] AND C[650-750] independently.
  Discovered via: ADF scan, SSD distance, Engle-Granger cointegration.
  Each pair trades the log price ratio as a z-score, sized to caps.

  Pairs (ticker_i, ticker_j, z_window, band):
    CTGI/EELT  (25,37) zw=40 band=0.5  - half-life 7.1d
    RTTH/NAYO  (18,35) zw=40 band=0.75 - half-life 7.9d
    HETT/ULXY  ( 7,40) zw=30 band=0.5  - half-life 5.4d
    EORC/NGTE  (13,45) zw=40 band=0.75 - half-life 5.4d
    MHRM/EAFC  (49,50) zw=40 band=0.3  - half-life SSD match
    SMAH/ILVX  (10,46) zw=30 band=0.75 - half-life SSD match
    HUXZ/ACAC  ( 8,27) zw=30 band=0.3  - half-life 5.2d
    AENO/NWIG  ( 1,20) zw=20 band=1.0  - half-life 4.5d
    ACIX/ITPA  (31,43) zw=20 band=0.3  - half-life 8.8d
    FWWG/BLBT  (36,41) zw=30 band=0.3  - half-life SSD match
    MTNS/BENI  (33,42) zw=40 band=1.0  - half-life 11.0d

COMPONENT 2 - FACTOR-RESIDUAL CROSS-SECTIONAL REVERSION
  Strip top-2 PCA factors from 3-day vol-normalised returns,
  trade the cross-sectional z-score across all 51 assets.
  Adds uncorrelated alpha on assets not consumed by pairs.
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# ---- Structural pairs: (i, j, z_window, band) ----
PAIRS = [
    (25, 37, 40, 0.50),   # CTGI/EELT
    (18, 35, 40, 0.75),   # RTTH/NAYO
    ( 7, 40, 30, 0.50),   # HETT/ULXY
    (13, 45, 40, 0.75),   # EORC/NGTE
    (49, 50, 40, 0.30),   # MHRM/EAFC
    (10, 46, 30, 0.75),   # SMAH/ILVX
    ( 8, 27, 30, 0.30),   # HUXZ/ACAC
    ( 1, 20, 20, 1.00),   # AENO/NWIG
    (31, 43, 20, 0.30),   # ACIX/ITPA
    (36, 41, 30, 0.30),   # FWWG/BLBT
    (33, 42, 40, 1.00),   # MTNS/BENI
]
PAIR_AGGR = 2.0

# ---- Factor-residual signal ----
NF          = 2
H           = 3
FITW        = 250
REFIT_EVERY = 20
VOL_WIN     = 60
AGGR        = 2.0
RESID_BAND  = 0.15

MIN_HISTORY = max(FITW, max(zw for _,_,zw,_ in PAIRS)) + 10

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

    if n_days <= _last_n:
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
    # Component 1: structural pairs
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
        target_dollars[i] -= f * CAPS[i]
        target_dollars[j] += f * CAPS[j]

    # =========================================================
    # Component 2: factor-residual reversion
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
    # Clip and execute
    # =========================================================
    target_dollars = np.clip(target_dollars, -CAPS, CAPS)
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    # Tighter band for pairs (fast signal), wider for resid
    band_arr = np.full(N_INST, RESID_BAND)
    pair_assets = set(i for i,j,_,_ in PAIRS) | set(j for i,j,_,_ in PAIRS)
    for idx in pair_assets:
        band_arr[idx] = 0.05

    drift   = np.abs(target_sh - _pos) * p_today
    trade   = drift > band_arr * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
