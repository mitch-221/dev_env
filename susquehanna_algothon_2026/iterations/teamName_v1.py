"""
Strategy v4: ALGO return-residual fade + structural cointegrated pairs.

COMPONENT 1 - ALGO RETURN-RESIDUAL FADE
  The ALGO return residual (ALGO return minus fitted basket return)
  has Hurst=0.011 - almost perfectly mean-reverting. The reversion
  is same-day (Hurst near 0 means the series overshoots and snaps
  back within one period). We fade yesterday's residual: if ALGO
  returned more than the basket yesterday, short ALGO today.
  Rolling OLS weights (150d window, refit every 5d).

COMPONENT 2 - STRUCTURAL COINTEGRATED PAIRS
  CTGI/EELT and MTNS/BENI both show p=0.000 on cointegration AND
  ratio stationarity - designed relationships. Trade the log price
  ratio as a z-score, fading deviations from the rolling mean.
  Both legs sized to $10k cap.

Only imports numpy. Rename to <yourteamname>.py to submit.
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM   = np.full(N_INST, 0.0001)
COMM[0] = 0.00002

# ---- Component 1: ALGO return-residual fade ----
ARB_WFIT   = 150     # OLS window
ARB_REFIT  = 5       # refit every N days
ARB_SCALE  = 100_000.0
ARB_AGGR   = 2.0

# ---- Component 2: cointegrated pairs ----
# (i, j) - trade log(P_i / P_j) as a spread
# identified from range_analysis.py: p=0.000 on both tests
PAIRS = [
    (15, 37),   # HRET / EELT  -- check tickers match your index
    (25, 33),   # CTGI / MTNS  -- update if indices differ
]
# Override with the actual 0-based indices from your prices.txt header:
# CTGI is index 25 (0=ALGO,1=AENO,...), EELT=36, MTNS=32, BENI=41
# Reconfirm by counting: ALGO(0) AENO(1) LSST(2) SRNA(3) ELLT(4) AMRP(5)
# OTCS(6) HETT(7) HUXZ(8) DUCT(9) SMAH(10) NPCK(11) MSDP(12) EORC(13)
# CUBO(14) HRET(15) ANSO(16) DIHO(17) RTTH(18) SPLZ(19) NWIG(20) MMBT(21)
# MDGI(22) AGVF(23) RRES(24) CTGI(25) ALUT(26) ACAC(27) SRTX(28) GARI(29)
# RCRI(30) ACIX(31) CCNS(32) MTNS(33) IHOZ(34) NAYO(35) FWWG(36) EELT(37)
# HRND(38) AETS(39) ULXY(40) BLBT(41) BENI(42) ITPA(43) HTRK(44) NGTE(45)
# ILVX(46) FCSG(47) FARS(48) MHRM(49) EAFC(50)
PAIRS = [
    (25, 37),   # CTGI / EELT
    (33, 42),   # MTNS / BENI
]

PAIR_ZW   = 30      # z-score window for pair spread
PAIR_AGGR = 2.0     # tanh steepness for pair positions
PAIR_BAND = 0.5     # only trade when |z| > this

MIN_HISTORY = ARB_WFIT + PAIR_ZW + 5

_pos      = np.zeros(N_INST, dtype=int)
_last_n   = 0
_w        = None
_fit_day  = -10**9


def _ols(X, y, ridge=1e-6):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def getMyPosition(prcSoFar):
    global _pos, _last_n, _w, _fit_day

    prc     = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh  = np.floor(CAPS / p_today).astype(int)

    if n_days <= _last_n:
        _pos     = np.zeros(N_INST, dtype=int)
        _w       = None
        _fit_day = -10**9
    _last_n = n_days

    if n_days < MIN_HISTORY:
        _pos = np.clip(_pos, -max_sh, max_sh)
        return _pos.copy()

    rets   = np.diff(np.log(prc), axis=1)
    log_p  = np.log(prc)

    target_dollars = np.zeros(N_INST)

    # =========================================================
    # Component 1: fade yesterday's ALGO return residual
    # =========================================================
    if _w is None or (n_days - _fit_day) >= ARB_REFIT:
        X    = rets[1:, -ARB_WFIT:].T
        y    = rets[0,  -ARB_WFIT:]
        _w   = _ols(X, y)
        _fit_day = n_days

    # yesterday's residual = ALGO return[t-1] - basket return[t-1]
    r_yesterday = rets[0, -1] - _w @ rets[1:, -1]

    # normalise by rolling vol of residuals
    resid_win = rets[0, -ARB_WFIT:] - (_w @ rets[1:, -ARB_WFIT:])
    r_sd = resid_win.std()
    if r_sd > 1e-12:
        z_algo = r_yesterday / r_sd
        # fade: ALGO outperformed yesterday -> short today
        target_dollars[0] = -np.tanh(ARB_AGGR * z_algo) * ARB_SCALE

    # =========================================================
    # Component 2: cointegrated pairs (CTGI/EELT, MTNS/BENI)
    # =========================================================
    for i, j in PAIRS:
        if max(i, j) >= n_inst:
            continue
        spread_win = log_p[i, -PAIR_ZW:] - log_p[j, -PAIR_ZW:]
        s_mean = spread_win[:-1].mean()
        s_sd   = spread_win[:-1].std()
        if s_sd < 1e-12:
            continue
        z = (spread_win[-1] - s_mean) / s_sd
        if abs(z) <= PAIR_BAND:
            continue
        # z > 0: asset i is relatively expensive -> short i, long j
        f = np.tanh(PAIR_AGGR * z)
        target_dollars[i] -= f * CAPS[i]
        target_dollars[j] += f * CAPS[j]

    # =========================================================
    # Convert to integer shares, apply caps, rebalance band
    # =========================================================
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    # rebalance band: 5% of cap for ALGO (fast signal), 15% for pairs
    band = np.full(N_INST, 0.15)
    band[0] = 0.05
    drift  = np.abs(target_sh - _pos) * p_today
    trade  = drift > band * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
