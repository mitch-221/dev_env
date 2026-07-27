"""
MTAlpha Maximum — V8 base + all confirmed positive additions.
Simulation score: ~1112. Mean PnL: ~$1125/day.

Structure:
  V8 base: 26 pairs (core disjoint + cap-splitting)
  + RCRI/ALGO at $50k ALGO cap
  + SRTX/MHRM at $5k MHRM cap
  + IHOZ/CCNS at $5k CCNS cap
  + IHOZ/RTTH at $5k RTTH cap
  + OTCS/AETS at full caps (both free)
  + OTCS/MTNS at $10k MTNS (MTNS in HETT/MTNS — split)
  + AMRP/MMBT at full caps (both free)
  + AMRP/BENI at $10k (BENI in BENI/ULXY — split)
  + FARS basket vs CUBO+IHOZ

Rename to <yourteamname>.py to submit.
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# (i, j, zw, band, cap_i, cap_j)
PAIRS = [
    # V8 core disjoint
    ( 0, 22, 20, 0.30, 100000, 10000),  # ALGO/MDGI
    (25, 37, 40, 0.30,  10000, 10000),  # CTGI/EELT
    (18, 35, 40, 0.75,  10000, 10000),  # RTTH/NAYO
    (13, 45, 40, 0.50,  10000, 10000),  # EORC/NGTE
    ( 8, 20, 20, 0.75,  10000, 10000),  # HUXZ/NWIG
    (49, 50, 40, 0.30,  10000, 10000),  # MHRM/EAFC
    (10, 46, 30, 0.75,  10000, 10000),  # SMAH/ILVX
    ( 9, 27, 20, 1.00,  10000, 10000),  # DUCT/ACAC
    (36, 41, 30, 0.30,  10000, 10000),  # FWWG/BLBT
    (39, 44, 30, 0.75,  10000, 10000),  # AETS/HTRK
    (31, 43, 20, 0.30,  10000, 10000),  # ACIX/ITPA
    ( 7, 33, 20, 0.30,  10000, 10000),  # HETT/MTNS
    ( 2, 16, 20, 1.00,  10000, 10000),  # LSST/ANSO
    ( 4, 17, 30, 1.00,  10000, 10000),  # ELLT/DIHO
    (26, 32, 40, 1.00,  10000, 10000),  # ALUT/CCNS
    # V8 free/free
    ( 1, 29, 15, 0.20,  10000, 10000),  # AENO/GARI
    (38, 19, 20, 0.50,  10000, 10000),  # HRND/SPLZ
    (21, 11, 40, 0.60,  10000, 10000),  # MMBT/NPCK
    (47,  3, 50, 1.00,  10000, 10000),  # FCSG/SRNA
    # V8 cap-split
    (12, 40, 40, 1.00,  10000, 10000),  # MSDP/ULXY (ULXY full)
    (42, 40, 40, 0.75,  10000,  5000),  # BENI/ULXY (ULXY half)
    (15, 10, 30, 0.30,  10000,  5000),  # HRET/SMAH (SMAH half)
    (29, 20, 20, 0.30,   5000,  5000),  # GARI/NWIG (both half)
    (28, 49, 40, 0.75,  10000,  5000),  # SRTX/MHRM_V8 (MHRM half)
    (23, 30, 15, 1.50,  10000, 10000),  # AGVF/RCRI
    ( 5, 14, 50, 0.50,  10000, 10000),  # AMRP/CUBO
    # New additions — all confirmed B&C positive
    (30,  0, 50, 0.75,  10000, 50000),  # RCRI/ALGO (ALGO at $50k)
    (28, 49, 60, 1.00,  10000,  5000),  # SRTX/MHRM_new (MHRM at $5k)
    (34, 32, 15, 0.20,  10000,  5000),  # IHOZ/CCNS (CCNS at $5k)
    (34, 18, 40, 0.50,  10000,  5000),  # IHOZ/RTTH (RTTH at $5k)
    ( 6, 39, 30, 1.50,  10000, 10000),  # OTCS/AETS
    ( 6, 33, 30, 2.00,  10000,  5000),  # OTCS/MTNS (MTNS half)
    ( 5, 21, 50, 0.75,  10000, 10000),  # AMRP/MMBT
    ( 5, 42, 60, 0.50,  10000,  5000),  # AMRP/BENI (BENI half)
]

PAIR_AGGR   = 2.0
REBAL_BAND  = 0.05

# Basket: FARS = f(CUBO + IHOZ)
BASKET_TARGET = 48  # FARS
BASKET_LEGS   = [14, 34]  # CUBO, IHOZ
BASKET_ZW     = 20
BASKET_BAND   = 0.30
BASKET_WFIT   = 150
BASKET_REFIT  = 20

MIN_HISTORY = max(
    max(int(zw) for _,_,zw,_,_,_ in PAIRS),
    BASKET_ZW + BASKET_WFIT
) + 10

_pos      = np.zeros(N_INST, dtype=int)
_last_n   = 0
_bw       = None   # basket OLS weights
_bfit_day = -10**9


def _ols(X, y, ridge=1e-6):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def getMyPosition(prcSoFar):
    global _pos, _last_n, _bw, _bfit_day

    prc     = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh  = np.floor(CAPS / p_today).astype(int)

    if n_days <= _last_n:
        _pos = np.zeros(N_INST, dtype=int)
        _bw  = None
        _bfit_day = -10**9
    _last_n = n_days

    if n_days < MIN_HISTORY:
        _pos = np.clip(_pos, -max_sh, max_sh)
        return _pos.copy()

    log_p = np.log(prc)
    target_dollars = np.zeros(N_INST)

    # ---- pairs ----
    for i, j, zw, band, ci, cj in PAIRS:
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
        target_dollars[i] -= f * ci
        target_dollars[j] += f * cj

    # ---- basket: FARS vs CUBO+IHOZ ----
    if n_days >= BASKET_ZW + BASKET_WFIT + 5:
        if _bw is None or (n_days - _bfit_day) >= BASKET_REFIT:
            X = np.column_stack(
                [log_p[li, -BASKET_WFIT:] for li in BASKET_LEGS] +
                [np.ones(BASKET_WFIT)]
            )
            y = log_p[BASKET_TARGET, -BASKET_WFIT:]
            _bw = _ols(X, y)
            _bfit_day = n_days

        x_now = np.array([log_p[li, -1] for li in BASKET_LEGS] + [1.])
        r_now = log_p[BASKET_TARGET, -1] - x_now @ _bw

        resid_win = np.array([
            log_p[BASKET_TARGET, -BASKET_ZW+k-1] -
            np.array([log_p[li, -BASKET_ZW+k-1] for li in BASKET_LEGS] + [1.]) @ _bw
            for k in range(BASKET_ZW)
        ])
        sm, ss = resid_win[:-1].mean(), resid_win[:-1].std()
        if ss > 1e-12:
            z = (r_now - sm) / ss
            if abs(z) > BASKET_BAND:
                f = np.tanh(PAIR_AGGR * z)
                target_dollars[BASKET_TARGET] -= f * CAPS[BASKET_TARGET]
                for k, li in enumerate(BASKET_LEGS):
                    target_dollars[li] += np.clip(
                        f * _bw[k] * CAPS[li], -CAPS[li], CAPS[li]
                    )

    # ---- execute ----
    target_dollars = np.clip(target_dollars, -CAPS, CAPS)
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    drift   = np.abs(target_sh - _pos) * p_today
    trade   = drift > REBAL_BAND * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
