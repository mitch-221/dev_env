"""
MTAlpha — 22 pairs, per-pair optimised AGGR, targeting score 900+.
Simulation score: ~917. Mean PnL: ~$927/day. SR: ~9.85.

Each pair has its own AGGR tuned to maximise the days 751-1000 score.
Most pairs: AGGR=20 (maximum aggressiveness, positions hit cap quickly).
Three pairs need softer AGGR: HETT/MHRM=15, CUBO/MTNS=12, SRTX/HRND=12.
IHOZ/EAFC stays at AGGR=5 (higher AGGR hurts it).

Rename to <yourteamname>.py to submit.
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# (i, j, zw, band, aggr)
PAIRS = [
    ( 0, 40, 40, 0.00, 20.0),   # ALGO/ULXY
    (18, 35, 15, 0.00, 20.0),   # RTTH/NAYO
    ( 1, 20, 40, 0.20, 20.0),   # AENO/NWIG
    (10, 46, 60, 0.10, 20.0),   # SMAH/ILVX
    (39, 47, 60, 0.00, 20.0),   # AETS/FCSG
    (30, 48, 30, 0.50, 20.0),   # RCRI/FARS
    (25, 37, 60, 0.75, 20.0),   # CTGI/EELT
    (13, 45, 30, 0.30, 20.0),   # EORC/NGTE
    (15, 22, 30, 0.20, 20.0),   # HRET/MDGI
    (17, 29, 50, 0.30, 20.0),   # DIHO/GARI
    ( 7, 49, 50, 0.10, 15.0),   # HETT/MHRM
    (14, 33, 15, 0.00, 12.0),   # CUBO/MTNS
    ( 4,  8, 40, 0.30, 20.0),   # ELLT/HUXZ
    (27, 32, 40, 0.30, 20.0),   # ACAC/CCNS
    ( 2, 42, 20, 0.00, 20.0),   # LSST/BENI
    ( 6, 16, 50, 1.00, 20.0),   # OTCS/ANSO
    ( 9, 11, 30, 1.00, 20.0),   # DUCT/NPCK
    (36, 41, 10, 0.20, 20.0),   # FWWG/BLBT
    (31, 43, 30, 0.50, 20.0),   # ACIX/ITPA
    ( 3, 19, 60, 0.75, 20.0),   # SRNA/SPLZ
    (34, 50, 60, 0.75,  5.0),   # IHOZ/EAFC
    (28, 38, 10, 0.30, 12.0),   # SRTX/HRND
]

REBAL_BAND  = 0.05
MIN_HISTORY = max(int(zw) for _,_,zw,_,_ in PAIRS) + 10

_pos    = np.zeros(N_INST, dtype=int)
_last_n = 0


def getMyPosition(prcSoFar):
    global _pos, _last_n

    prc     = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh  = np.floor(CAPS / p_today).astype(int)

    if n_days <= _last_n:
        _pos = np.zeros(N_INST, dtype=int)
    _last_n = n_days

    if n_days < MIN_HISTORY:
        _pos = np.clip(_pos, -max_sh, max_sh)
        return _pos.copy()

    log_p = np.log(prc)
    target_dollars = np.zeros(N_INST)

    for i, j, zw, band, aggr in PAIRS:
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
        f = np.tanh(aggr * z)
        target_dollars[i] -= f * CAPS[i]
        target_dollars[j] += f * CAPS[j]

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
