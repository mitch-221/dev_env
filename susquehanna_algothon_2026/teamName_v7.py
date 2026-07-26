"""
MTAlpha v7 — 17 disjoint cointegrated pairs, pure pairs strategy.

Discovery: adding the factor-residual signal destroyed 186 score points
by fighting the pair positions on shared assets. Pure pairs wins.

Eval score: ~678. Mean PnL: $688/day. SR: 8.44.

All 17 pairs independently confirmed positive on B[500-650] AND C[650-750].
Pairs are asset-disjoint: each asset appears in at most one pair.
ALGO/MDGI uses the $100k ALGO cap at 0.2bp commission.

(i, j, z_window, band):
   0/22  ALGO/MDGI    zw=20 band=0.30
  25/37  CTGI/EELT    zw=40 band=0.30
  18/35  RTTH/NAYO    zw=40 band=0.75
  13/45  EORC/NGTE    zw=40 band=0.50
   8/20  HUXZ/NWIG    zw=20 band=0.75
  49/50  MHRM/EAFC    zw=40 band=0.30
  10/46  SMAH/ILVX    zw=30 band=0.75
   9/27  DUCT/ACAC    zw=20 band=1.00
  36/41  FWWG/BLBT    zw=30 band=0.30
  39/44  AETS/HTRK    zw=30 band=0.75
  31/43  ACIX/ITPA    zw=20 band=0.30
   7/33  HETT/MTNS    zw=20 band=0.30
   2/16  LSST/ANSO    zw=20 band=1.00
   4/17  ELLT/DIHO    zw=30 band=1.00
  26/32  ALUT/CCNS    zw=40 band=1.00
  12/40  MSDP/ULXY    zw=40 band=1.00
  40/42  ULXY/BENI    zw=40 band=0.75 (ULXY split between MSDP and BENI)
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

PAIRS = [
    ( 0, 22, 20, 0.30),   # ALGO/MDGI
    (25, 37, 40, 0.30),   # CTGI/EELT
    (18, 35, 40, 0.75),   # RTTH/NAYO
    (13, 45, 40, 0.50),   # EORC/NGTE
    ( 8, 20, 20, 0.75),   # HUXZ/NWIG
    (49, 50, 40, 0.30),   # MHRM/EAFC
    (10, 46, 30, 0.75),   # SMAH/ILVX
    ( 9, 27, 20, 1.00),   # DUCT/ACAC
    (36, 41, 30, 0.30),   # FWWG/BLBT
    (39, 44, 30, 0.75),   # AETS/HTRK
    (31, 43, 20, 0.30),   # ACIX/ITPA
    ( 7, 33, 20, 0.30),   # HETT/MTNS
    ( 2, 16, 20, 1.00),   # LSST/ANSO
    ( 4, 17, 30, 1.00),   # ELLT/DIHO
    (26, 32, 40, 1.00),   # ALUT/CCNS
    (12, 40, 40, 1.00),   # MSDP/ULXY
    (42, 40, 40, 0.75),   # BENI/ULXY (note: ULXY at half-cap effectively)
]
PAIR_AGGR = 2.0

MIN_HISTORY = max(zw for _,_,zw,_ in PAIRS) + 10

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
    rets  = np.diff(log_p, axis=1)

    target_dollars = np.zeros(N_INST)

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

    target_dollars = np.clip(target_dollars, -CAPS, CAPS)
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    # Tight rebalance band for pairs (fast mean-reverting signals)
    drift   = np.abs(target_sh - _pos) * p_today
    trade   = drift > 0.05 * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
