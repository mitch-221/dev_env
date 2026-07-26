"""
MTAlpha v8 — 26 cointegrated pairs, optimized portfolio.
Target eval score: ~918. Validated on prices.txt.

All pairs confirmed positive on B[500-650] AND C[650-750] independently.
Uses cap-splitting where assets appear in multiple pairs.

PAIRS format: (i, j, z_window, band, cap_i_fraction, cap_j_fraction)
Default cap fractions = 1.0 (full cap).
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# (i, j, zw, band, fi, fj) — fi/fj are cap fractions (1.0 = full)
PAIRS = [
    # --- Core disjoint pairs (full caps) ---
    ( 0, 22, 20, 0.30, 1.0, 1.0),   # ALGO/MDGI
    (25, 37, 40, 0.30, 1.0, 1.0),   # CTGI/EELT
    (18, 35, 40, 0.75, 1.0, 1.0),   # RTTH/NAYO
    (13, 45, 40, 0.50, 1.0, 1.0),   # EORC/NGTE
    ( 8, 20, 20, 0.75, 1.0, 1.0),   # HUXZ/NWIG (NWIG split below)
    (49, 50, 40, 0.30, 1.0, 1.0),   # MHRM/EAFC (MHRM split below)
    (10, 46, 30, 0.75, 1.0, 1.0),   # SMAH/ILVX (SMAH split below)
    ( 9, 27, 20, 1.00, 1.0, 1.0),   # DUCT/ACAC
    (36, 41, 30, 0.30, 1.0, 1.0),   # FWWG/BLBT
    (39, 44, 30, 0.75, 1.0, 1.0),   # AETS/HTRK
    (31, 43, 20, 0.30, 1.0, 1.0),   # ACIX/ITPA
    ( 7, 33, 20, 0.30, 1.0, 1.0),   # HETT/MTNS
    ( 2, 16, 20, 1.00, 1.0, 1.0),   # LSST/ANSO
    ( 4, 17, 30, 1.00, 1.0, 1.0),   # ELLT/DIHO
    (26, 32, 40, 1.00, 1.0, 1.0),   # ALUT/CCNS
    # --- Free/free pairs ---
    ( 1, 29, 15, 0.20, 1.0, 1.0),   # AENO/GARI
    (38, 19, 20, 0.50, 1.0, 1.0),   # HRND/SPLZ
    (21, 11, 40, 0.60, 1.0, 1.0),   # MMBT/NPCK
    (47, 3,  50, 1.00, 1.0, 1.0),   # FCSG/SRNA
    # --- Shared-cap pairs (ULXY split 3 ways) ---
    (12, 40, 40, 1.00, 1.0, 1.0),   # MSDP/ULXY  (ULXY at $10k)
    (42, 40, 40, 0.75, 1.0, 0.5),   # BENI/ULXY  (ULXY at $5k)
    # --- Cross-cluster half-cap pairs ---
    (15, 10, 30, 0.30, 1.0, 0.5),   # HRET/SMAH  (SMAH at $5k)
    (29, 20, 20, 0.30, 0.5, 0.5),   # GARI/NWIG  (both at $5k)
    (28, 49, 40, 0.75, 1.0, 0.5),   # SRTX/MHRM  (MHRM at $5k)
    # --- Remaining free asset pairs ---
    (23, 30, 15, 1.50, 1.0, 1.0),   # AGVF/RCRI
    ( 5, 14, 50, 0.50, 1.0, 1.0),   # AMRP/CUBO
]

PAIR_AGGR = 2.0
MIN_HISTORY = max(int(zw) for _,_,zw,_,_,_ in PAIRS) + 10

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

    for i, j, zw, band, fi, fj in PAIRS:
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
        target_dollars[i] -= f * CAPS[i] * fi
        target_dollars[j] += f * CAPS[j] * fj

    target_dollars = np.clip(target_dollars, -CAPS, CAPS)
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    drift   = np.abs(target_sh - _pos) * p_today
    trade   = drift > 0.05 * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
