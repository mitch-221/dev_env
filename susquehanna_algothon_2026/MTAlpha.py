"""
MTAlpha Maximum — 22 core pairs + 4 half-cap split pairs = 26 total.
Simulation score: ~1035. Official eval target: 900+.

New additions (free asset vs used partner at half cap):
  RRES/RTTH:  RRES (free) + RTTH at $5k  — score +29
  MSDP/NWIG:  MSDP (free) + NWIG at $5k  — score +29
  AGVF/DUCT:  AGVF (free) + DUCT at $5k  — score +22
  HTRK/CUBO:  HTRK (free) + CUBO at $5k  — score +10

Note: cap fractions — used asset gets half ($5k), free asset gets full ($10k).
The combined exposure stays within caps through the clip in getMyPosition.

eval.py: set numTestDays = 250 to score days 751-1000.
Rename to <yourteamname>.py to submit.
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# (i, j, zw, band, aggr, cap_i, cap_j)
PAIRS = [
    # === Core 22 pairs (all at full cap) ===
    ( 0, 40, 40, 0.00,  20.0, 100000, 10000),  # ALGO/ULXY
    (18, 35, 15, 0.00, 100.0,  10000, 10000),  # RTTH/NAYO
    ( 1, 20, 40, 0.20,  20.0,  10000, 10000),  # AENO/NWIG
    (10, 46, 60, 0.10,  20.0,  10000, 10000),  # SMAH/ILVX
    (39, 47, 60, 0.00,  20.0,  10000, 10000),  # AETS/FCSG
    (30, 48, 30, 0.50,  20.0,  10000, 10000),  # RCRI/FARS
    (25, 37, 60, 0.75,  20.0,  10000, 10000),  # CTGI/EELT
    (13, 45, 30, 0.30,  20.0,  10000, 10000),  # EORC/NGTE
    (15, 22, 30, 0.20,  20.0,  10000, 10000),  # HRET/MDGI
    (17, 29, 50, 0.30,  20.0,  10000, 10000),  # DIHO/GARI
    ( 7, 49, 50, 0.10,  15.0,  10000, 10000),  # HETT/MHRM
    (14, 33, 15, 0.00,  12.0,  10000, 10000),  # CUBO/MTNS
    ( 4,  8, 40, 0.30,  20.0,  10000, 10000),  # ELLT/HUXZ
    (27, 32, 40, 0.30,  20.0,  10000, 10000),  # ACAC/CCNS
    ( 2, 42, 20, 0.00,  20.0,  10000, 10000),  # LSST/BENI
    ( 6, 16, 50, 1.00,  20.0,  10000, 10000),  # OTCS/ANSO
    (28, 38, 10, 0.30,  12.0,  10000, 10000),  # SRTX/HRND
    ( 9, 11, 30, 1.00,  20.0,  10000, 10000),  # DUCT/NPCK
    (36, 41, 60, 0.05,  20.0,  10000, 10000),  # FWWG/BLBT
    (31, 43, 25, 0.30,  50.0,  10000, 10000),  # ACIX/ITPA
    ( 3, 19, 60, 0.75,  25.0,  10000, 10000),  # SRNA/SPLZ
    (34, 50, 40, 0.40,  50.0,  10000, 10000),  # IHOZ/EAFC
    # === 4 half-cap split pairs (free asset full, used partner half) ===
    (24, 18,  8, 0.10,  30.0,  10000,  5000),  # RRES/RTTH(half)
    (12, 20, 40, 0.05,  30.0,  10000,  5000),  # MSDP/NWIG(half)
    (23,  9, 40, 0.00,  30.0,  10000,  5000),  # AGVF/DUCT(half)
    (44, 14, 30, 0.10,  30.0,  10000,  5000),  # HTRK/CUBO(half)
]

REBAL_BAND  = 0.00
MIN_HISTORY = max(int(zw) for _,_,zw,_,_,_,_ in PAIRS) + 10

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

    for i, j, zw, band, aggr, ci, cj in PAIRS:
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
        target_dollars[i] -= f * ci
        target_dollars[j] += f * cj

    target_dollars = np.clip(target_dollars, -CAPS, CAPS)
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    _pos = target_sh
    return _pos.copy()
