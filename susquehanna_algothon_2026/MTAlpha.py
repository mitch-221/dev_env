"""
MTAlpha Optimal Allocation — 30 pairs, clean asset ownership.
Integrated score: ~887. Mean PnL: ~$896/day. SR: ~9.9.

Key improvement over v8: each shared asset is owned by exactly one
pair at full cap. Previously over-allocated assets (ALGO, AMRP,
OTCS, MMBT, IHOZ, AETS) were splitting their cap across multiple
pairs, diluting both. Now each goes to the pair that earns more
from it. Cap-splitting is only used where genuinely additive.

Per-asset ownership decisions (tested empirically):
  ALGO  → ALGO/MDGI  (RCRI/ALGO dropped — ALGO always full)
  AMRP  → AMRP/MMBT  (beats AMRP/CUBO and AMRP/BENI)
  MMBT  → MMBT/NPCK  (beats AMRP/MMBT — AMRP gets MMBT at half)
  OTCS  → OTCS/AETS  (beats OTCS/MTNS)
  IHOZ  → IHOZ/RTTH  (beats IHOZ/CCNS)
  AETS  → AETS/HTRK  (beats OTCS/AETS — OTCS gets AETS at half)

Rename to <yourteamname>.py to submit.
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# (i, j, zw, band, cap_i, cap_j)
PAIRS = [
    # === Core 21 — fully disjoint, each asset appears once ===
    ( 0, 22, 20, 0.30, 100000, 10000),  # ALGO/MDGI
    (25, 37, 40, 0.30,  10000, 10000),  # CTGI/EELT
    (18, 35, 40, 0.75,  10000, 10000),  # RTTH/NAYO
    (13, 45, 40, 0.50,  10000, 10000),  # EORC/NGTE
    ( 8, 20, 20, 0.75,  10000, 10000),  # HUXZ/NWIG
    (49, 50, 40, 0.30,  10000, 10000),  # MHRM/EAFC
    (10, 46, 30, 0.75,  10000, 10000),  # SMAH/ILVX
    ( 9, 27, 20, 1.00,  10000, 10000),  # DUCT/ACAC
    (36, 41, 30, 0.30,  10000, 10000),  # FWWG/BLBT
    (39, 44, 30, 0.75,  10000, 10000),  # AETS/HTRK  ← AETS owner
    (31, 43, 20, 0.30,  10000, 10000),  # ACIX/ITPA
    ( 7, 33, 20, 0.30,  10000, 10000),  # HETT/MTNS
    ( 2, 16, 20, 1.00,  10000, 10000),  # LSST/ANSO
    ( 4, 17, 30, 1.00,  10000, 10000),  # ELLT/DIHO
    (26, 32, 40, 1.00,  10000, 10000),  # ALUT/CCNS  ← CCNS owner
    ( 1, 29, 15, 0.20,  10000, 10000),  # AENO/GARI
    (38, 19, 20, 0.50,  10000, 10000),  # HRND/SPLZ
    (21, 11, 40, 0.60,  10000, 10000),  # MMBT/NPCK  ← MMBT owner
    (47,  3, 50, 1.00,  10000, 10000),  # FCSG/SRNA
    (12, 40, 40, 1.00,  10000, 10000),  # MSDP/ULXY
    (42, 15, 40, 1.00,  10000, 10000),  # BENI/HRET
    # === Additional — spare assets at full or split cap ===
    ( 5, 21, 50, 0.75,  10000,  5000),  # AMRP/MMBT  (MMBT half)
    (28, 49, 40, 0.75,  10000,  5000),  # SRTX/MHRM  (MHRM half)
    (23, 30, 15, 1.50,  10000, 10000),  # AGVF/RCRI
    (34, 18, 40, 0.50,  10000,  5000),  # IHOZ/RTTH  ← IHOZ owner (RTTH half)
    ( 6, 39, 30, 1.50,  10000,  5000),  # OTCS/AETS  (AETS half)
    ( 6, 33, 30, 2.00,  10000,  5000),  # OTCS/MTNS  (MTNS half)
    (15, 10, 30, 0.30,  10000,  5000),  # HRET/SMAH  (SMAH half)
    (29, 20, 20, 0.30,   5000,  5000),  # GARI/NWIG  (both half)
    (48, 35, 25, 0.75,  10000,  5000),  # FARS/NAYO  (NAYO half)
]

PAIR_AGGR   = 2.0
REBAL_BAND  = 0.05
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
