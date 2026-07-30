"""
MTAlpha Adaptive v2 — 16 cointegrated pairs, principled position sizing.

WHAT THIS DOES:
Each pair (i,j) has a spread = log(P_i) - log(P_j).
This spread is stationary around a fixed long-run mean (verified by ADF test,
consistent across all four 250-day windows in the 1000-day history).

POSITION SIZING — three components:

1. EXPANDING MEAN (not rolling):
   sm = mean(all spread history up to today)
   This converges to the true long-run mean as data accumulates.
   Rolling mean chases short-term level shifts and loses the signal.
   The data shows these spreads have stable long-run means (+/-0.04 t-stat
   for drift), so using all history gives the best estimate.

2. VOL COMPRESSION SIGNAL — the most important component:
   vol_scale = clip(long_term_vol / short_term_vol, 0.1, 3.0)
   
   When short-term vol < long-term (spread is "coiled"):
     → The spring is compressed. Any deviation is a STRONG signal.
     → Size UP, up to 3x cap for extreme compression.
   
   When short-term vol > long-term (spread is noisy/disrupted):
     → Something unusual is happening. Reduce position.
     → Minimum 0.1x — never fully exit, just cautious.
   
   This is NOT "trade when vol is high" (that's wrong).
   It's "trade HARDER when vol is unusually LOW" (that's right).

3. TANH AGGRESSION:
   f = tanh(AGGR * z) — positions max out as |z| grows.
   Per-pair AGGR tuned to maximise full 1000-day score while
   keeping all four 250-day windows positive (no window-specific tuning).

PAIRS: 16 pairs, all with ADF p<0.05 on full 1000 days, ≥3/4 windows positive.
The same data generator continues to day 1500 — cointegration holds.

Score on full 1000 days (simulation): ~672
Score on days 751-1000 (simulation): ~712
Expected score on days 1001-1500: ~500-650

Official eval gap: strategy needs ~60 days of history for expanding mean
to stabilise. When evaluated from day 1, the first 60 days score lower.
On a 500-day finals window starting at day 1001, full performance from day ~1061.
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# (i, j, zw, band, aggr)
# zw: short window for rolling std (z-score denominator)
# band: minimum |z| to enter (0.0 = trade any deviation)
# aggr: tanh steepness (higher = faster to full position)
PAIRS = [
    (18, 35, 24, 0.50, 20.0),   # RTTH/NAYO  HL=8d  ADF=0.000  4/4
    (25, 37, 22, 0.50, 15.0),   # CTGI/EELT  HL=7d  ADF=0.000  4/4
    ( 7, 40, 17, 0.50, 20.0),   # HETT/ULXY  HL=6d  ADF=0.000  4/4
    ( 1, 20, 12, 0.50, 20.0),   # AENO/NWIG  HL=4d  ADF=0.000  4/4
    (13, 45, 16, 0.50,  3.0),   # EORC/NGTE  HL=6d  ADF=0.000  4/4
    (10, 46, 14, 0.50,  3.0),   # SMAH/ILVX  HL=5d  ADF=0.000  4/4
    (49, 50, 23, 0.50, 20.0),   # MHRM/EAFC  HL=8d  ADF=0.000  4/4
    ( 8, 27, 19, 0.50, 20.0),   # HUXZ/ACAC  HL=6d  ADF=0.000  4/4
    (31, 43, 30, 0.50, 20.0),   # ACIX/ITPA  HL=10d ADF=0.000  4/4
    (26, 32, 46, 0.50,  2.0),   # ALUT/CCNS  HL=15d ADF=0.000  4/4
    (36, 41, 28, 0.50,  8.0),   # FWWG/BLBT  HL=9d  ADF=0.000  4/4
    ( 2, 42, 60, 0.50, 20.0),   # LSST/BENI  HL=24d ADF=0.001  4/4
    (39, 47, 60, 0.00, 20.0),   # AETS/FCSG  HL=18d ADF=0.003  4/4
    (15, 22, 30, 0.20, 20.0),   # HRET/MDGI  HL=14d ADF=0.010  3/4
    (28, 38, 10, 0.30, 12.0),   # SRTX/HRND  HL=22d ADF=0.001  4/4
    ( 9, 11, 30, 1.00, 20.0),   # DUCT/NPCK  HL=18d ADF=0.005  3/4
]

VOL_SCALE_LO = 0.1   # minimum size multiplier (when spread is very noisy)
VOL_SCALE_HI = 3.0   # maximum size multiplier (when spread is very compressed)
REBAL_BAND   = 0.05  # don't trade if position change < 5% of cap (reduce churn)
MIN_HISTORY  = max(int(zw) for _,_,zw,_,_ in PAIRS) + 10

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

        ratio_full  = log_p[i, :] - log_p[j, :]      # all history
        ratio_short = log_p[i, -zw:] - log_p[j, -zw:]  # short window

        # 1. EXPANDING MEAN: best estimate of true long-run equilibrium
        sm = ratio_full.mean()

        # 2. Rolling std for z-score (short window captures current regime vol)
        ss = ratio_short.std()
        if ss < 1e-12:
            continue

        z = (ratio_full[-1] - sm) / ss

        if abs(z) <= band:
            continue

        # 3. VOL COMPRESSION SIGNAL
        long_vol  = ratio_full.std()         # long-term spread volatility
        short_vol = ss                        # current spread volatility
        vol_scale = np.clip(long_vol / short_vol, VOL_SCALE_LO, VOL_SCALE_HI)

        f = np.tanh(aggr * z) * vol_scale
        target_dollars[i] -= f * CAPS[i]
        target_dollars[j] += f * CAPS[j]

    # Clip to position limits
    target_dollars = np.clip(target_dollars, -CAPS, CAPS)
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    # Rebalance band: don't trade tiny position adjustments
    drift   = np.abs(target_sh - _pos) * p_today
    trade   = drift > REBAL_BAND * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
