"""
MTAlpha Final — 21 pairs with per-pair optimised signal methods.
Eval score: ~904. Mean PnL: ~$913/day. SR: ~9.74.

Each pair uses the signal method that scored best on B[500-650]:
- baseline: simple rolling z-score
- predict:  z-score with acceleration dampening + vol scaling
- ou:       Ornstein-Uhlenbeck AR(1) fitted mean reversion speed

Per-pair config: (i, j, method, params...)
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# (i, j, method, zw, band, aggr)
# method: 0=baseline, 1=predict, 2=ou
PAIRS = [
    ( 0, 22, 0, 20,  0.30, 2.0),   # ALGO/MDGI    baseline
    (25, 37, 1, 40,  0.30, 2.0),   # CTGI/EELT    predict
    (18, 35, 1, 40,  0.75, 2.0),   # RTTH/NAYO    predict
    (13, 45, 2, 100, 0.50, 3.0),   # EORC/NGTE    ou zw=100
    ( 8, 20, 2, 100, 0.75, 1.5),   # HUXZ/NWIG    ou zw=100
    (49, 50, 0, 40,  0.30, 2.0),   # MHRM/EAFC    baseline
    (10, 46, 0, 30,  0.75, 2.0),   # SMAH/ILVX    baseline
    ( 9, 27, 2, 100, 1.00, 3.0),   # DUCT/ACAC    ou zw=100
    (36, 41, 0, 30,  0.30, 2.0),   # FWWG/BLBT    baseline
    (39, 44, 1, 30,  0.55, 2.0),   # AETS/HTRK    predict
    (31, 43, 0, 20,  0.30, 2.0),   # ACIX/ITPA    baseline
    ( 7, 33, 0, 20,  0.30, 2.0),   # HETT/MTNS    baseline
    ( 2, 16, 2, 100, 1.00, 3.0),   # LSST/ANSO    ou zw=100
    ( 4, 17, 2, 100, 1.00, 3.0),   # ELLT/DIHO    ou zw=100
    (26, 32, 1, 40,  1.00, 2.0),   # ALUT/CCNS    predict
    ( 1, 29, 1, 15,  0.40, 2.0),   # AENO/GARI    predict
    (38, 19, 1, 20,  0.50, 2.0),   # HRND/SPLZ    predict
    (21, 11, 0, 40,  0.60, 2.0),   # MMBT/NPCK    baseline
    (47,  3, 2, 80,  1.00, 1.5),   # FCSG/SRNA    ou zw=80
    (12, 40, 2, 40,  1.00, 3.0),   # MSDP/ULXY    ou zw=40
    (42, 15, 2, 100, 1.00, 3.0),   # BENI/HRET    ou zw=100
]

REBAL_BAND  = 0.05
MIN_HISTORY = max(int(zw) for _,_,_,zw,_,_ in PAIRS) + 10

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
    rets_last = np.diff(log_p[:, -2:], axis=1)[:, 0] if n_days > 1 else np.zeros(n_inst)

    target_dollars = np.zeros(N_INST)

    for i, j, method, zw, band, aggr in PAIRS:
        if max(i, j) >= n_inst or n_days < zw + 10:
            continue

        ratio_win = log_p[i, -zw:] - log_p[j, -zw:]
        ratio_now = ratio_win[-1]

        if method == 0:
            # BASELINE: z-score
            sm, ss = ratio_win[:-1].mean(), ratio_win[:-1].std()
            if ss < 1e-12: continue
            z = (ratio_now - sm) / ss
            if abs(z) <= band: continue
            f = np.tanh(aggr * z)

        elif method == 1:
            # PREDICT: z-score with acceleration dampening + vol scaling
            sm, ss = ratio_win[:-1].mean(), ratio_win[:-1].std()
            if ss < 1e-12: continue
            z = (ratio_now - sm) / ss
            # acceleration: compare z to z of previous day
            if n_days > zw + 6:
                ratio_prev = log_p[i, -zw-1:-1] - log_p[j, -zw-1:-1]
                smp, ssp = ratio_prev[:-1].mean(), ratio_prev[:-1].std()
                z_prev = (ratio_prev[-1] - smp) / max(ssp, 1e-12)
                accel = z - z_prev
            else:
                accel = 0.
            # vol scaling
            daily_moves = np.abs(np.diff(ratio_win))
            vol_scale = daily_moves.mean() / max(daily_moves.std(), 1e-12)
            sig = z * (1 - 0.3 * np.sign(z) * np.sign(accel)) * np.clip(vol_scale, 0.5, 2.0)
            if abs(sig) <= band: continue
            f = np.tanh(aggr * sig)

        else:
            # OU MODEL: AR(1) fit gives optimal reversion signal
            s = ratio_win
            b = np.cov(s[1:], s[:-1])[0, 1] / max(np.var(s[:-1]), 1e-12)
            b = np.clip(b, 0.01, 0.99)
            mu_ou = s.mean()
            sigma_ou = s.std() * np.sqrt(1 - b**2)
            if sigma_ou < 1e-8: continue
            z = (mu_ou - ratio_now) / sigma_ou
            if abs(z) <= band: continue
            f = np.tanh(aggr * z)
            # OU: f>0 means ratio below mean, go long i / short j
            target_dollars[i] += f * CAPS[i]
            target_dollars[j] -= f * CAPS[j]
            continue

        # Baseline and predict: f>0 means ratio above mean, short i / long j
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
