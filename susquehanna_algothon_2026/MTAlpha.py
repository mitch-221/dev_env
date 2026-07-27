"""
Advanced pairs optimization using:
1. Ornstein-Uhlenbeck (OU) model - optimal position sizing from mean reversion speed
2. Kalman filter - adaptive spread tracking
3. Walk-forward return prediction - use spread momentum/acceleration
4. Ensemble: combine signals for better entry/exit

Run: python advanced_pairs.py prices.txt
Takes ~5 minutes for full run.
"""

import sys
import numpy as np
from statsmodels.tsa.stattools import adfuller

arr = np.loadtxt('prices.txt', skiprows=1)
prc = arr.T
n_inst, n_days = prc.shape
CAPS = np.full(51, 10_000.); CAPS[0] = 100_000.
COMM = np.full(51, 0.0001); COMM[0] = 0.00002
log_p = np.log(prc)
rets = np.diff(log_p, axis=1)

TICKERS = ['ALGO','AENO','LSST','SRNA','ELLT','AMRP','OTCS','HETT','HUXZ','DUCT',
    'SMAH','NPCK','MSDP','EORC','CUBO','HRET','ANSO','DIHO','RTTH','SPLZ',
    'NWIG','MMBT','MDGI','AGVF','RRES','CTGI','ALUT','ACAC','SRTX','GARI',
    'RCRI','ACIX','CCNS','MTNS','IHOZ','NAYO','FWWG','EELT','HRND','AETS',
    'ULXY','BLBT','BENI','ITPA','HTRK','NGTE','ILVX','FCSG','FARS','MHRM','EAFC']
T = {t: i for i, t in enumerate(TICKERS)}

d500, dB = 500, 650

def objective(p):
    if len(p) == 0: return 0.
    mu, sd = p.mean(), p.std()
    if mu < 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.)

def rpt(pnl, label, s=100):
    mu, sd = pnl[s:].mean(), pnl[s:].std()
    sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0
    ev = objective(pnl[d500:])
    print(f"  {label:<58} EVAL:{ev:>+6.0f}  B:{objective(pnl[d500:dB]):>+5.0f}  "
          f"C:{objective(pnl[dB:]):>+5.0f}  mu${mu:>6.0f}  SR{sr:.2f}")

# All confirmed pairs from v8/v9
ALL_PAIRS = [
    ( 0, 22, 20, 0.30),  # ALGO/MDGI
    (25, 37, 40, 0.30),  # CTGI/EELT
    (18, 35, 40, 0.75),  # RTTH/NAYO
    (13, 45, 40, 0.50),  # EORC/NGTE
    ( 8, 20, 20, 0.75),  # HUXZ/NWIG
    (49, 50, 40, 0.30),  # MHRM/EAFC
    (10, 46, 30, 0.75),  # SMAH/ILVX
    ( 9, 27, 20, 1.00),  # DUCT/ACAC
    (36, 41, 30, 0.30),  # FWWG/BLBT
    (39, 44, 30, 0.75),  # AETS/HTRK
    (31, 43, 20, 0.30),  # ACIX/ITPA
    ( 7, 33, 20, 0.30),  # HETT/MTNS
    ( 2, 16, 20, 1.00),  # LSST/ANSO
    ( 4, 17, 30, 1.00),  # ELLT/DIHO
    (26, 32, 40, 1.00),  # ALUT/CCNS
    ( 1, 29, 15, 0.20),  # AENO/GARI
    (38, 19, 20, 0.50),  # HRND/SPLZ
    (21, 11, 40, 0.60),  # MMBT/NPCK
    (47,  3, 50, 1.00),  # FCSG/SRNA
    (12, 40, 40, 1.00),  # MSDP/ULXY
    (42, 15, 40, 1.00),  # BENI/HRET
]

# ================================================================
print("=" * 72)
print("METHOD 1: BASELINE z-score (reference)")
print("=" * 72)

def pair_baseline(i, j, zw=30, band=0.5, aggr=2.0):
    ratio = log_p[i] - log_p[j]
    pos_i = 0.; pos_j = 0.; pnl = np.zeros(n_days)
    for t in range(zw + 1, n_days - 1):
        win = ratio[t - zw:t]
        sm, ss = win.mean(), win.std()
        if ss < 1e-12: continue
        z = (ratio[t] - sm) / ss
        if float(abs(z)) <= band: ni, nj = 0., 0.
        else:
            f = np.tanh(aggr * z)
            ni = -f * CAPS[i]; nj = f * CAPS[j]
        cost = (abs(ni - pos_i) * COMM[i] + abs(nj - pos_j) * COMM[j])
        pos_i = ni; pos_j = nj
        pnl[t + 1] = pos_i * rets[i, t] + pos_j * rets[j, t] - cost
    return pnl

base_port = sum(pair_baseline(i, j, zw, b) for i, j, zw, b in ALL_PAIRS)
rpt(base_port, "Baseline z-score (21 pairs)")

# ================================================================
print()
print("=" * 72)
print("METHOD 2: OU MODEL")
print("  Fit OU process to spread: ds = kappa*(mu-s)*dt + sigma*dW")
print("  Optimal position: proportional to (mu-s)/sigma²")
print("  = how far from mean relative to noise => size up when more certain")
print("=" * 72)

def pair_ou_model(i, j, zw=60, min_kappa=0.01, aggr=2.0):
    """OU model: estimate kappa, mu, sigma from rolling window.
    Position = tanh(aggr * (mu - s) / sigma) — naturally sized by
    reversion strength relative to noise."""
    ratio = log_p[i] - log_p[j]
    pos_i = 0.; pos_j = 0.; pnl = np.zeros(n_days)
    for t in range(zw + 5, n_days - 1):
        s = ratio[t - zw:t]
        # AR(1) fit: s[t] = a + b*s[t-1] + e  => kappa = -log(b)/dt
        b = np.cov(s[1:], s[:-1])[0, 1] / np.var(s[:-1])
        b = np.clip(b, 0.01, 0.99)  # OU requires 0 < b < 1
        kappa = -np.log(b)          # mean reversion speed
        mu_ou = s.mean()            # long-run mean
        sigma = s.std() * np.sqrt(1 - b**2)  # noise std
        if sigma < 1e-8 or kappa < min_kappa: continue
        # OU optimal signal: how many sigmas from mean, discounted by speed
        z = (mu_ou - ratio[t]) / sigma
        f = np.tanh(aggr * z)
        ni = f * CAPS[i]; nj = -f * CAPS[j]  # long i when i is cheap (z>0)
        cost = (abs(ni - pos_i) * COMM[i] + abs(nj - pos_j) * COMM[j])
        pos_i = ni; pos_j = nj
        pnl[t + 1] = pos_i * rets[i, t] + pos_j * rets[j, t] - cost
    return pnl

ou_results = []
for i, j, zw, band in ALL_PAIRS:
    best_both = -np.inf; best_p = None; best_cfg = None
    for zw_ou in [40, 60, 80, 100]:
        for ag in [1.5, 2.0, 3.0]:
            p = pair_ou_model(i, j, zw_ou, aggr=ag)
            sb = objective(p[d500:dB]); sc = objective(p[dB:])
            both = min(sb, sc)
            if both > best_both:
                best_both = both; best_p = p; best_cfg = (zw_ou, ag)
    ou_results.append((best_both, i, j, best_cfg, best_p))

ou_port = sum(r[4] for r in ou_results)
rpt(ou_port, "OU model (per-pair optimised)")
# Show individual improvements
ou_results.sort(key=lambda r: -r[0])
print("  Top OU improvements vs baseline:")
for both, i, j, cfg, p in ou_results[:8]:
    bp = pair_baseline(i, j, *(x for _, _, zw, b in ALL_PAIRS if _ == i and __ == j
                                for x in [zw, b]) if any(_ == i and __ == j
                                for _, __, ___, ____ in ALL_PAIRS) else [30, 0.5])
    print(f"    {TICKERS[i]}/{TICKERS[j]}: OU B={objective(p[d500:dB]):+.0f} C={objective(p[dB:]):+.0f} cfg={cfg}")

# ================================================================
print()
print("=" * 72)
print("METHOD 3: KALMAN FILTER")
print("  Track dynamic mean and variance of spread adaptively.")
print("  State: [level, trend]. Obs = ratio. Position from filtered z.")
print("=" * 72)

def pair_kalman(i, j, q=1e-4, r=1e-2, aggr=2.0, band=0.5):
    """Kalman filter — scalar implementation to avoid array issues."""
    ratio = log_p[i] - log_p[j]
    # state: [level, trend], scalar Kalman
    x0, x1 = float(ratio[0]), 0.0   # level, trend
    p00, p01, p11 = 0.1, 0.0, 0.1   # covariance elements
    q0, q1 = q, q * 0.1
    pos_i = 0.; pos_j = 0.; pnl = np.zeros(n_days)
    for t in range(1, n_days - 1):
        # predict: x = F @ x
        x0p = x0 + x1; x1p = x1
        p00p = p00 + 2*p01 + p11 + q0
        p01p = p01 + p11
        p11p = p11 + q1
        # update: H = [1, 0], innov = ratio[t] - x0p
        obs = float(ratio[t])
        innov = obs - x0p
        S = p00p + r
        k0 = p00p / S; k1 = p01p / S
        x0 = x0p + k0 * innov
        x1 = x1p + k1 * innov
        p00 = (1 - k0) * p00p
        p01 = (1 - k0) * p01p
        p11 = p11p - k1 * p01p
        # signal
        unc = max((p00 + r) ** 0.5, 1e-12)
        z = (x0 - obs) / unc
        if abs(z) <= band: ni, nj = 0., 0.
        else:
            f = float(np.tanh(aggr * z))
            ni = f * CAPS[i]; nj = -f * CAPS[j]
        cost = abs(ni - pos_i) * COMM[i] + abs(nj - pos_j) * COMM[j]
        pos_i = ni; pos_j = nj
        pnl[t + 1] = pos_i * rets[i, t] + pos_j * rets[j, t] - cost
    return pnl

kalman_results = []
for i, j, zw, band in ALL_PAIRS:
    best_both = -np.inf; best_p = None; best_cfg = None
    for q in [1e-5, 1e-4, 1e-3]:
        for r in [1e-3, 1e-2, 1e-1]:
            for ag in [1.5, 2.0, 3.0]:
                for b in [0.0, 0.3, 0.5]:
                    p = pair_kalman(i, j, q, r, ag, b)
                    sb = objective(p[d500:dB]); sc = objective(p[dB:])
                    both = min(sb, sc)
                    if both > best_both:
                        best_both = both; best_p = p; best_cfg = (q, r, ag, b)
    kalman_results.append((best_both, i, j, best_cfg, best_p))

kalman_port = sum(r[4] for r in kalman_results)
rpt(kalman_port, "Kalman filter (per-pair optimised)")

# ================================================================
print()
print("=" * 72)
print("METHOD 4: WALK-FORWARD RETURN PREDICTION")
print("  Predict NEXT DAY'S spread change using:")
print("  - Current z-score (mean reversion signal)")
print("  - Rate of change of z (momentum/acceleration)")
print("  - Recent spread vol (scale position by opportunity)")
print("=" * 72)

def pair_predict(i, j, zw=30, band=0.3, aggr=2.0, use_accel=True, use_vol_scale=True):
    ratio = log_p[i] - log_p[j]
    pos_i = 0.; pos_j = 0.; pnl = np.zeros(n_days)
    for t in range(zw + 5, n_days - 1):
        win = ratio[t - zw:t]
        sm, ss = win.mean(), win.std()
        if ss < 1e-12: continue
        z = (ratio[t] - sm) / ss
        # acceleration: change in z from yesterday
        if t > zw + 5:
            win_prev = ratio[t - zw - 1:t - 1]
            sm_p, ss_p = win_prev.mean(), win_prev.std()
            z_prev = (ratio[t - 1] - sm_p) / max(ss_p, 1e-12)
            accel = z - z_prev   # positive = moving away from mean (delay entry)
        else:
            accel = 0.
        # vol scaling: how large are typical spread moves recently?
        daily_moves = np.abs(np.diff(win))
        vol_scale = daily_moves.mean() / max(daily_moves.std(), 1e-12)
        # combined signal
        if use_accel:
            # dampen signal if spread is accelerating away from mean
            # (wait for turn, don't catch falling knife)
            signal = z * (1 - 0.3 * np.sign(z) * np.sign(accel))
        else:
            signal = z
        if use_vol_scale:
            # size up when spread is volatile (more opportunity)
            signal = signal * np.clip(vol_scale, 0.5, 2.0)
        if float(abs(signal)) <= band: ni, nj = 0., 0.
        else:
            f = np.tanh(aggr * signal)
            ni = -f * CAPS[i]; nj = f * CAPS[j]
        cost = (abs(ni - pos_i) * COMM[i] + abs(nj - pos_j) * COMM[j])
        pos_i = ni; pos_j = nj
        pnl[t + 1] = pos_i * rets[i, t] + pos_j * rets[j, t] - cost
    return pnl

pred_port = sum(pair_predict(i, j, zw, b) for i, j, zw, b in ALL_PAIRS)
rpt(pred_port, "Walk-forward prediction (accel + vol scale)")

# ================================================================
print()
print("=" * 72)
print("METHOD 5: ENSEMBLE — vote across methods")
print("  Average signals from baseline, OU, and Kalman.")
print("  Reduces noise from any single model being wrong.")
print("=" * 72)

def pair_ensemble(i, j, zw=30, band_base=0.5, zw_ou=60, q=1e-4, r=1e-2, aggr=2.0):
    ratio = log_p[i] - log_p[j]
    # Kalman state
    kx0, kx1 = float(ratio[0]), 0.0
    kp00, kp01, kp11 = 0.1, 0.0, 0.1
    q0k, q1k = q, q * 0.1

    pos_i = 0.; pos_j = 0.; pnl = np.zeros(n_days)
    start = max(zw + 5, zw_ou + 5)

    for t in range(1, n_days - 1):
        # Kalman update (scalar)
        kx0p = kx0 + kx1; kx1p = kx1
        kp00p = kp00 + 2*kp01 + kp11 + q0k
        kp01p = kp01 + kp11
        kp11p = kp11 + q1k
        obs_t = float(ratio[t])
        kinnov = obs_t - kx0p
        kS = kp00p + r
        kk0 = kp00p / kS; kk1 = kp01p / kS
        kx0 = kx0p + kk0 * kinnov
        kx1 = kx1p + kk1 * kinnov
        kp00 = (1 - kk0) * kp00p
        kp01 = (1 - kk0) * kp01p
        kp11 = kp11p - kk1 * kp01p

        if t < start:
            continue

        # Signal 1: baseline z-score
        win = ratio[t - zw:t]
        sm, ss = win.mean(), win.std()
        z_base = (ratio[t] - sm) / ss if ss > 1e-12 else 0.

        # Signal 2: OU model
        s_ou = ratio[t - zw_ou:t]
        b_ou = np.cov(s_ou[1:], s_ou[:-1])[0, 1] / max(np.var(s_ou[:-1]), 1e-12)
        b_ou = np.clip(b_ou, 0.01, 0.99)
        mu_ou = s_ou.mean()
        sigma_ou = s_ou.std() * np.sqrt(1 - b_ou**2)
        z_ou = (mu_ou - ratio[t]) / max(sigma_ou, 1e-12)

        # Signal 3: Kalman
        z_kal = (kx0 - obs_t) / max((kp00 + r)**0.5, 1e-12)

        # Ensemble: weighted average (equal weight)
        z_ens = (z_base + z_ou + z_kal) / 3.0

        # Only trade when ensemble agrees (all same sign)
        signs = [np.sign(z_base), np.sign(z_ou), np.sign(z_kal)]
        agreement = abs(sum(signs)) / 3.0  # 1.0 = all agree, 0.33 = 2v1

        if agreement < 0.67:  # require at least 2/3 agreement
            ni, nj = 0., 0.
        else:
            f = np.tanh(aggr * z_ens * agreement)
            ni = -f * CAPS[i]; nj = f * CAPS[j]

        cost = (abs(ni - pos_i) * COMM[i] + abs(nj - pos_j) * COMM[j])
        pos_i = ni; pos_j = nj
        if t < n_days - 1:
            pnl[t + 1] = pos_i * rets[i, t] + pos_j * rets[j, t] - cost
    return pnl

ens_port = sum(pair_ensemble(i, j, zw, b) for i, j, zw, b in ALL_PAIRS)
rpt(ens_port, "Ensemble (baseline + OU + Kalman)")

# ================================================================
print()
print("=" * 72)
print("METHOD 6: PER-PAIR OPTIMISED — find best method per pair")
print("=" * 72)
best_per_pair = {}
for i, j, zw, band in ALL_PAIRS:
    candidates = {
        'baseline': pair_baseline(i, j, zw, band),
        'ou': pair_ou_model(i, j, max(zw*2, 60)),
        'kalman': pair_kalman(i, j),
        'predict': pair_predict(i, j, zw, band),
        'ensemble': pair_ensemble(i, j, zw, band),
    }
    # Score each on B only (tune on B, validate on C)
    best_name = max(candidates, key=lambda k: objective(candidates[k][d500:dB]))
    best_per_pair[(i,j)] = (best_name, candidates[best_name])

per_pair_port = sum(v[1] for v in best_per_pair.values())
rpt(per_pair_port, "Per-pair best method (tuned on B)")

# Breakdown
print("  Method selection per pair:")
from collections import Counter
method_counts = Counter(v[0] for v in best_per_pair.values())
for method, count in method_counts.most_common():
    print(f"    {method}: {count} pairs")

# ================================================================
print()
print("=" * 72)
print("SUMMARY: best approach per metric")
print("=" * 72)
methods = {
    'baseline': base_port,
    'ou_model': ou_port,
    'kalman': kalman_port,
    'prediction': pred_port,
    'ensemble': ens_port,
    'per_pair': per_pair_port,
}
for name, port in sorted(methods.items(), key=lambda kv: -objective(kv[1][d500:])):
    ep = port[d500:]
    mu_e, sd_e = ep.mean(), ep.std()
    sr_e = np.sqrt(250) * mu_e / sd_e if sd_e > 1e-12 else 0
    print(f"  {name:<15}: Score={objective(ep):>+6.0f}  mu=${mu_e:>6.0f}  "
          f"B={objective(port[d500:dB]):>+5.0f}  C={objective(port[dB:]):>+5.0f}")

# Best method — build the submission
best_method = max(methods, key=lambda k: objective(methods[k][d500:]))
print(f"\nBest method: {best_method} — Score {objective(methods[best_method][d500:]):.0f}")
print()
print("Paste the winning method into the strategy file.")
