"""
Is ALGO an index of the other 50 assets? Test + arbitrage simulation.

Tests:
  1. Return-space regression: ALGO returns vs the 50 others' returns.
     Fit on first half, report OUT-OF-SAMPLE R^2 on second half.
     An index gives R^2 near 1 OOS with stable weights.
  2. Simple candidates: equal-weight return, price-weighted return.
  3. Weight stability: correlation of fitted weights, half 1 vs half 2.
  4. If OOS R^2 is meaningful: walk-forward INDEX-ARB simulation -
     rolling regression fair value for ALGO, z-score of the deviation,
     trade ALGO against it (optionally hedged with the basket),
     scored on the A/B/C segments with the competition objective.

Usage:
    python algo_index_test.py prices.txt
"""

import sys
import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM = np.full(N_INST, 0.0001)
COMM[0] = 0.00002
WARM = 160


def load_prices(path):
    with open(path) as f:
        first = f.readline()
    skip = 0
    try:
        [float(x) for x in first.replace(",", " ").split()]
    except ValueError:
        skip = 1
    try:
        arr = np.loadtxt(path, skiprows=skip)
    except ValueError:
        arr = np.loadtxt(path, delimiter=",", skiprows=skip)
    if arr.shape[0] == N_INST and arr.shape[1] != N_INST:
        return arr
    if arr.shape[1] == N_INST:
        return arr.T
    raise ValueError(f"bad shape {arr.shape}")


def objective(pnl):
    if len(pnl) == 0:
        return 0.0
    mu, sd = pnl.mean(), pnl.std()
    if mu < 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def ols(X, y, ridge=1e-8):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - yhat) ** 2).sum() / ss if ss > 0 else np.nan


def main(path):
    prc = load_prices(path)
    n_inst, n_days = prc.shape
    rets = np.diff(np.log(prc), axis=1)
    a = rets[0]              # ALGO returns
    B = rets[1:].T           # (n_days-1, 50)
    half = len(a) // 2

    print("=" * 66)
    print("1. IS ALGO A LINEAR COMBINATION OF THE OTHER 50? (returns)")
    print("=" * 66)
    Xtr = np.hstack([B[:half], np.ones((half, 1))])
    w = ols(Xtr, a[:half])
    Xte = np.hstack([B[half:], np.ones((len(a) - half, 1))])
    print(f"  in-sample R^2:      {r2(a[:half], Xtr @ w):.4f}")
    oos = r2(a[half:], Xte @ w)
    print(f"  OUT-OF-SAMPLE R^2:  {oos:.4f}   <-- the number that matters")
    resid = a[half:] - Xte @ w
    print(f"  residual vol: {resid.std():.5f}  vs ALGO vol {a.std():.5f} "
          f"({resid.std()/a.std():.1%} unexplained)")

    # weight stability
    w2 = ols(np.hstack([B[half:], np.ones((len(a) - half, 1))]), a[half:])
    wc = np.corrcoef(w[:-1], w2[:-1])[0, 1]
    print(f"  weight stability (half1 vs half2 corr): {wc:+.3f}")
    top = np.argsort(-np.abs(w[:-1]))[:8]
    print("  largest weights (asset: w): "
          + ", ".join(f"{i+1}:{w[i]:+.3f}" for i in top))

    print()
    print("=" * 66)
    print("2. SIMPLE INDEX CANDIDATES (correlation with ALGO returns)")
    print("=" * 66)
    ew = B.mean(axis=1)
    pw = (prc[1:, :-1] * rets[1:].T.T).sum(axis=0) / prc[1:, :-1].sum(axis=0)
    print(f"  equal-weight basket:  corr {np.corrcoef(a, ew)[0,1]:+.4f}")
    print(f"  price-weighted basket: corr {np.corrcoef(a, pw)[0,1]:+.4f}")

    if oos < 0.10:
        print("\nOOS R^2 too low - ALGO is not usefully explained by the "
              "basket. No index-arb here.")
        return

    print()
    print("=" * 66)
    print("3. WALK-FORWARD INDEX-ARB SIMULATION")
    print("=" * 66)
    # rolling PRICE-level fair value: log p0 ~ basket log prices
    log_p = np.log(prc)
    W = 150          # rolling fit window
    ZW = 20          # z-score window for deviation
    dev = np.full(n_days, np.nan)
    for t in range(W, n_days):
        Xw = np.hstack([log_p[1:, t - W:t].T, np.ones((W, 1))])
        yw = log_p[0, t - W:t]
        wt = ols(Xw, yw, ridge=1e-6)
        fair = np.hstack([log_p[1:, t], 1.0]) @ wt
        dev[t] = log_p[0, t] - fair

    pos = 0.0
    pnl = np.zeros(n_days)
    for t in range(WARM, n_days - 1):
        window = dev[t - ZW:t]
        if np.isnan(window).any() or np.isnan(dev[t]):
            continue
        sd = window.std()
        if sd < 1e-10:
            continue
        z = (dev[t] - window.mean()) / sd
        new_pos = -np.tanh(z) * CAPS[0]
        cost = abs(new_pos - pos) * COMM[0]
        pos = new_pos
        pnl[t + 1] = pos * (log_p[0, t + 1] - log_p[0, t]) - cost

    d500 = min(500, int(n_days * 2 / 3))
    dB = min(650, n_days - 100)
    segs = [(f"A[{WARM}-{d500}]", pnl[WARM:d500]),
            (f"B[{d500}-{dB}]", pnl[d500:dB]),
            (f"C[{dB}-{n_days}]", pnl[dB:]),
            ("full", pnl[WARM:])]
    if n_days > 800:
        segs.append((f"LEADERBOARD[750-{n_days}]", pnl[750:]))
    print("  ALGO-only deviation trading (unhedged):")
    for name, p in segs:
        mu, sd = p.mean(), p.std()
        sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0
        print(f"    {name:<22} mu ${mu:>7,.0f}  sd ${sd:>7,.0f}  "
              f"SR {sr:>5.2f}  Score {objective(p):>7,.0f}")
    print("\n  If scores are strong, next step: hedge the basket leg and")
    print("  integrate into teamName as the primary ALGO component.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python algo_index_test.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
