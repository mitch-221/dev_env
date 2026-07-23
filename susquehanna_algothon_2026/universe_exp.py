"""
Universe diagnostics for the Algothon price data.

Usage:
    python explore.py prices.txt

Answers, per section:
  1. Basics          - days, price ranges, per-asset daily vol
  2. Reversion vs momentum
                     - lag-1..5 return autocorrelation per asset
                     - horizon-vs-horizon return predictability
  3. ALGO structure  - ALGO's own autocorr, and lead-lag: do assets
                       1-50 predict ALGO's next-day move (or vice versa)?
  4. Factor structure- correlation matrix summary + PCA: how much
                       variance does the top factor explain?
  5. Signal sanity   - quick IC (rank correlation) of simple momentum
                       and reversal signals vs next-day returns
"""

import sys
import numpy as np

N_INST = 51


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
    raise ValueError(f"Expected one dimension == {N_INST}, got {arr.shape}")


def autocorr(x, lag):
    if len(x) <= lag + 2:
        return np.nan
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def rank_ic(signal, fwd_ret):
    """Cross-sectional Spearman IC averaged over days."""
    from scipy.stats import spearmanr
    ics = []
    for t in range(signal.shape[1]):
        s, r = signal[:, t], fwd_ret[:, t]
        ok = np.isfinite(s) & np.isfinite(r)
        if ok.sum() > 5 and s[ok].std() > 0 and r[ok].std() > 0:
            ics.append(spearmanr(s[ok], r[ok]).statistic)
    ics = np.array(ics)
    return ics.mean(), ics.mean() / (ics.std() / np.sqrt(len(ics)))


def main(path):
    prc = load_prices(path)
    n_inst, n_days = prc.shape
    rets = np.diff(np.log(prc), axis=1)

    print("=" * 64)
    print("1. BASICS")
    print("=" * 64)
    print(f"Assets: {n_inst}   Days: {n_days}")
    vol = rets.std(axis=1)
    print(f"Daily vol  - ALGO: {vol[0]:.4f}   others: "
          f"min {vol[1:].min():.4f}  median {np.median(vol[1:]):.4f}  "
          f"max {vol[1:].max():.4f}")
    drift = rets.mean(axis=1)
    print(f"Daily drift- ALGO: {drift[0]:+.5f}  others median: "
          f"{np.median(drift[1:]):+.5f}")

    print()
    print("=" * 64)
    print("2. REVERSION vs MOMENTUM (return autocorrelation)")
    print("=" * 64)
    print("Negative lag-1 autocorr => mean reversion; positive => momentum")
    for lag in (1, 2, 3, 5):
        acs = np.array([autocorr(rets[i], lag) for i in range(n_inst)])
        n_neg = (acs[1:] < 0).sum()
        print(f"lag {lag}:  ALGO {acs[0]:+.3f} | others: "
              f"mean {np.nanmean(acs[1:]):+.3f}, "
              f"{n_neg}/50 negative, "
              f"strongest neg {np.nanmin(acs[1:]):+.3f}, "
              f"strongest pos {np.nanmax(acs[1:]):+.3f}")

    # Multi-horizon: does past k-day return predict next-day return?
    print("\nPast k-day return vs next-day return (pooled corr):")
    for k in (3, 5, 10, 20, 40):
        past = np.array([np.convolve(rets[i], np.ones(k), "valid")
                         for i in range(n_inst)])[:, :-1]
        fwd = rets[:, k:]
        c = np.corrcoef(past.ravel(), fwd.ravel())[0, 1]
        print(f"  k={k:>2}: {c:+.4f}")

    print()
    print("=" * 64)
    print("3. ALGO STRUCTURE & LEAD-LAG")
    print("=" * 64)
    algo = rets[0]
    others = rets[1:]
    # Contemporaneous correlation of each asset with ALGO
    c_now = np.array([np.corrcoef(algo, others[i])[0, 1]
                      for i in range(50)])
    print(f"Contemporaneous corr with ALGO: mean {c_now.mean():+.3f}, "
          f"max |{np.abs(c_now).max():.3f}|")
    # Others today -> ALGO tomorrow
    c_lead = np.array([np.corrcoef(others[i][:-1], algo[1:])[0, 1]
                       for i in range(50)])
    order = np.argsort(-np.abs(c_lead))
    print("Top 5 assets whose TODAY return predicts ALGO TOMORROW:")
    for i in order[:5]:
        print(f"  asset {i+1:>2}: corr {c_lead[i]:+.3f}")
    # ALGO today -> others tomorrow
    c_lag = np.array([np.corrcoef(algo[:-1], others[i][1:])[0, 1]
                      for i in range(50)])
    order = np.argsort(-np.abs(c_lag))
    print("Top 5 assets that FOLLOW ALGO with one day lag:")
    for i in order[:5]:
        print(f"  asset {i+1:>2}: corr {c_lag[i]:+.3f}")

    print()
    print("=" * 64)
    print("4. FACTOR STRUCTURE (PCA on return correlation)")
    print("=" * 64)
    cm = np.corrcoef(rets)
    off = cm[np.triu_indices(n_inst, k=1)]
    print(f"Pairwise corr: mean {off.mean():+.3f}, "
          f"5%/95%: {np.percentile(off, 5):+.3f} / {np.percentile(off, 95):+.3f}")
    evals = np.linalg.eigvalsh(cm)[::-1]
    share = evals / evals.sum()
    print(f"Top PCA factors explain: "
          f"{share[0]:.1%}, {share[1]:.1%}, {share[2]:.1%} "
          f"(equal-weight baseline would be {1/n_inst:.1%} each)")

    print()
    print("=" * 64)
    print("5. QUICK SIGNAL ICs (cross-sectional, next-day returns)")
    print("=" * 64)
    fwd = rets[:, 1:]

    def trailing_sum(k):
        return np.array([np.convolve(rets[i], np.ones(k), "valid")
                         for i in range(n_inst)])[:, :-1]

    for name, k, sign in (("5d reversal", 5, -1), ("10d reversal", 10, -1),
                          ("20d momentum", 20, +1), ("40d momentum", 40, +1)):
        sig = sign * trailing_sum(k)
        m, t = rank_ic(sig, fwd[:, k - 1:])
        verdict = "***" if abs(t) > 3 else ("*" if abs(t) > 2 else "")
        print(f"  {name:>12}: mean IC {m:+.4f}  (t-stat {t:+.1f}) {verdict}")

    print("\nPositive IC with good t-stat => signal predicts returns.")
    print("If reversal ICs dominate, weight W_REV up; if momentum, W_MOM.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
