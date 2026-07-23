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
    tickers = None
    with open(path) as f:
        first = f.readline()
    skip = 0
    try:
        [float(x) for x in first.replace(",", " ").split()]
    except ValueError:
        skip = 1
        tickers = first.replace(",", " ").split()
    try:
        arr = np.loadtxt(path, skiprows=skip)
    except ValueError:
        arr = np.loadtxt(path, delimiter=",", skiprows=skip)
    if arr.shape[0] == N_INST and arr.shape[1] != N_INST:
        pass
    elif arr.shape[1] == N_INST:
        arr = arr.T
    else:
        raise ValueError(f"Expected one dimension == {N_INST}, got {arr.shape}")
    if tickers is None or len(tickers) != N_INST:
        tickers = [f"A{i:02d}" for i in range(N_INST)]
    return arr, tickers


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


def pairs_analysis(prc, rets, tickers, out_prefix="universe"):
    """Write correlation matrix CSV + a pairs report ranked by the
    out-of-sample net Sharpe of a toy z-score reversion strategy.

    Every one of the 51*50/2 pairs is tested. Hedge ratio is fit on the
    first 60% of days (train); the Sharpe is computed only on the last
    40% (test), net of real commissions. This guards against the
    multiple-testing trap of screening 1275 pairs.
    """
    import csv
    from statsmodels.tsa.stattools import adfuller

    n_inst, n_days = prc.shape
    cm = np.corrcoef(rets)

    # ---- full correlation matrix -> CSV ----
    cm_path = f"{out_prefix}_corr_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + tickers)
        for i in range(n_inst):
            w.writerow([tickers[i]] + [f"{cm[i, j]:.4f}" for j in range(n_inst)])

    log_p = np.log(prc)
    split = int(n_days * 0.6)
    Z_WINDOW = 30          # rolling window for spread z-score
    Z_CAP = 2.0            # position = -clip(z, -cap, cap)
    Z_BAND = 0.25          # only rebalance when z moves this much (cost control)
    COMM = np.full(n_inst, 0.0001)
    COMM[0] = 0.00002

    def pair_backtest(i, j):
        """Returns (oos_sharpe, beta, half_life, spr_vol, adf_p)."""
        x, y = log_p[j], log_p[i]
        xt, yt = x[:split], y[:split]
        vx = np.var(xt)
        if vx < 1e-12:
            return None
        beta = np.cov(yt, xt)[0, 1] / vx
        spread = y - beta * x

        # diagnostics on train segment
        st = spread[:split]
        try:
            adf_p = adfuller(st, autolag="AIC")[1]
        except Exception:
            adf_p = np.nan
        s0, s1 = st[:-1], st[1:]
        b = np.cov(s1 - s0, s0)[0, 1] / np.var(s0)
        half_life = -np.log(2) / b if b < 0 else np.inf
        spr_vol = np.diff(st).std()

        # rolling z-score over full series
        n = len(spread)
        if n <= Z_WINDOW + 5:
            return None
        z = np.full(n, np.nan)
        for t in range(Z_WINDOW, n):
            w_ = spread[t - Z_WINDOW:t]
            sd = w_.std()
            if sd > 1e-12:
                z[t] = (spread[t] - w_.mean()) / sd

        # positions in $ per $1 of unit size: leg i = -z, leg j = +z*beta
        pos = 0.0
        pnl = np.zeros(n)
        gross = np.abs(1.0) + np.abs(beta)
        for t in range(Z_WINDOW, n - 1):
            zt = z[t]
            if np.isfinite(zt) and abs(zt - (-pos)) > Z_BAND:
                new_pos = -np.clip(zt, -Z_CAP, Z_CAP)
            else:
                new_pos = pos
            turn = abs(new_pos - pos)
            cost = turn * (COMM[i] + abs(beta) * COMM[j])
            pos = new_pos
            # next-day pnl of $pos in leg i, $-pos*beta... expressed via spread
            ds = spread[t + 1] - spread[t]
            pnl[t + 1] = pos * ds - cost
        oos = pnl[max(split, Z_WINDOW + 1):]
        full = pnl[Z_WINDOW + 1:]
        sd, sdf = oos.std(), full.std()
        if sd < 1e-12 or sdf < 1e-12:
            return None
        sharpe = np.sqrt(249) * oos.mean() / sd
        sharpe_full = np.sqrt(249) * full.mean() / sdf
        return sharpe, beta, half_life, spr_vol, adf_p, sharpe_full

    results = []
    for i in range(n_inst):
        for j in range(i + 1, n_inst):
            r = pair_backtest(i, j)
            if r is not None:
                results.append((r[0], i, j, cm[i, j], r[1], r[2], r[3],
                                r[4], r[5]))
    results.sort(key=lambda r: -r[0])

    rep_path = f"{out_prefix}_pairs_report.txt"
    with open(rep_path, "w") as f:
        f.write("PAIRS / SPREAD ANALYSIS - ranked by OUT-OF-SAMPLE net Sharpe\n")
        f.write("=" * 84 + "\n")
        f.write(f"All {len(results)} pairs backtested with a z-score reversion "
                f"rule (window {Z_WINDOW}d,\n")
        f.write(f"cap {Z_CAP}, rebalance band {Z_BAND}), real commissions "
                f"(1bp / 0.2bp ALGO).\n")
        f.write(f"Hedge ratio fit on days 0-{split} (train); OOS Sharpe "
                f"measured on days {split}-{n_days}.\n")
        f.write("Full Sharpe covers the whole period (incl. the days beta was "
                "fit on - flattering).\n")
        f.write("A big Full-vs-OOS gap = relationship was luck or has "
                "drifted. Rank by OOS.\n\n")
        f.write(f"{'pair':<14}{'OOS Sharpe':>11}{'Full Sharpe':>12}"
                f"{'ret corr':>10}{'beta':>8}"
                f"{'ADF p':>8}{'half-life':>11}{'spr vol':>9}\n")
        f.write("-" * 84 + "\n")
        for s, i, j, c, beta, hl, sv, adf_p, sf in results[:40]:
            name = f"{tickers[i]}/{tickers[j]}"
            hl_s = f"{hl:.1f}" if np.isfinite(hl) else "inf"
            f.write(f"{name:<14}{s:>+11.2f}{sf:>+12.2f}{c:>+10.3f}{beta:>8.2f}"
                    f"{adf_p:>8.3f}{hl_s:>11}{sv:>9.4f}\n")

        f.write("\n" + "=" * 84 + "\n")
        strong = [r for r in results if r[0] > 1.0]
        f.write(f"Pairs with OOS net Sharpe > 1.0: {len(strong)}\n")
        f.write("Sanity notes:\n")
        f.write("- With ~1275 pairs tested, expect a handful of OOS Sharpes "
                "near +/-1 by pure luck.\n")
        f.write("  Trust pairs where high OOS Sharpe coincides with low ADF p "
                "and sensible half-life.\n")
        f.write("- Check the same pair ranks well if you nudge Z_WINDOW; "
                "fragile rankings are noise.\n")
        f.write("- If ONE ticker appears near the top paired with many "
                "different partners, the edge is\n")
        f.write("  that single asset's own mean reversion, not a pairs "
                "relationship - trade it outright\n")
        f.write("  instead (cheaper: one leg of commission, not two).\n")

    print(f"\nWrote: {cm_path}")
    print(f"Wrote: {rep_path}")
    if results:
        s, i, j, c, beta, hl, sv, adf_p, sf = results[0]
        print(f"Best OOS pair: {tickers[i]}/{tickers[j]} "
              f"OOS Sharpe {s:+.2f} (full {sf:+.2f}, corr {c:+.3f}, "
              f"ADF p {adf_p:.3f})")
        print(f"Pairs with OOS net Sharpe > 1.0: "
              f"{sum(1 for r in results if r[0] > 1.0)}")

    # ---- full correlation matrix -> CSV ----
    cm_path = f"{out_prefix}_corr_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + tickers)
        for i in range(n_inst):
            w.writerow([tickers[i]] + [f"{cm[i, j]:.4f}" for j in range(n_inst)])


def main(path):
    prc, tickers = load_prices(path)
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

    print()
    print("=" * 64)
    print("6. CORRELATION MATRIX & PAIRS (written to files)")
    print("=" * 64)
    pairs_analysis(prc, rets, tickers)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
