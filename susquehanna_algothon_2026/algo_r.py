"""
Tests for predictability in the 17% of ALGO variance unexplained by
the contemporaneous basket.

1. RESIDUAL STATS     autocorr, variance ratio, distribution of the
                      fitted residual (ALGO - weighted basket)
2. LAGGED BASKET      does yesterday's basket return predict today's
                      ALGO residual? (weighted and equal-weight)
3. CONSTITUENT LAGS   which of the 50 assets individually predict
                      today's ALGO return with a 1-day lag?
4. STRATEGIES         walk-forward simulation of the best predictors,
                      scored on A/B/C segments with real commissions

Usage:
    python algo_residual.py prices.txt
"""

import sys
import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM = np.full(N_INST, 0.0001)
COMM[0] = 0.00002
WFIT = 150
WARM = WFIT + 5


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
        raise ValueError(f"bad shape {arr.shape}")
    if tickers is None or len(tickers) != N_INST:
        tickers = [f"A{i:02d}" for i in range(N_INST)]
    return arr, tickers


def objective(pnl):
    if len(pnl) == 0:
        return 0.0
    mu, sd = pnl.mean(), pnl.std()
    if mu < 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def ols(X, y, ridge=1e-6):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def rolling_resid(prc, wfit=WFIT, refit=5):
    """Walk-forward ALGO residual: ALGO return - fitted basket return."""
    n_inst, n_days = prc.shape
    rets = np.diff(np.log(prc), axis=1)
    w = None
    last_fit = -10**9
    resid = np.full(n_days - 1, np.nan)
    for t in range(wfit, n_days - 1):
        if w is None or (t - last_fit) >= refit:
            X = rets[1:, t - wfit:t].T
            y = rets[0, t - wfit:t]
            w = ols(X, y)
            last_fit = t
        resid[t] = rets[0, t] - w @ rets[1:, t]
    return resid, rets


def main(path):
    prc, tickers = load_prices(path)
    n_inst, n_days = prc.shape
    d500 = min(500, int(n_days * 2 / 3))
    dB = min(650, n_days - 100)
    segs = [(f"A[{WARM}-{d500}]", slice(WARM, d500)),
            (f"B[{d500}-{dB}]", slice(d500, dB)),
            (f"C[{dB}-{n_days}]", slice(dB, n_days))]

    resid, rets = rolling_resid(prc)
    # resid[t] = ALGO return on day t minus basket contribution
    # valid from index WFIT onwards

    # ================================================================
    print("=" * 66)
    print("1. RESIDUAL STATS")
    print("=" * 66)
    r = resid[WFIT:]
    r = r[np.isfinite(r)]
    print(f"  vol: {r.std():.5f}/day  (${r.std()*CAPS[0]:,.0f} at $100k)")
    print(f"  mean: {r.mean():+.6f}  (drift t={r.mean()/(r.std()/np.sqrt(len(r))):+.2f})")
    acs = [np.corrcoef(r[:-k], r[k:])[0, 1] for k in range(1, 6)]
    print("  autocorr lags 1-5: " + "  ".join(f"{x:+.3f}" for x in acs))
    for q in (2, 5, 10):
        k2 = len(r) // q * q
        vr = r[:k2].reshape(-1, q).sum(axis=1).var() / (q * r.var())
        print(f"  variance ratio q={q}: {vr:.3f}  "
              f"({'trending' if vr > 1.1 else 'reverting' if vr < 0.9 else 'random'})")
    for label, sl in ((f"pre-{d500}", slice(WFIT, d500)),
                      (f"post-{d500}", slice(d500, None))):
        rv = resid[sl]
        rv = rv[np.isfinite(rv)]
        if len(rv) > 10:
            ac1 = np.corrcoef(rv[:-1], rv[1:])[0, 1]
            print(f"  {label}: vol {rv.std():.5f}  lag-1 ac {ac1:+.3f}")

    # ================================================================
    print()
    print("=" * 66)
    print("2. LAGGED BASKET -> ALGO RESIDUAL")
    print("=" * 66)
    basket_ew = rets[1:].mean(axis=0)   # equal-weight
    # test: basket_t-1 predicts resid_t
    x_lag = basket_ew[:-1]
    y_res = resid[1:]
    ok = np.isfinite(y_res) & np.isfinite(x_lag)
    c = np.corrcoef(x_lag[ok & (np.arange(len(ok)) >= WFIT)],
                    y_res[ok & (np.arange(len(ok)) >= WFIT)])[0, 1]
    n = ok.sum()
    t_ew = c * np.sqrt(n) / np.sqrt(max(1 - c**2, 1e-9))
    star = " ***" if abs(t_ew) > 3 else (" *" if abs(t_ew) > 2 else "")
    print(f"  equal-weight basket(t-1) -> ALGO residual(t): "
          f"corr {c:+.4f}  t={t_ew:+.2f}{star}")

    # test: ALGO return(t-1) predicts ALGO return(t)
    a = rets[0]
    c2 = np.corrcoef(a[WFIT:-1], a[WFIT+1:])[0, 1]
    t2 = c2 * np.sqrt(n_days - WFIT) / np.sqrt(max(1 - c2**2, 1e-9))
    star2 = " ***" if abs(t2) > 3 else (" *" if abs(t2) > 2 else "")
    print(f"  ALGO(t-1) -> ALGO(t): corr {c2:+.4f}  t={t2:+.2f}{star2}")

    # test: basket(t-1) -> ALGO(t) raw (not residual)
    c3 = np.corrcoef(basket_ew[WFIT:-1], a[WFIT+1:])[0, 1]
    t3 = c3 * np.sqrt(n_days - WFIT) / np.sqrt(max(1 - c3**2, 1e-9))
    star3 = " ***" if abs(t3) > 3 else (" *" if abs(t3) > 2 else "")
    print(f"  basket(t-1) -> ALGO(t) raw: corr {c3:+.4f}  t={t3:+.2f}{star3}")

    # ================================================================
    print()
    print("=" * 66)
    print("3. INDIVIDUAL CONSTITUENT LAGS -> ALGO return")
    print("=" * 66)
    n_test = n_days - WFIT - 1
    lag_corrs = []
    for i in range(1, n_inst):
        xi = rets[i, WFIT:-1]
        yi = a[WFIT+1:]
        ok = np.isfinite(xi) & np.isfinite(yi)
        if ok.sum() < 30:
            lag_corrs.append((0.0, i))
            continue
        c = np.corrcoef(xi[ok], yi[ok])[0, 1]
        lag_corrs.append((c, i))
    lag_corrs.sort(key=lambda x: -abs(x[0]))
    print("  Top 10 constituents (by |lag-1 corr| with ALGO next-day return):")
    print(f"  {'asset':<8} {'ticker':<8} {'corr':>8}  {'t-stat':>8}")
    for c, i in lag_corrs[:10]:
        t = c * np.sqrt(n_test) / np.sqrt(max(1 - c**2, 1e-9))
        star = " ***" if abs(t) > 3 else (" *" if abs(t) > 2 else "")
        print(f"  {i:<8} {tickers[i]:<8} {c:>+8.4f}  {t:>+8.2f}{star}")

    # ================================================================
    print()
    print("=" * 66)
    print("4. STRATEGY SIMULATIONS (ALGO only, real commissions)")
    print("=" * 66)

    def run_algo_strat(signal_fn, start=WARM):
        pos = 0.0
        pnl = np.zeros(n_days)
        for t in range(start, n_days - 1):
            sig = signal_fn(t)
            if not np.isfinite(sig):
                continue
            new_pos = np.tanh(sig) * CAPS[0]
            cost = abs(new_pos - pos) * COMM[0]
            pos = new_pos
            pnl[t + 1] = pos * rets[0, t] - cost
        return pnl

    def score_segs(pnl, label):
        print(f"  {label}:")
        for name, sl in segs:
            p = pnl[sl]
            mu, sd = p.mean(), p.std()
            sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0
            print(f"    {name:<18} mu ${mu:>7,.0f}  sd ${sd:>6,.0f}  "
                  f"SR {sr:>5.2f}  Score {objective(p):>7,.0f}")

    # Strategy A: basket(t-1) -> ALGO(t)
    vol_mkt = np.maximum(
        np.array([basket_ew[max(0, t-60):t].std() for t in range(n_days-1)]),
        1e-8)
    score_segs(run_algo_strat(
        lambda t: basket_ew[t-1] / vol_mkt[t-1] if t > 0 else 0.0),
        "basket(t-1) -> long/short ALGO(t)")

    # Strategy B: ALGO(t-1) -> ALGO(t)
    vol_a = np.maximum(
        np.array([a[max(0, t-60):t].std() for t in range(n_days-1)]),
        1e-8)
    score_segs(run_algo_strat(
        lambda t: a[t-1] / vol_a[t-1] if t > 0 else 0.0),
        "ALGO(t-1) -> ALGO(t) [momentum]")

    # Strategy C: top constituent lag (best from section 3)
    best_c, best_i = lag_corrs[0]
    best_ri = rets[best_i]
    vol_bi = np.maximum(
        np.array([best_ri[max(0, t-60):t].std() for t in range(n_days-1)]),
        1e-8)
    score_segs(run_algo_strat(
        lambda t: np.sign(best_c) * best_ri[t-1] / vol_bi[t-1] if t > 0 else 0.0),
        f"{tickers[best_i]}(t-1) -> ALGO(t) [top constituent]")

    # Strategy D: resid autocorr (fade the residual)
    score_segs(run_algo_strat(
        lambda t: -resid[t-1] / max(np.nanstd(resid[max(0,t-60):t]), 1e-8)
        if t > WFIT and np.isfinite(resid[t-1]) else 0.0),
        "fade ALGO residual(t-1)")

    if n_days > 800:
        print(f"\n  LEADERBOARD[750-{n_days}] scores:")
        for label, fn in [
            ("basket lag", lambda t: basket_ew[t-1] / vol_mkt[t-1]),
            ("ALGO lag", lambda t: a[t-1] / vol_a[t-1]),
        ]:
            pnl = run_algo_strat(fn)
            p = pnl[750:]
            mu, sd = p.mean(), p.std()
            sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0
            print(f"    {label:<14} mu ${mu:>7,.0f}  Score {objective(p):>7,.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python algo_residual.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
