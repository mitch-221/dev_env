"""
Range-bound asset detection + ALGO residual predictor analysis.

1. RANGE-BOUND TEST
   ADF test on log price level of each asset. Stationary price level
   = range-bound. Also Hurst exponent (H < 0.5 = mean-reverting,
   H > 0.5 = trending, H ~ 0.5 = random walk).

2. PRICE RATIO STATIONARITY
   For each pair where one asset is range-bound, test whether the
   log price *ratio* is stationary (more powerful than single-asset
   ADF). Also tests linear combinations (cointegration).

3. ALGO RESIDUAL PREDICTORS
   The 17% of ALGO variance unexplained by the basket. Tests:
   - Is the residual itself stationary (range-bound)?
   - Which individual assets' price levels predict the residual?
   - Rolling regression: can we predict tomorrow's residual from
     today's asset prices / returns?

4. WALK-FORWARD RESIDUAL MEAN REVERSION
   If the residual is stationary or predictable, simulate trading
   it: long ALGO when residual is below its rolling mean by >1sd,
   short when above, gated on significance.

Usage:
    python range_analysis.py prices.txt
"""

import sys
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM = np.full(N_INST, 0.0001)
COMM[0] = 0.00002
WFIT = 150
WARM = WFIT + 20


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


def hurst(ts, max_lag=20):
    """Hurst exponent via R/S analysis."""
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return h


def ols(X, y, ridge=1e-6):
    if X.ndim == 1:
        X = X[:, None]
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def rolling_algo_resid(prc, wfit=WFIT, refit=5):
    """Walk-forward ALGO residual series."""
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
    return resid


def main(path):
    prc, tickers = load_prices(path)
    n_inst, n_days = prc.shape
    log_p = np.log(prc)
    rets = np.diff(log_p, axis=1)

    d500 = min(500, int(n_days * 2 / 3))
    dB = min(650, n_days - 100)

    # ================================================================
    print("=" * 70)
    print("1. RANGE-BOUND TEST (ADF on log price + Hurst exponent)")
    print("=" * 70)
    print(f"  ADF p < 0.05 => price level is stationary (range-bound)")
    print(f"  Hurst < 0.5 => mean-reverting, > 0.5 => trending, ~0.5 => random")
    print()
    print(f"  {'ticker':<8} {'ADF p':>7} {'Hurst':>7}  verdict")
    print(f"  {'-'*40}")

    stationary = []
    hurst_vals = []
    adf_ps = []
    for i in range(n_inst):
        try:
            p_adf = adfuller(log_p[i], autolag='AIC')[1]
        except Exception:
            p_adf = np.nan
        h = hurst(log_p[i])
        hurst_vals.append(h)
        adf_ps.append(p_adf)
        if p_adf < 0.05:
            stationary.append(i)
            verdict = "RANGE-BOUND ***"
        elif p_adf < 0.10:
            verdict = "weak range-bound"
        elif h < 0.4:
            verdict = "mean-reverting (returns)"
        elif h > 0.6:
            verdict = "trending"
        else:
            verdict = "random walk"
        print(f"  {tickers[i]:<8} {p_adf:>7.3f} {h:>7.3f}  {verdict}")

    print(f"\n  Stationary (range-bound) assets: {len(stationary)}")
    print(f"  Mean Hurst across all: {np.mean(hurst_vals):.3f}")

    # ================================================================
    print()
    print("=" * 70)
    print("2. PRICE RATIO / COINTEGRATION TESTS")
    print("=" * 70)
    print("  Testing log price ratios for stationarity.")
    print("  A stationary ratio means two assets move together in level.")
    print()

    # Test all pairs involving stationary assets or ALGO
    test_set = list(set(stationary + [0]))
    results = []
    for i in test_set:
        for j in range(n_inst):
            if j <= i:
                continue
            try:
                # Engle-Granger cointegration test
                _, p_coint, _ = coint(log_p[i], log_p[j])
                # Simple ratio stationarity
                ratio = log_p[i] - log_p[j]
                p_ratio = adfuller(ratio, autolag='AIC')[1]
                results.append((min(p_coint, p_ratio), i, j,
                                p_coint, p_ratio))
            except Exception:
                pass
    results.sort()
    if results:
        print(f"  {'pair':<14} {'coint p':>9} {'ratio p':>9}  flag")
        print(f"  {'-'*50}")
        for best_p, i, j, pc, pr in results[:20]:
            flag = " ***" if best_p < 0.05 else (" *" if best_p < 0.10 else "")
            print(f"  {tickers[i]}/{tickers[j]:<8} {pc:>9.3f} {pr:>9.3f}{flag}")
    else:
        print("  No pairs tested (no stationary assets or ALGO found).")

    # ================================================================
    print()
    print("=" * 70)
    print("3. ALGO RESIDUAL ANALYSIS")
    print("=" * 70)

    resid = rolling_algo_resid(prc)
    r = resid[WARM:]
    r = r[np.isfinite(r)]

    print(f"  Residual vol: {r.std():.5f}/day  (${r.std()*CAPS[0]:,.0f} at $100k)")

    # Is the residual itself stationary?
    try:
        p_adf_r = adfuller(r, autolag='AIC')[1]
        h_r = hurst(r)
        print(f"  Residual ADF p: {p_adf_r:.4f}  Hurst: {h_r:.3f}")
        if p_adf_r < 0.05:
            print("  >>> RESIDUAL IS STATIONARY - mean reverts in level! <<<")
        elif h_r < 0.45:
            print("  Residual mean-reverts in returns (consistent with our finding)")
        else:
            print("  Residual is a random walk - no structural edge here")
    except Exception as e:
        print(f"  ADF failed: {e}")

    # Which asset LEVELS predict the residual?
    print()
    print("  Which log-price levels best predict the ALGO residual?")
    print(f"  {'ticker':<8} {'corr(level,resid)':>18} {'corr(lag1,resid)':>18}")
    print(f"  {'-'*50}")
    level_corrs = []
    for i in range(1, n_inst):
        lp_i = log_p[i, WARM:-1]       # align with resid
        res_i = resid[WARM:]
        res_i = res_i[:len(lp_i)]
        ok = np.isfinite(res_i)
        if ok.sum() < 30:
            continue
        c_now = np.corrcoef(lp_i[ok], res_i[ok])[0, 1]
        # lagged: does asset i level today predict residual tomorrow?
        res_lag = resid[WARM + 1:]
        res_lag = res_lag[:len(lp_i) - 1]
        ok2 = np.isfinite(res_lag)
        c_lag = np.corrcoef(lp_i[:-1][ok2], res_lag[ok2])[0, 1]
        level_corrs.append((abs(c_now) + abs(c_lag), i, c_now, c_lag))
    level_corrs.sort(reverse=True)
    for _, i, c_now, c_lag in level_corrs[:15]:
        flag = " *" if max(abs(c_now), abs(c_lag)) > 0.3 else ""
        print(f"  {tickers[i]:<8} {c_now:>+18.4f} {c_lag:>+18.4f}{flag}")

    # ================================================================
    print()
    print("=" * 70)
    print("4. WALK-FORWARD: TRADE ALGO ON ITS OWN RESIDUAL MEAN REVERSION")
    print("=" * 70)

    # Roll a 60-day window; trade when residual > 1 std from mean
    WIN = 60
    pos = 0.0
    pnl = np.zeros(n_days)
    for t in range(WARM + WIN, n_days - 1):
        win = resid[t - WIN:t]
        win = win[np.isfinite(win)]
        if len(win) < 10:
            continue
        r_now = resid[t]
        if not np.isfinite(r_now):
            continue
        mu_w, sd_w = win.mean(), win.std()
        if sd_w < 1e-12:
            continue
        z = (r_now - mu_w) / sd_w
        # fade: if residual is high, expect ALGO to underperform tomorrow
        new_pos = -np.tanh(2.0 * z) * CAPS[0]
        cost = abs(new_pos - pos) * COMM[0]
        pos = new_pos
        pnl[t + 1] = pos * rets[0, t] - cost

    start = WARM + WIN
    segs = [(f"A[{start}-{d500}]", pnl[start:d500]),
            (f"B[{d500}-{dB}]",   pnl[d500:dB]),
            (f"C[{dB}-{n_days}]", pnl[dB:n_days]),
            ("full",              pnl[start:])]
    if n_days > 800:
        segs.append((f"LEADERBOARD[750-{n_days}]", pnl[750:]))
    for name, p in segs:
        mu, sd = p.mean(), p.std()
        sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0
        print(f"  {name:<26} mu ${mu:>6,.0f}  sd ${sd:>6,.0f}  "
              f"SR {sr:>5.2f}  Score {objective(p):>6,.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python range_analysis.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
