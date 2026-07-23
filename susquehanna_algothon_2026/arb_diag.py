"""
Arb activity + spread distribution diagnostic.

Answers: is the arb component actually trading, what does the ALGO
spread look like, and what is the arb alone worth?

  1. WIRING     which teamName_v3 file was imported (full path) and
                its key parameters - catches stale/wrong-module issues
  2. SPREAD     distribution of the ALGO-vs-basket spread: std, range,
                text histogram, AR(1), implied half-life, daily move
                in dollars at $100k
  3. ACTIVITY   arb-only run: first trading day, % days with a
                position, mean/max ALGO position, daily turnover,
                number of ALGO trades
  4. PNL        arb-only segment scores under full eval mechanics

Usage:
    python arb_diag.py prices.txt
"""

import sys
import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM = np.full(N_INST, 0.0001)
COMM[0] = 0.00002


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


def main(path):
    import teamName_v3 as S

    print("=" * 70)
    print("1. WIRING CHECK")
    print("=" * 70)
    print(f"  imported from: {S.__file__}")
    for k in ("ARB_WFIT", "ARB_ZW", "ARB_REFIT", "ARB_SCALE",
              "FITW", "MIN_HISTORY", "W_RESID", "REBAL_BAND"):
        print(f"  {k} = {getattr(S, k)}")

    prc = load_prices(path)
    n_inst, n_days = prc.shape
    log_p = np.log(prc)
    rets = np.diff(log_p, axis=1)

    # ---- 2. spread distribution (standalone, rolling weights) ----
    print()
    print("=" * 70)
    print("2. SPREAD DISTRIBUTION (rolling weights, walk-forward)")
    print("=" * 70)
    wfit, refit = S.ARB_WFIT, S.ARB_REFIT
    start = wfit + 5
    w = None
    last_fit = -10**9
    spread = np.full(n_days, np.nan)
    for t in range(start, n_days):
        if w is None or (t - last_fit) >= refit:
            X = rets[1:, t - wfit:t].T
            y = rets[0, t - wfit:t]
            XtX = X.T @ X + 1e-6 * np.eye(50)
            w = np.linalg.solve(XtX, X.T @ y)
            last_fit = t
        spread[t] = log_p[0, t] - w @ log_p[1:, t]
    s = spread[start:]
    # de-mean with rolling 20d mean (weights drift moves the level)
    sm = np.array([s[max(0, i - 20):i].mean() if i > 5 else np.nan
                   for i in range(len(s))])
    dev = (s - sm)[25:]
    dev = dev[np.isfinite(dev)]
    ds = np.diff(s)
    ds = ds[np.isfinite(ds)]
    print(f"  spread level std (20d-demeaned): {dev.std():.5f} log-pts "
          f"= ${dev.std()*S.ARB_SCALE:,.0f} at ${S.ARB_SCALE:,.0f}")
    print(f"  daily spread move std: {ds.std():.5f} = "
          f"${ds.std()*S.ARB_SCALE:,.0f}/day gross potential")
    ac = np.corrcoef(dev[:-1], dev[1:])[0, 1]
    hl = -np.log(2) / np.log(ac) if 0 < ac < 1 else np.inf
    print(f"  demeaned spread AR(1): {ac:+.3f}   implied half-life: "
          + (f"{hl:.1f}d" if np.isfinite(hl) else "inf (not reverting)"))
    # text histogram of dev in units of its std
    z = dev / dev.std()
    edges = [-np.inf, -2, -1, -0.5, 0.5, 1, 2, np.inf]
    labels = ["<-2s", "-2..-1", "-1..-.5", "-.5...5", ".5..1", "1..2", ">2s"]
    print("  deviation histogram (z units):")
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        cnt = ((z > lo) & (z <= hi)).sum()
        bar = "#" * int(60 * cnt / len(z))
        print(f"    {lab:>8} {cnt:>5}  {bar}")

    # ---- 3+4. arb-only run through eval mechanics ----
    print()
    print("=" * 70)
    print("3. ARB-ONLY ACTIVITY & PNL (W_RESID forced to 0)")
    print("=" * 70)
    saved = S.W_RESID
    S.W_RESID = 0.0
    S._pos = np.zeros(N_INST, dtype=int)
    S._last_n = 0
    S._w = None
    S._V = None

    WARM = 130
    pos = np.zeros(n_inst, dtype=int)
    cash = 0.0
    pnl = np.zeros(n_days)
    value_prev = 0.0
    algo_pos_d = np.zeros(n_days)
    algo_turn = 0.0
    n_algo_trades = 0
    first_trade = None
    for t in range(WARM, n_days):
        p = prc[:, t]
        new_pos = np.asarray(S.getMyPosition(prc[:, :t + 1]), dtype=int)
        mx = np.floor(CAPS / p).astype(int)
        new_pos = np.clip(new_pos, -mx, mx)
        delta = new_pos - pos
        if delta[0] != 0:
            n_algo_trades += 1
            algo_turn += abs(delta[0]) * p[0]
            if first_trade is None:
                first_trade = t
        traded = np.abs(delta) * p
        cash -= (delta * p).sum() + (traded * COMM).sum()
        pos = new_pos
        algo_pos_d[t] = pos[0] * p[0]
        value = cash + (pos * p).sum()
        pnl[t] = value - value_prev
        value_prev = value
    S.W_RESID = saved

    active = np.abs(algo_pos_d[WARM:]) > 1000
    print(f"  first ALGO trade on day: {first_trade}")
    print(f"  days with |ALGO pos| > $1k: {active.sum()} / {n_days - WARM} "
          f"({active.mean():.0%})")
    print(f"  mean |ALGO pos|: ${np.abs(algo_pos_d[WARM:]).mean():,.0f}   "
          f"max: ${np.abs(algo_pos_d).max():,.0f}")
    print(f"  ALGO trades: {n_algo_trades}   total ALGO turnover: "
          f"${algo_turn:,.0f}")

    d500 = min(500, int(n_days * 2 / 3))
    dB = min(650, n_days - 100)
    segs = [(f"A[{WARM}-{d500}]", pnl[WARM:d500]),
            (f"B[{d500}-{dB}]", pnl[d500:dB]),
            (f"C[{dB}-{n_days}]", pnl[dB:]),
            ("full", pnl[WARM:])]
    if n_days > 800:
        segs.append((f"LEADERBOARD[750-{n_days}]", pnl[750:]))
    print("\n  arb-only scores:")
    for name, p in segs:
        mu, sd = p.mean(), p.std()
        sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0
        print(f"    {name:<22} mu ${mu:>7,.0f}  sd ${sd:>7,.0f}  "
              f"SR {sr:>5.2f}  Score {objective(p):>7,.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python arb_diag.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
