"""
Hedged ALGO index-arbitrage lab.

Construction:
  - Rolling OLS (returns) of ALGO on the 50 basket assets -> weights w
  - Spread s_t = log p_ALGO - sum_i w_i log p_i  (rolling-demeaned)
  - z = zscore of s over trailing ZW days
  - Position: ALGO = -f(z) * $SCALE, basket leg i = +f(z) * SCALE * w_i
    (market moves cancel; PnL = -f(z) * SCALE * d(spread))
  - Basket legs clipped to their $10k caps; commissions charged.

Grid over fit window, z window, and linear-vs-tanh sizing; scored on
A/B/C segments (+ LEADERBOARD segment if the file has >800 days).

Also prints spread diagnostics: level vol, daily reversion (AR1 of
spread), implied half-life - the physics of the trade.

Usage:
    python arb_lab.py prices.txt
"""

import sys
import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM = np.full(N_INST, 0.0001)
COMM[0] = 0.00002
SCALE = 100_000.0
REFIT_EVERY = 5


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


def ols(X, y, ridge=1e-6):
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def run_arb(prc, wfit, zw, sizing="tanh", deadband=0.0):
    n_inst, n_days = prc.shape
    log_p = np.log(prc)
    rets = np.diff(log_p, axis=1)
    start = wfit + zw + 5

    w = None
    last_fit = -10**9
    spread = np.full(n_days, np.nan)
    pos_dollars = np.zeros(n_inst)   # dollar positions
    pnl = np.zeros(n_days)

    for t in range(start, n_days - 1):
        # refit weights on trailing returns
        if w is None or (t - last_fit) >= REFIT_EVERY:
            X = rets[1:, t - wfit:t].T
            y = rets[0, t - wfit:t]
            w = ols(X, y)
            last_fit = t
        # spread history under current weights (rolling window only)
        for k in range(t - zw, t + 1):
            spread[k] = log_p[0, k] - w @ log_p[1:, k]
        s_win = spread[t - zw:t]
        sd = s_win.std()
        if sd < 1e-12:
            continue
        z = (spread[t] - s_win.mean()) / sd

        zz = np.sign(z) * max(abs(z) - deadband, 0.0)
        f = np.tanh(zz) if sizing == "tanh" else np.clip(zz / 2.0, -1, 1)
        # target dollars: ALGO leg and hedge legs
        tgt = np.zeros(n_inst)
        tgt[0] = -f * SCALE
        tgt[1:] = f * SCALE * w
        tgt = np.clip(tgt, -CAPS, CAPS)

        cost = (np.abs(tgt - pos_dollars) * COMM).sum()
        pos_dollars = tgt
        pnl[t + 1] = (pos_dollars * rets[:, t]).sum() - cost
    return pnl, start


def spread_diag(prc, wfit=250):
    log_p = np.log(prc)
    rets = np.diff(log_p, axis=1)
    X = rets[1:, :wfit].T
    y = rets[0, :wfit]
    w = ols(X, y)
    s = log_p[0] - w @ log_p[1:]
    s = s - s.mean()
    ds = np.diff(s)
    ac = np.corrcoef(s[:-1], s[1:])[0, 1]
    hl = -np.log(2) / np.log(max(ac, 1e-6)) if 0 < ac < 1 else np.inf
    print(f"Spread diagnostics (weights fit on first {wfit}d):")
    print(f"  level std: {s.std():.4f} log-pts "
          f"(${s.std()*SCALE:,.0f} on ${SCALE:,.0f})")
    print(f"  daily change std: {ds.std():.4f} (${ds.std()*SCALE:,.0f})")
    print(f"  spread AR(1): {ac:+.3f}  implied half-life: "
          f"{hl:.1f}d" if np.isfinite(hl) else
          f"  spread AR(1): {ac:+.3f}  (not reverting in-sample)")


def main(path):
    prc = load_prices(path)
    n_days = prc.shape[1]
    d500 = min(500, int(n_days * 2 / 3))
    dB = min(650, n_days - 100)

    spread_diag(prc)
    print()

    def segrow(pnl, start):
        segs = [("A", pnl[start:d500]), ("B", pnl[d500:dB]),
                ("C", pnl[dB:n_days]), ("full", pnl[start:])]
        if n_days > 800:
            segs.append(("LB", pnl[750:]))
        out = ""
        for name, p in segs:
            out += f"{name}:{objective(p):>7,.0f}  "
        mu, sd = pnl[start:].mean(), pnl[start:].std()
        sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0
        return out + f"(mu ${mu:,.0f} sd ${sd:,.0f} SR {sr:.2f})"

    print(f"{'config':<26}segment scores")
    print("-" * 100)
    for wfit in (150, 250):
        for zw in (10, 20, 30, 40):
            for db in (0.0, 0.5, 1.0):
                pnl, start = run_arb(prc, wfit, zw, "tanh", db)
                print(f"wfit {wfit} zw {zw:>2} db {db:.1f}   "
                      + segrow(pnl, start))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python arb_lab.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
