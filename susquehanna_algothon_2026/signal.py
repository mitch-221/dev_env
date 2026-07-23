"""
Signal family scanner - tests structurally different signal families
walk-forward under the competition objective, with an automatic noise
benchmark so results can be judged against pure luck.

Families tested (all walk-forward, no future data):
  xs_rev_5 / xs_rev_10   cross-sectional reversion: short recent
                         winners vs universe, long recent losers
  factor_resid           PCA factor-residual reversion: fit top-k
                         factors on trailing window, revert residuals
                         (invisible to pairwise tests)
  ts_mom_60 / ts_mom_120 long-horizon time-series momentum
  mkt_leadlag            yesterday's universe return -> today's assets
  algo_leadlag           yesterday's basket return -> ALGO position
  own_rev_3              short-horizon own reversion (known-dead
                         baseline; should score ~0)

Protocol:
  - Signals sized to caps via tanh, commissions charged.
  - Train = first 60% of days, holdout = last 40%.
  - Each family also run on 3 pure random-walk datasets; the noise
    band (max train score across null runs) is printed next to real
    results. A family is interesting ONLY if its train score clears
    the noise band AND its holdout confirms.

Usage:
    python signal_scan.py prices.txt
"""

import sys
import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM = np.full(N_INST, 0.0001)
COMM[0] = 0.00002
TRAIN_FRAC = 0.6
WARM = 130


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
    return arr


def objective(pnl):
    mu, sd = pnl.mean(), pnl.std()
    if mu < 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def simulate(prc, signal_fn, start=WARM):
    """Run a signal function walk-forward. signal_fn(rets, t) returns a
    51-vector of desired exposures in [-1, 1] using data up to day t
    inclusive; position held over day t -> t+1."""
    n_inst, n_days = prc.shape
    rets = np.diff(np.log(prc), axis=1)  # rets[:, t] = day t-1 -> t
    pos = np.zeros(n_inst)
    pnl = np.zeros(n_days)
    for t in range(start, n_days - 1):
        expo = np.clip(signal_fn(rets[:, :t], t), -1, 1)
        target = expo * CAPS
        turn = np.abs(target - pos)
        cost = (turn * COMM).sum()
        pos = target
        r_next = rets[:, t]  # log-return day t -> t+1
        pnl[t + 1] = (pos * r_next).sum() - cost
    return pnl


def zscore_xs(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def make_families():
    fams = {}

    def xs_rev(k):
        def f(rets, t):
            recent = rets[:, -k:].sum(axis=1)
            return np.tanh(-zscore_xs(recent))
        return f
    fams["xs_rev_5"] = xs_rev(5)
    fams["xs_rev_10"] = xs_rev(10)

    def factor_resid(rets, t, state={}):
        # refit PCA every 20 days on trailing 250d
        if t % 20 == 0 or "V" not in state:
            R = rets[:, -250:]
            Rc = R - R.mean(axis=1, keepdims=True)
            C = np.cov(Rc)
            w, V = np.linalg.eigh(C)
            state["V"] = V[:, -3:]           # top 3 factors
        V = state["V"]
        r5 = rets[:, -5:].sum(axis=1)
        resid = r5 - V @ (V.T @ r5)
        return np.tanh(-zscore_xs(resid))
    fams["factor_resid"] = factor_resid

    def ts_mom(k):
        def f(rets, t):
            m = rets[:, -k:].sum(axis=1)
            vol = rets[:, -60:].std(axis=1) * np.sqrt(k)
            return np.tanh(m / np.maximum(vol, 1e-8))
        return f
    fams["ts_mom_60"] = ts_mom(60)
    fams["ts_mom_120"] = ts_mom(120)

    def mkt_leadlag(rets, t):
        mkt_yday = rets[:, -1].mean()
        sgn = np.tanh(mkt_yday / max(rets.mean(axis=0)[-60:].std(), 1e-8))
        return np.full(N_INST, sgn)
    fams["mkt_leadlag"] = mkt_leadlag

    def algo_leadlag(rets, t):
        basket_yday = rets[1:, -1].mean()
        expo = np.zeros(N_INST)
        expo[0] = np.tanh(basket_yday /
                          max(rets[1:].mean(axis=0)[-60:].std(), 1e-8))
        return expo
    fams["algo_leadlag"] = algo_leadlag

    def own_rev(rets, t):
        r3 = rets[:, -3:].sum(axis=1)
        vol = rets[:, -30:].std(axis=1) * np.sqrt(3)
        return np.tanh(-r3 / np.maximum(vol, 1e-8))
    fams["own_rev_3"] = own_rev

    return fams


def null_prices(seed, n_days):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.008, 0.025, N_INST)
    rets = rng.normal(0, vols[:, None], (N_INST, n_days - 1))
    p0 = np.log(rng.uniform(20, 150, N_INST))[:, None]
    return np.exp(np.cumsum(np.hstack([p0, rets]), axis=1))


def evaluate(prc, fams):
    n_days = prc.shape[1]
    split = int(n_days * TRAIN_FRAC)
    out = {}
    for name, fn in fams.items():
        pnl = simulate(prc, fn)
        out[name] = (objective(pnl[WARM:split]), objective(pnl[split:]),
                     np.sqrt(250) * pnl[split:].mean() /
                     max(pnl[split:].std(), 1e-9))
    return out


def main(path):
    prc = load_prices(path)
    n_days = prc.shape[1]
    fams = make_families()

    print("Scanning signal families on real data...")
    real = evaluate(prc, fams)

    print("Building noise benchmark (3 random-walk universes)...")
    null_tr = {name: [] for name in fams}
    for seed in range(3):
        nres = evaluate(null_prices(seed + 7, n_days), fams)
        for name, (tr, ho, sr) in nres.items():
            null_tr[name].append(tr)

    print()
    print(f"{'family':<15}{'train Sc':>10}{'holdout Sc':>12}"
          f"{'holdout SR':>12}{'noise max tr':>14}  verdict")
    print("-" * 76)
    for name, (tr, ho, sr) in sorted(real.items(), key=lambda kv: -kv[1][1]):
        nmax = max(null_tr[name])
        if tr > max(2 * nmax, 50) and ho > 0.5 * tr:
            verdict = "REAL - investigate"
        elif tr > nmax and ho > 0:
            verdict = "weak - maybe"
        else:
            verdict = "noise"
        print(f"{name:<15}{tr:>10,.0f}{ho:>12,.0f}{sr:>12.2f}"
              f"{nmax:>14,.0f}  {verdict}")

    print("\nRead: a family matters only if train clears the noise band")
    print("AND holdout confirms. 'holdout SR' is the honest Sharpe.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python signal_scan.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
