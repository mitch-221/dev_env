"""
Deep-dive diagnostics. Usage: python deep_dive.py prices.txt

Sections:
  1. REGIME MAP     rolling-window scores for each family; explicit
                    pre/post day-500 comparison (dataset splice point)
  2. DRIFT          per-asset mean-return t-stats + walk-forward
                    drift-hold strategy score
  3. FACTOR SWEEP   factor_resid parameter grid, scored on recent
                    segments (tune on B, validate on C)
  4. XS_REV SWEEP   cross-sectional reversion grid, same protocol
  5. MARKET         rolling AR(1) of the equal-weight market; market
                    momentum vs reversal by period
  6. SEASONALITY    day-index mod-m patterns + FFT top periods
  7. ALGO WORKUP    drift, autocorr, variance ratios, conditional
                    effects, three simple ALGO strategies scored
  8. THRESHOLD      next-day behavior after >2-sigma moves

Segments: A = pre-500, B = days 500-650 (tune), C = days 650-750
(validate). The hidden leaderboard window continues AFTER day 750,
so B/C consistency matters more than anything measured on A.
"""

import sys
import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM = np.full(N_INST, 0.0001)
COMM[0] = 0.00002
WARM = 130

_report = []


def P(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    _report.append(line)


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


def zscore_xs(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def simulate(prc, signal_fn, start=WARM):
    n_inst, n_days = prc.shape
    rets = np.diff(np.log(prc), axis=1)
    pos = np.zeros(n_inst)
    pnl = np.zeros(n_days)
    for t in range(start, n_days - 1):
        expo = np.clip(signal_fn(rets[:, :t], t), -1, 1)
        target = expo * CAPS
        cost = (np.abs(target - pos) * COMM).sum()
        pos = target
        pnl[t + 1] = (pos * rets[:, t]).sum() - cost
    return pnl


# ---------------- signal families (parameterized) ----------------

def make_xs_rev(k, mode="z"):
    def f(rets, t):
        recent = rets[:, -k:].sum(axis=1)
        if mode == "rank":
            r = recent.argsort().argsort().astype(float)
            sig = (r - r.mean()) / r.std()
        else:
            sig = zscore_xs(recent)
        return np.tanh(-sig)
    return f


def make_factor_resid(nf, h, fitw):
    state = {}
    def f(rets, t):
        if t % 20 == 0 or "V" not in state:
            R = rets[:, -fitw:]
            Rc = R - R.mean(axis=1, keepdims=True)
            w, V = np.linalg.eigh(np.cov(Rc))
            state["V"] = V[:, -nf:]
        V = state["V"]
        rh = rets[:, -h:].sum(axis=1)
        resid = rh - V @ (V.T @ rh)
        return np.tanh(-zscore_xs(resid))
    return f


def make_ts_mom(k):
    def f(rets, t):
        m = rets[:, -k:].sum(axis=1)
        vol = rets[:, -60:].std(axis=1) * np.sqrt(k)
        return np.tanh(m / np.maximum(vol, 1e-8))
    return f


def make_mkt_lag(sign=+1):
    def f(rets, t):
        mkt = rets.mean(axis=0)
        s = np.tanh(sign * mkt[-1] / max(mkt[-60:].std(), 1e-8))
        return np.full(N_INST, s)
    return f


def seg_scores(pnl, segs):
    return [objective(pnl[a:b]) for a, b in segs]


def main(path):
    prc, tickers = load_prices(path)
    n_inst, n_days = prc.shape
    rets = np.diff(np.log(prc), axis=1)
    mkt = rets.mean(axis=0)

    # segments (adapt if file shorter/longer than 750)
    d500 = min(500, int(n_days * 2 / 3))
    dB = min(650, n_days - 100)
    segs = [(WARM, d500), (d500, dB), (dB, n_days)]
    seg_names = [f"A[{WARM}-{d500}]", f"B[{d500}-{dB}]", f"C[{dB}-{n_days}]"]
    P(f"Days: {n_days}.  Segments: {', '.join(seg_names)}")
    P("B and C are what matter - the hidden window continues after "
      f"day {n_days}.")

    # ============ 1. REGIME MAP ============
    P("\n" + "=" * 70)
    P("1. REGIME MAP - rolling 100d scores per family")
    P("=" * 70)
    fams = {
        "xs_rev_5": make_xs_rev(5),
        "factor_resid(3,5,250)": make_factor_resid(3, 5, 250),
        "ts_mom_60": make_ts_mom(60),
        "mkt_lag+1": make_mkt_lag(+1),
        "mkt_lag-1": make_mkt_lag(-1),
    }
    windows = [(s, min(s + 100, n_days)) for s in range(WARM, n_days - 50, 100)]
    hdr = "".join(f"{f'{a}-{b}':>12}" for a, b in windows)
    P(f"{'family':<22}{hdr}")
    for name, fn in fams.items():
        pnl = simulate(prc, fn)
        row = "".join(f"{objective(pnl[a:b]):>12,.0f}" for a, b in windows)
        P(f"{name:<22}{row}")
    P("Look for sign flips around day 500 = regime change at the splice.")

    # ============ 2. DRIFT ============
    P("\n" + "=" * 70)
    P("2. DRIFT - per-asset mean return t-stats")
    P("=" * 70)
    for label, sl in (("full", slice(None)), (f"post-{d500}", slice(d500, None))):
        r = rets[:, sl]
        t_ = r.mean(axis=1) / (r.std(axis=1) / np.sqrt(r.shape[1]))
        sig2 = (np.abs(t_) > 2).sum()
        sig3 = (np.abs(t_) > 3).sum()
        exp2 = int(0.046 * n_inst)
        P(f"{label:>10}: |t|>2: {sig2}/{n_inst} (chance ~{exp2}), "
          f"|t|>3: {sig3}, max |t|: {np.abs(t_).max():.2f} "
          f"({tickers[np.abs(t_).argmax()]})")
    # walk-forward drift-hold: sign of trailing 250d mean if |t|>1.5
    def drift_hold(rr, t):
        w = rr[:, -250:]
        tt = w.mean(axis=1) / (w.std(axis=1) / np.sqrt(w.shape[1]) + 1e-12)
        return np.where(np.abs(tt) > 1.5, np.sign(tt), 0.0)
    pnl = simulate(prc, drift_hold, start=max(WARM, 250))
    P(f"Walk-forward drift-hold scores  "
      + "  ".join(f"{n}: {s:,.0f}" for n, s in
                  zip(seg_names, seg_scores(pnl, segs))))

    # ============ 3. FACTOR SWEEP ============
    P("\n" + "=" * 70)
    P("3. FACTOR_RESID SWEEP (scored on B=tune / C=validate)")
    P("=" * 70)
    P(f"{'nf':>3}{'h':>4}{'fitw':>6}{'A':>9}{'B':>9}{'C':>9}  flag")
    best = []
    for nf in (1, 2, 3, 5):
        for h in (1, 3, 5, 10):
            for fitw in (120, 250):
                pnl = simulate(prc, make_factor_resid(nf, h, fitw))
                a, b, c = seg_scores(pnl, segs)
                flag = "<< B&C pos" if b > 0 and c > 0 else ""
                if b > 0 and c > 0:
                    best.append((min(b, c), nf, h, fitw, a, b, c))
                P(f"{nf:>3}{h:>4}{fitw:>6}{a:>9,.0f}{b:>9,.0f}{c:>9,.0f}  {flag}")
    if best:
        best.sort(reverse=True)
        _, nf, h, fitw, a, b, c = best[0]
        P(f"Best stable config: nf={nf} h={h} fitw={fitw} "
          f"(A {a:,.0f} | B {b:,.0f} | C {c:,.0f})")

    # ============ 4. XS_REV SWEEP ============
    P("\n" + "=" * 70)
    P("4. XS_REV SWEEP")
    P("=" * 70)
    P(f"{'k':>3}{'mode':>6}{'A':>9}{'B':>9}{'C':>9}  flag")
    for k in (1, 2, 3, 5, 10, 20):
        for mode in ("z", "rank"):
            pnl = simulate(prc, make_xs_rev(k, mode))
            a, b, c = seg_scores(pnl, segs)
            flag = "<< B&C pos" if b > 0 and c > 0 else ""
            P(f"{k:>3}{mode:>6}{a:>9,.0f}{b:>9,.0f}{c:>9,.0f}  {flag}")

    # ============ 5. MARKET ============
    P("\n" + "=" * 70)
    P("5. MARKET (equal-weight) - rolling AR(1)")
    P("=" * 70)
    for s in range(WARM, len(mkt) - 100, 100):
        w = mkt[s:s + 100]
        ac = np.corrcoef(w[:-1], w[1:])[0, 1]
        tstat = ac * np.sqrt(len(w) - 1)
        P(f"  days {s:>4}-{s+100:<4} AR(1) {ac:+.3f}  (t {tstat:+.1f})")
    P("Persistent sign = market-level momentum(+) / reversal(-) to trade.")

    # ============ 6. SEASONALITY ============
    P("\n" + "=" * 70)
    P("6. SEASONALITY")
    P("=" * 70)
    for m in (2, 3, 4, 5, 7, 10):
        means, ts = [], []
        for k in range(m):
            b = mkt[np.arange(len(mkt)) % m == k]
            means.append(b.mean())
            ts.append(b.mean() / (b.std() / np.sqrt(len(b))))
        mx = np.abs(ts).max()
        note = "  <-- check" if mx > 2.5 else ""
        P(f"  mod {m:>2}: max |t| {mx:.2f}{note}")
    spec = np.abs(np.fft.rfft(mkt - mkt.mean()))
    freqs = np.fft.rfftfreq(len(mkt))
    top = spec[1:].argsort()[::-1][:5] + 1
    P("  FFT top periods (days): "
      + ", ".join(f"{1/freqs[i]:.1f}" for i in top if freqs[i] > 0))
    P("  (isolated strong short periods = cyclical component)")

    # ============ 7. ALGO WORKUP ============
    P("\n" + "=" * 70)
    P("7. ALGO WORKUP")
    P("=" * 70)
    a = rets[0]
    t_drift = a.mean() / (a.std() / np.sqrt(len(a)))
    P(f"  drift t-stat: {t_drift:+.2f}   daily vol: {a.std():.4f}")
    acs = [np.corrcoef(a[:-l], a[l:])[0, 1] for l in range(1, 11)]
    P("  autocorr lags 1-10: " + " ".join(f"{x:+.2f}" for x in acs))
    for q in (2, 5, 10, 20):
        k = len(a) // q * q
        vr = a[:k].reshape(-1, q).sum(axis=1).var() / (q * a.var())
        P(f"  variance ratio q={q:>2}: {vr:.2f} "
          f"({'trending' if vr > 1.15 else 'reverting' if vr < 0.85 else 'random'})")
    up = a[1:][a[:-1] > 0]
    dn = a[1:][a[:-1] < 0]
    P(f"  E[ret | prev up]: {up.mean():+.5f}   "
      f"E[ret | prev down]: {dn.mean():+.5f}")
    big = np.abs(a[:-1]) > 2 * a.std()
    if big.sum() > 5:
        P(f"  E[ret | prev |move|>2sig]: {a[1:][big].mean():+.5f} "
          f"(n={big.sum()})")
    def algo_only(fn):
        def g(rr, t):
            e = np.zeros(N_INST)
            e[0] = fn(rr[0])
            return e
        return g
    strats = {
        "trend20": algo_only(lambda r: np.tanh(np.sign(r[-20:].sum()))),
        "rev1": algo_only(lambda r: -np.tanh(r[-1] / max(r[-30:].std(), 1e-8))),
        "hold+": algo_only(lambda r: 1.0),
    }
    for name, fn in strats.items():
        pnl = simulate(prc, fn)
        P(f"  ALGO {name:<8} " + "  ".join(
            f"{n}: {s:,.0f}" for n, s in zip(seg_names, seg_scores(pnl, segs))))

    # ============ 8. THRESHOLD ============
    P("\n" + "=" * 70)
    P("8. THRESHOLD EFFECTS (all assets pooled)")
    P("=" * 70)
    vol = rets.std(axis=1, keepdims=True)
    zr = rets / vol
    for label, sl in (("full", slice(None)), (f"post-{d500}", slice(d500, None))):
        z = zr[:, sl]
        prev, nxt = z[:, :-1].ravel(), z[:, 1:].ravel()
        for thr in (1.5, 2.0):
            m = np.abs(prev) > thr
            if m.sum() < 30:
                continue
            cont = (np.sign(prev[m]) * nxt[m])
            t = cont.mean() / (cont.std() / np.sqrt(m.sum()))
            direction = "CONTINUATION" if cont.mean() > 0 else "REVERSAL"
            star = " ***" if abs(t) > 3 else (" *" if abs(t) > 2 else "")
            P(f"  {label:>10} after |z|>{thr}: mean signed next-ret "
              f"{cont.mean():+.4f} (t {t:+.1f}, n={m.sum()}) "
              f"{direction}{star}")

    # ============ 9. PER-ASSET FINGERPRINTS ============
    P("\n" + "=" * 70)
    P("9. PER-ASSET FINGERPRINTS - do assets have stable individual")
    P("   character (trend/revert/drift), and does it survive the splice?")
    P("=" * 70)

    def fingerprint(r):
        """per-asset (lag1 autocorr, variance ratio q=10, mean ret)"""
        n = r.shape[1]
        ac = np.array([np.corrcoef(r[i, :-1], r[i, 1:])[0, 1]
                       for i in range(n_inst)])
        k = n // 10 * 10
        vr = np.array([r[i, :k].reshape(-1, 10).sum(axis=1).var() /
                       (10 * r[i].var()) for i in range(n_inst)])
        return ac, vr, r.mean(axis=1)

    ac_pre, vr_pre, mu_pre = fingerprint(rets[:, :d500])
    ac_post, vr_post, mu_post = fingerprint(rets[:, d500:])

    se = 1 / np.sqrt(d500)
    n_rev = (ac_pre < -2 * se).sum()
    n_trd = (ac_pre > 2 * se).sum()
    P(f"  pre-{d500} classification: {n_rev} reverting, {n_trd} trending, "
      f"{n_inst - n_rev - n_trd} random  (|ac| > 2se = {2*se:.3f})")
    se2 = 1 / np.sqrt(n_days - d500)
    n_rev2 = (ac_post < -2 * se2).sum()
    n_trd2 = (ac_post > 2 * se2).sum()
    P(f"  post-{d500} classification: {n_rev2} reverting, {n_trd2} trending, "
      f"{n_inst - n_rev2 - n_trd2} random")

    def pers(x, y, label):
        c = np.corrcoef(x, y)[0, 1]
        t = c * np.sqrt((n_inst - 2) / max(1 - c * c, 1e-9))
        star = " ***" if abs(t) > 3 else (" *" if abs(t) > 2 else "")
        P(f"  persistence of {label:<14} pre-vs-post corr {c:+.3f} "
          f"(t {t:+.1f}){star}")
        return c

    P("  --- persistence across the splice (the decisive stats) ---")
    pers(ac_pre, ac_post, "autocorr")
    pers(vr_pre, vr_post, "variance ratio")
    pers(mu_pre, mu_post, "drift")
    P("  Positive & significant => asset character is a stable property:")
    P("  trade each asset by its own measured character (strategy below).")
    P("  Near zero => per-asset differences are sampling noise; pooled")
    P("  tests were not hiding anything.")

    # walk-forward per-asset adaptive strategy: each asset traded by its
    # own trailing-250d lag-1 autocorr (momentum if +, reversal if -),
    # only where the autocorr is significant.
    def adaptive(rr, t):
        w = rr[:, -250:]
        ac = np.array([np.corrcoef(w[i, :-1], w[i, 1:])[0, 1]
                       for i in range(n_inst)])
        tstat = ac * np.sqrt(w.shape[1])
        vol = w.std(axis=1)
        last = rr[:, -1]
        expo = np.tanh(5 * ac * last / np.maximum(vol, 1e-8))
        return np.where(np.abs(tstat) > 1.5, expo, 0.0)

    pnl = simulate(prc, adaptive, start=max(WARM, 250))
    P("  Walk-forward per-asset-adaptive scores  "
      + "  ".join(f"{n}: {s:,.0f}" for n, s in
                  zip(seg_names, seg_scores(pnl, segs))))

    with open("deep_dive_report.txt", "w") as f:
        f.write("\n".join(_report))
    print("\nWrote: deep_dive_report.txt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python deep_dive.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
