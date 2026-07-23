"""
Pairs portfolio builder for the Algothon objective function.

    Score = mu * SR^2/(SR^2+1)   if mu >= 0   (SR = sqrt(250)*mu/sigma)
          = mu                    if mu < 0

Key consequences (baked into this tool):
  - Sharpe is scale-invariant, mu is not => always size to the caps.
  - SR^2/(SR^2+1) saturates fast (0.9 at SR=3) => once SR ~3, only mu
    matters; adding more (uncorrelated) pairs adds mu AND diversifies.
  - Selection is done on the TRAIN segment; the holdout score is
    reported honestly and never used for selection.

Method:
  1. Backtest every pair with the z-score reversion rule, each sized to
     fully use position caps at max z (asset caps: $100k ALGO, $10k rest).
  2. Greedy portfolio construction: repeatedly add the pair that most
     improves the TRAIN objective score, requiring asset-disjointness
     (each asset in at most one pair) so caps are respected by design.
  3. Report train/holdout/full mu, sigma, SR and Score for the chosen
     set, plus a paste-ready Python list for the final strategy.

Usage:
    python portfolio_builder.py prices.txt [max_pairs]

Outputs:
    chosen_pairs.txt   - human-readable report
    pairs_config.py    - paste-ready config for teamName.py
"""

import sys
import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM = np.full(N_INST, 0.0001)
COMM[0] = 0.00002

Z_WINDOW = 30
Z_CAP = 2.0
Z_BAND = 0.25
TRAIN_FRAC = 0.6


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


def objective(pnl):
    """The competition's piecewise score on a daily PnL array."""
    mu = pnl.mean()
    sd = pnl.std()
    if mu < 0 or sd < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def stats(pnl):
    mu, sd = pnl.mean(), pnl.std()
    sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0.0
    return mu, sd, sr, objective(pnl)


W_BETA = 120           # rolling window for hedge-ratio estimation
MIN_PAIR_DOLLARS = 2_000  # skip pairs whose remaining capacity is dust


def _rolling_stats(x, y, w):
    """Rolling mean/var/cov over trailing w points, aligned so index t
    uses points [t-w, t). O(T) via cumulative sums."""
    n = len(x)
    cx = np.concatenate(([0.0], np.cumsum(x)))
    cy = np.concatenate(([0.0], np.cumsum(y)))
    cxx = np.concatenate(([0.0], np.cumsum(x * x)))
    cyy = np.concatenate(([0.0], np.cumsum(y * y)))
    cxy = np.concatenate(([0.0], np.cumsum(x * y)))
    mx = np.full(n, np.nan); my = np.full(n, np.nan)
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    cvxy = np.full(n, np.nan)
    t = np.arange(w, n)
    mx[t] = (cx[t] - cx[t - w]) / w
    my[t] = (cy[t] - cy[t - w]) / w
    vx[t] = (cxx[t] - cxx[t - w]) / w - mx[t] ** 2
    vy[t] = (cyy[t] - cyy[t - w]) / w - my[t] ** 2
    cvxy[t] = (cxy[t] - cxy[t - w]) / w - mx[t] * my[t]
    return mx, my, vx, vy, cvxy


def pair_pnl(log_p, i, j):
    """Walk-forward daily $ PnL of the z-score rule on pair (i, j) at
    UNIT size (S = 1 dollar per unit of z). Scale linearly later.

    Beta is re-estimated every day from the trailing W_BETA window;
    the z-score uses the trailing Z_WINDOW with the current beta.
    No future data is used anywhere, so the whole PnL series is
    out-of-sample by construction.
    Returns (pnl_unit, med_abs_beta) or None.
    """
    x, y = log_p[j], log_p[i]
    n = len(x)
    start = max(W_BETA, Z_WINDOW)
    if n <= start + 10:
        return None

    # rolling beta from W_BETA window
    _, _, vxB, _, cvB = _rolling_stats(x, y, W_BETA)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_t = cvB / vxB
    beta_t = np.where(np.isfinite(beta_t), beta_t, 0.0)

    # rolling moments over Z_WINDOW for z-score of s = y - beta_t * x
    mxZ, myZ, vxZ, vyZ, cvZ = _rolling_stats(x, y, Z_WINDOW)

    pos = 0.0
    prev_hj = 0.0
    betas = []
    pnl = np.zeros(n)
    for t in range(start, n - 1):
        b = beta_t[t]
        betas.append(abs(b))
        s_now = y[t] - b * x[t]
        s_mean = myZ[t] - b * mxZ[t]
        s_var = vyZ[t] + b * b * vxZ[t] - 2 * b * cvZ[t]
        if not np.isfinite(s_var) or s_var <= 1e-14:
            continue
        z = (s_now - s_mean) / np.sqrt(s_var)

        if abs(z - (-pos)) > Z_BAND:
            new_pos = -np.clip(z, -Z_CAP, Z_CAP)
        else:
            new_pos = pos
        # dollar legs at unit size: h_i = -pos, h_j = +pos*b
        hi, hj = -new_pos, new_pos * b
        cost = abs(-new_pos - (-pos)) * COMM[i] + abs(hj - prev_hj) * COMM[j]
        prev_hj = hj
        pos = new_pos
        ds = (y[t + 1] - y[t]) - b * (x[t + 1] - x[t])
        pnl[t + 1] = pos * ds - cost
    med_ab = float(np.median(betas)) if betas else 0.0
    return pnl, max(med_ab, 1e-6)


def main(path, max_pairs=40):
    prc, tickers = load_prices(path)
    n_inst, n_days = prc.shape
    log_p = np.log(prc)
    split = int(n_days * TRAIN_FRAC)
    warm = max(W_BETA, Z_WINDOW) + 1
    half = (warm + split) // 2   # midpoint of usable train segment

    print(f"Backtesting all pairs ({n_inst * (n_inst - 1) // 2}) "
          f"walk-forward (rolling beta {W_BETA}d)...")
    candidates = []  # (train_score, i, j, med_ab, pnl_unit)
    n_stable = 0
    for i in range(n_inst):
        for j in range(i + 1, n_inst):
            r = pair_pnl(log_p, i, j)
            if r is None:
                continue
            pnl, med_ab = r
            # stability screen: positive objective in BOTH train halves
            s1 = objective(pnl[warm:half])
            s2 = objective(pnl[half:split])
            if s1 <= 0 or s2 <= 0:
                continue
            n_stable += 1
            tr = objective(pnl[warm:split])
            candidates.append((tr, i, j, med_ab, pnl))
    candidates.sort(key=lambda c: -c[0])
    print(f"Candidates passing stability screen "
          f"(positive score in both train halves): {n_stable}")

    # ---- greedy selection on TRAIN objective with capacity model ----
    # Each asset has dollar capacity; a pair consumes S*Z_CAP from leg i
    # and S*Z_CAP*|beta| from leg j. ALGO's $100k = room for ~10 pairs.
    rem = CAPS.copy()
    chosen = []          # (S, train_score_solo, i, j, med_ab, pnl_unit)
    port = np.zeros(n_days)
    best_score = 0.0
    improved = True
    while improved and len(chosen) < max_pairs:
        improved = False
        best_add = None
        for c in candidates:
            tr, i, j, med_ab, pnl = c
            S = min(rem[i], rem[j] / med_ab) / Z_CAP
            if S * Z_CAP < MIN_PAIR_DOLLARS:
                continue
            s = objective((port + S * pnl)[warm:split])
            if s > best_score and (best_add is None or s > best_add[0]):
                best_add = (s, S, c)
        if best_add is not None:
            best_score, S, c = best_add
            tr, i, j, med_ab, pnl = c
            chosen.append((S, tr, i, j, med_ab, pnl))
            rem[i] -= S * Z_CAP
            rem[j] -= S * Z_CAP * med_ab
            port = port + S * pnl
            improved = True
            print(f"  + {tickers[i]}/{tickers[j]:<6} S=${S:>7,.0f} "
                  f"train score -> {best_score:,.0f}")

    # ---- honest evaluation ----
    seg = {
        "train": port[warm:split],
        "holdout": port[split:],
        "full": port[warm:],
    }
    print("\nPortfolio evaluation (holdout untouched by selection):")
    for name, p in seg.items():
        mu, sd, sr, sc = stats(p)
        print(f"  {name:<8} mu ${mu:>8,.0f}/day  sigma ${sd:>8,.0f}  "
              f"SR {sr:>5.2f}  Score {sc:>8,.0f}")

    max_deploy = sum(Z_CAP * S * (1 + ab) for S, _, _, _, ab, _ in chosen)
    algo_pairs = sum(1 for S, _, i, j, _, _ in chosen if i == 0 or j == 0)
    print(f"\nPairs chosen: {len(chosen)} ({algo_pairs} involving ALGO)   "
          f"max gross deployment: ${max_deploy:,.0f}")
    print(f"ALGO capacity used: ${CAPS[0] - rem[0]:,.0f} / $100,000")

    # ---- reports ----
    with open("chosen_pairs.txt", "w") as f:
        f.write("CHOSEN PAIRS PORTFOLIO (walk-forward, capacity model)\n")
        f.write("=" * 78 + "\n")
        f.write(f"Selected on train days {warm}-{split}; holdout "
                f"{split}-{n_days} untouched by selection.\n")
        f.write("Betas are rolling (re-fit daily from trailing "
                f"{W_BETA}d) - no staleness.\n\n")
        f.write(f"{'pair':<14}{'idx':>9}{'med|b|':>8}{'unit $S':>10}"
                f"{'train Sc':>10}{'holdout Sc':>11}\n")
        f.write("-" * 78 + "\n")
        for S, tr, i, j, ab, pnl in chosen:
            ho = objective(S * pnl[split:])
            f.write(f"{tickers[i]}/{tickers[j]:<8}{f'({i},{j})':>9}"
                    f"{ab:>8.3f}{S:>10,.0f}{S/1*tr:>10,.0f}{ho:>11,.0f}\n")
        f.write("\nPortfolio:\n")
        for name, p in seg.items():
            mu, sd, sr, sc = stats(p)
            f.write(f"  {name:<8} mu ${mu:>8,.0f}/day  sigma ${sd:>8,.0f}  "
                    f"SR {sr:>5.2f}  Score {sc:>8,.0f}\n")
        f.write(f"\nMax gross deployment at |z|={Z_CAP}: ${max_deploy:,.0f}\n")

    with open("pairs_config.py", "w") as f:
        f.write("# Auto-generated by portfolio_builder.py\n")
        f.write("# (i, j, size_S)  - live strategy re-estimates beta on a\n")
        f.write(f"# rolling {W_BETA}d window, z on {Z_WINDOW}d, matching "
                "this simulation.\n")
        f.write("PAIRS = [\n")
        for S, tr, i, j, ab, pnl in chosen:
            f.write(f"    ({i}, {j}, {S:.0f}),   "
                    f"# {tickers[i]}/{tickers[j]}, med|beta| {ab:.3f}\n")
        f.write("]\n")
        f.write(f"W_BETA = {W_BETA}\nZ_WINDOW = {Z_WINDOW}\n"
                f"Z_CAP = {Z_CAP}\nZ_BAND = {Z_BAND}\n")

    print("\nWrote: chosen_pairs.txt")
    print("Wrote: pairs_config.py  (paste-ready for the strategy)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python portfolio_builder.py prices.txt [max_pairs]")
        sys.exit(1)
    mp = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    main(sys.argv[1], mp)
