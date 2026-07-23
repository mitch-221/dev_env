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


def pair_pnl(log_p, i, j, split):
    """Daily $ PnL of the sized z-score rule on pair (i, j).

    Sizing: at |z| = Z_CAP the dollar positions exactly hit the caps:
      leg i dollars = -z * S, leg j dollars = +z * S * beta
      S = min(cap_i, cap_j / |beta|) / Z_CAP
    Returns (pnl, beta, S) or None.
    """
    x, y = log_p[j], log_p[i]
    vx = np.var(x[:split])
    if vx < 1e-12:
        return None
    beta = np.cov(y[:split], x[:split])[0, 1] / vx
    spread = y - beta * x

    ab = max(abs(beta), 1e-6)
    S = min(CAPS[i], CAPS[j] / ab) / Z_CAP

    n = len(spread)
    if n <= Z_WINDOW + 5:
        return None
    z = np.full(n, np.nan)
    for t in range(Z_WINDOW, n):
        w = spread[t - Z_WINDOW:t]
        sd = w.std()
        if sd > 1e-12:
            z[t] = (spread[t] - w.mean()) / sd

    pos = 0.0
    pnl = np.zeros(n)
    for t in range(Z_WINDOW, n - 1):
        zt = z[t]
        if np.isfinite(zt) and abs(zt - (-pos)) > Z_BAND:
            new_pos = -np.clip(zt, -Z_CAP, Z_CAP)
        else:
            new_pos = pos
        turn = abs(new_pos - pos) * S
        cost = turn * (COMM[i] + abs(beta) * COMM[j])
        pos = new_pos
        ds = spread[t + 1] - spread[t]
        pnl[t + 1] = pos * S * ds - cost
    return pnl, beta, S


def main(path, max_pairs=25):
    prc, tickers = load_prices(path)
    n_inst, n_days = prc.shape
    log_p = np.log(prc)
    split = int(n_days * TRAIN_FRAC)
    warm = Z_WINDOW + 1

    print(f"Backtesting all pairs ({n_inst * (n_inst - 1) // 2})...")
    candidates = []  # (train_score, i, j, beta, S, pnl)
    for i in range(n_inst):
        for j in range(i + 1, n_inst):
            r = pair_pnl(log_p, i, j, split)
            if r is None:
                continue
            pnl, beta, S = r
            tr = objective(pnl[warm:split])
            candidates.append((tr, i, j, beta, S, pnl))
    candidates.sort(key=lambda c: -c[0])
    print(f"Candidates with positive train score: "
          f"{sum(1 for c in candidates if c[0] > 0)}")

    # ---- greedy selection on TRAIN objective, asset-disjoint ----
    chosen = []
    used = set()
    port = np.zeros(n_days)
    best_score = -np.inf
    improved = True
    while improved and len(chosen) < max_pairs:
        improved = False
        best_add = None
        for c in candidates:
            tr, i, j, beta, S, pnl = c
            if i in used or j in used:
                continue
            s = objective((port + pnl)[warm:split])
            if s > best_score and (best_add is None or s > best_add[0]):
                best_add = (s, c)
        if best_add is not None:
            best_score, c = best_add
            tr, i, j, beta, S, pnl = c
            chosen.append(c)
            used.update((i, j))
            port = port + pnl
            improved = True
            print(f"  + {tickers[i]}/{tickers[j]:<6} "
                  f"train score -> {best_score:,.0f}")

    # ---- honest evaluation ----
    seg = {
        "train": port[warm:split],
        "holdout": port[split:],
        "full": port[warm:],
    }
    print("\nPortfolio evaluation:")
    for name, p in seg.items():
        mu, sd, sr, sc = stats(p)
        print(f"  {name:<8} mu ${mu:>8,.0f}/day  sigma ${sd:>8,.0f}  "
              f"SR {sr:>5.2f}  Score {sc:>8,.0f}")

    max_deploy = sum(
        Z_CAP * c[4] * (1 + abs(c[3])) for c in chosen)
    print(f"\nPairs chosen: {len(chosen)}   "
          f"max gross deployment: ${max_deploy:,.0f}")

    # ---- reports ----
    with open("chosen_pairs.txt", "w") as f:
        f.write("CHOSEN PAIRS PORTFOLIO\n")
        f.write("=" * 78 + "\n")
        f.write(f"Selected on train days {warm}-{split}; holdout "
                f"{split}-{n_days} untouched by selection.\n\n")
        f.write(f"{'pair':<14}{'idx':>9}{'beta':>8}{'unit $S':>10}"
                f"{'train Sc':>10}{'holdout Sc':>11}\n")
        f.write("-" * 78 + "\n")
        for tr, i, j, beta, S, pnl in chosen:
            ho = objective(pnl[split:])
            f.write(f"{tickers[i]}/{tickers[j]:<8}{f'({i},{j})':>9}"
                    f"{beta:>8.3f}{S:>10,.0f}{tr:>10,.0f}{ho:>11,.0f}\n")
        f.write("\nPortfolio:\n")
        for name, p in seg.items():
            mu, sd, sr, sc = stats(p)
            f.write(f"  {name:<8} mu ${mu:>8,.0f}/day  sigma ${sd:>8,.0f}  "
                    f"SR {sr:>5.2f}  Score {sc:>8,.0f}\n")
        f.write(f"\nMax gross deployment at |z|={Z_CAP}: ${max_deploy:,.0f}\n")

    with open("pairs_config.py", "w") as f:
        f.write("# Auto-generated by portfolio_builder.py\n")
        f.write("# (i, j, beta_hint)  - beta_hint is the train-fit hedge "
                "ratio;\n")
        f.write("# the live strategy should re-estimate beta on a rolling "
                "window\n")
        f.write("# and can use this as a sanity fallback.\n")
        f.write("PAIRS = [\n")
        for tr, i, j, beta, S, pnl in chosen:
            f.write(f"    ({i}, {j}, {beta:.4f}),   "
                    f"# {tickers[i]}/{tickers[j]}\n")
        f.write("]\n")
        f.write(f"Z_WINDOW = {Z_WINDOW}\nZ_CAP = {Z_CAP}\nZ_BAND = {Z_BAND}\n")

    print("\nWrote: chosen_pairs.txt")
    print("Wrote: pairs_config.py  (paste-ready for the strategy)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python portfolio_builder.py prices.txt [max_pairs]")
        sys.exit(1)
    mp = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    main(sys.argv[1], mp)
