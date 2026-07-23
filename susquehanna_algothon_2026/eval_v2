"""
Evaluate a strategy file under the A/B/C segment protocol used in
deep_dive.py, with the exact competition mechanics (daily clipping,
per-asset commissions) and objective function.

Usage:
    python eval_strategy.py prices.txt [strategy_module]
    (default strategy_module: teamName_v2)
"""

import sys
import importlib
import numpy as np

N_INST = 51
CAPS = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM = np.full(N_INST, 0.0001)
COMM[0] = 0.00002
WARM = 130


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


def run(prc, getMyPosition, start=WARM):
    n_inst, n_days = prc.shape
    pos = np.zeros(n_inst, dtype=int)
    cash = 0.0
    pnl = np.zeros(n_days)
    value_prev = 0.0
    volume = 0.0
    for t in range(start, n_days):
        p = prc[:, t]
        new_pos = np.asarray(getMyPosition(prc[:, : t + 1]), dtype=int)
        mx = np.floor(CAPS / p).astype(int)
        new_pos = np.clip(new_pos, -mx, mx)
        delta = new_pos - pos
        traded = np.abs(delta) * p
        volume += traded.sum()
        cash -= (delta * p).sum() + (traded * COMM).sum()
        pos = new_pos
        value = cash + (pos * p).sum()
        pnl[t] = value - value_prev
        value_prev = value
    return pnl, volume


def main(path, mod_name="teamName_v2"):
    prc = load_prices(path)
    n_days = prc.shape[1]
    mod = importlib.import_module(mod_name)

    pnl, volume = run(prc, mod.getMyPosition)

    d500 = min(500, int(n_days * 2 / 3))
    dB = min(650, n_days - 100)
    segs = {
        f"A[{WARM}-{d500}]": pnl[WARM:d500],
        f"B[{d500}-{dB}]": pnl[d500:dB],
        f"C[{dB}-{n_days}]": pnl[dB:n_days],
        "full": pnl[WARM:],
    }
    print(f"Strategy: {mod_name}")
    for name, p in segs.items():
        mu, sd = p.mean(), p.std()
        sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0.0
        print(f"  {name:<14} mu ${mu:>7,.0f}/day  sd ${sd:>7,.0f}  "
              f"SR {sr:>5.2f}  Score {objective(p):>8,.0f}")
    # 250-day-equivalent traded volume for the min-activity check
    v250 = volume * 250 / (n_days - WARM)
    print(f"  volume/250d: ${v250:,.0f}  (min required: $25,000)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_strategy.py prices.txt [strategy_module]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "teamName_v2")
