"""
Component attribution for teamName_v2. Runs the exact eval protocol on:
  1. resid-only        (market overlay off)
  2. mkt-only          (resid core off)
  3. combined          (as-is)
  4. resid-only, tight band + stricter AR gate variants

Usage:
    python attribution.py prices.txt
"""

import sys
import numpy as np
import teamName_v2 as S
import eval_strategy as E


def reset():
    S._pos = np.zeros(S.N_INST, dtype=int)
    S._last_n = 0
    S._V = None
    S._last_fit = -10**9


def evaluate(prc, label, **overrides):
    saved = {k: getattr(S, k) for k in overrides}
    for k, v in overrides.items():
        setattr(S, k, v)
    reset()
    pnl, volume = E.run(prc, S.getMyPosition)
    for k, v in saved.items():
        setattr(S, k, v)

    n_days = prc.shape[1]
    d500 = min(500, int(n_days * 2 / 3))
    dB = min(650, n_days - 100)
    segs = [("A", pnl[E.WARM:d500]), ("B", pnl[d500:dB]),
            ("C", pnl[dB:]), ("full", pnl[E.WARM:])]
    row = f"{label:<28}"
    for name, p in segs:
        row += f"{name}:{E.objective(p):>6,.0f}  "
    mu = pnl[E.WARM:].mean()
    sd = pnl[E.WARM:].std()
    sr = np.sqrt(250) * mu / sd if sd > 1e-12 else 0
    row += f"(full mu ${mu:,.0f} sd ${sd:,.0f} SR {sr:.2f})"
    print(row)


def main(path):
    prc = E.load_prices(path)
    print(f"{'variant':<34}segment scores")
    print("-" * 106)
    evaluate(prc, "prev best (raw, AGGR 2, 100k)", VOL_NORM=False, AGGR=2.0)
    evaluate(prc, "raw resid, AGGR 1.5", VOL_NORM=False, AGGR=1.5)
    print("-" * 106)
    for cap in (100_000, 50_000, 20_000):
        for ag in (1.5, 2.0, 3.0):
            evaluate(prc, f"VOLNORM, ALGO {cap//1000}k, AGGR {ag}",
                     VOL_NORM=True, ALGO_SIZE_CAP=float(cap), AGGR=ag)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python attribution.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
