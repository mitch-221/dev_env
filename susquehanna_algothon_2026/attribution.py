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
    print(f"{'variant':<28}segment scores")
    print("-" * 100)
    evaluate(prc, "combined (as-is)")
    evaluate(prc, "resid only", W_MKT=0.0)
    evaluate(prc, "mkt only", W_RESID=0.0)
    print("-" * 100)
    evaluate(prc, "resid only, band 2%", W_MKT=0.0, REBAL_BAND=0.02)
    evaluate(prc, "resid only, band 15%", W_MKT=0.0, REBAL_BAND=0.15)
    evaluate(prc, "resid only, nf=1", W_MKT=0.0, NF=1)
    evaluate(prc, "resid only, nf=3", W_MKT=0.0, NF=3)
    print("-" * 100)
    evaluate(prc, "combined, AR gate t>2", AR_TMIN=2.0)
    evaluate(prc, "combined, mkt weight 0.3", W_MKT=0.3)
    evaluate(prc, "mkt only, AR gate t>2", W_RESID=0.0, AR_TMIN=2.0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python attribution.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
