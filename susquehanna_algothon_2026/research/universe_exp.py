"""
Structural pairs backtest: CTGI/EELT and MTNS/BENI.

Both pairs show p=0.000 on both Engle-Granger cointegration AND
log-price ratio stationarity. These are structurally designed
relationships, not statistical coincidences.

The trade:
  - Track the log price ratio: spread_t = log(P_i / P_j)
  - Z-score the spread over a rolling window
  - Trade: when spread is high (i expensive vs j), short i, long j
  - Size: each leg sized to its $10k cap

Also tests EELT as a sub-index by running the same approach on
EELT against each of its cointegrated partners.

Usage: python pairs_backtest.py prices.txt
"""

import sys
import numpy as np
from statsmodels.tsa.stattools import adfuller

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0
COMM   = np.full(N_INST, 0.0001)
COMM[0] = 0.00002

# Ticker to index mapping (0=ALGO, 1=AENO, ...)
TICKERS = [
    'ALGO','AENO','LSST','SRNA','ELLT','AMRP','OTCS','HETT','HUXZ','DUCT',
    'SMAH','NPCK','MSDP','EORC','CUBO','HRET','ANSO','DIHO','RTTH','SPLZ',
    'NWIG','MMBT','MDGI','AGVF','RRES','CTGI','ALUT','ACAC','SRTX','GARI',
    'RCRI','ACIX','CCNS','MTNS','IHOZ','NAYO','FWWG','EELT','HRND','AETS',
    'ULXY','BLBT','BENI','ITPA','HTRK','NGTE','ILVX','FCSG','FARS','MHRM',
    'EAFC'
]

# Structural pairs confirmed at p=0.000 on both tests
STRUCTURAL_PAIRS = [
    ('CTGI', 'EELT'),   # both range-bound, ratio p=0.000
    ('MTNS', 'BENI'),   # both range-bound, ratio p=0.000
]

# EELT cluster: cointegrated partners (for sub-index analysis)
EELT_CLUSTER = [
    'FARS', 'ILVX', 'BENI', 'ITPA', 'AETS',
    'HTRK', 'FCSG', 'HRND', 'EAFC', 'MHRM', 'ULXY', 'BLBT'
]


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
    if tickers is None:
        tickers = TICKERS
    return arr, tickers


def objective(pnl):
    if len(pnl) == 0: return 0.0
    mu, sd = pnl.mean(), pnl.std()
    if mu < 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr * sr / (sr * sr + 1.0)


def idx(ticker, tickers):
    try:
        return tickers.index(ticker)
    except ValueError:
        if ticker in TICKERS:
            return TICKERS.index(ticker)
        raise ValueError(f"Ticker {ticker} not found")


def pair_backtest(prc, i, j, zw=30, aggr=2.0, band=0.3, tickers=None):
    """
    Walk-forward pairs trade on log price ratio.
    Short i / long j when ratio is high (i expensive).
    Returns daily PnL array.
    """
    n_inst, n_days = prc.shape
    log_p = np.log(prc)
    ratio = log_p[i] - log_p[j]   # log(P_i / P_j)

    pos_i = 0.0; pos_j = 0.0
    pnl = np.zeros(n_days)
    rets = np.diff(log_p, axis=1)

    for t in range(zw + 1, n_days - 1):
        win = ratio[t - zw:t]
        s_mean, s_sd = win.mean(), win.std()
        if s_sd < 1e-12:
            continue
        z = (ratio[t] - s_mean) / s_sd

        if abs(z) > band:
            f = np.tanh(aggr * z)
            # short i (expensive), long j (cheap)
            new_i = -f * CAPS[i]
            new_j =  f * CAPS[j]
        else:
            new_i = 0.0; new_j = 0.0

        cost = (abs(new_i - pos_i) * COMM[i] +
                abs(new_j - pos_j) * COMM[j])
        pos_i = new_i; pos_j = new_j
        pnl[t + 1] = (pos_i * rets[i, t] +
                      pos_j * rets[j, t] - cost)
    return pnl


def main(path):
    prc, tickers = load_prices(path)
    n_inst, n_days = prc.shape
    log_p = np.log(prc)
    d500 = min(500, int(n_days * 2/3))
    dB   = min(650, n_days - 100)

    def seg_print(pnl, label, start=30):
        segs = [(f"A[{start}-{d500}]", pnl[start:d500]),
                (f"B[{d500}-{dB}]",    pnl[d500:dB]),
                (f"C[{dB}-{n_days}]",  pnl[dB:]),
                ("full",               pnl[start:])]
        if n_days > 800:
            segs.append((f"LB[750-{n_days}]", pnl[750:]))
        print(f"\n  {label}")
        for name, p in segs:
            mu, sd = p.mean(), p.std()
            sr = np.sqrt(250)*mu/sd if sd > 1e-12 else 0
            print(f"    {name:<22} mu ${mu:>6,.0f}  sd ${sd:>5,.0f}  "
                  f"SR {sr:>5.2f}  Score {objective(p):>6,.0f}")

    # ============================================================
    print("=" * 70)
    print("1. RATIO DIAGNOSTICS")
    print("=" * 70)
    for ta, tb in STRUCTURAL_PAIRS:
        ia, ib = idx(ta, tickers), idx(tb, tickers)
        ratio = log_p[ia] - log_p[ib]
        p_adf = adfuller(ratio, autolag='AIC')[1]
        ac1 = np.corrcoef(ratio[:-1], ratio[1:])[0, 1]
        hl  = -np.log(2) / np.log(ac1) if 0 < ac1 < 1 else np.inf
        print(f"  {ta}/{tb}: ratio ADF p={p_adf:.4f}  "
              f"AR(1)={ac1:+.3f}  half-life={hl:.1f}d  "
              f"ratio std={ratio.std():.4f}")

    # ============================================================
    print()
    print("=" * 70)
    print("2. STRUCTURAL PAIRS (CTGI/EELT and MTNS/BENI)")
    print("=" * 70)

    # Grid: z-window x band
    print("\n  Parameter grid (z-window x band), scored on B:")
    print(f"  {'pair':<12} {'zw':>4} {'band':>5}  "
          f"{'A':>8} {'B':>8} {'C':>8}  {'full SR':>8}")
    print("  " + "-" * 60)

    best = {}
    for ta, tb in STRUCTURAL_PAIRS:
        ia, ib = idx(ta, tickers), idx(tb, tickers)
        best_b = -np.inf; best_cfg = None; best_pnl = None
        for zw in (10, 20, 30, 40):
            for band in (0.3, 0.5, 1.0):
                pnl = pair_backtest(prc, ia, ib, zw=zw, band=band)
                sa = objective(pnl[zw:d500])
                sb = objective(pnl[d500:dB])
                sc = objective(pnl[dB:])
                sf = pnl[zw:].mean() / max(pnl[zw:].std(), 1e-9)
                print(f"  {ta}/{tb:<6} {zw:>4} {band:>5.1f}"
                      f"  {sa:>8,.0f} {sb:>8,.0f} {sc:>8,.0f}"
                      f"  {np.sqrt(250)*sf:>8.2f}")
                if sb > best_b:
                    best_b = sb
                    best_cfg = (zw, band)
                    best_pnl = pnl
        best[(ta, tb)] = (best_cfg, best_pnl)

    print()
    print("  Best configs (by B score):")
    for (ta, tb), ((zw, band), pnl) in best.items():
        ia, ib = idx(ta, tickers), idx(tb, tickers)
        seg_print(pnl, f"{ta}/{tb} (zw={zw}, band={band})", start=zw)

    # ============================================================
    print()
    print("=" * 70)
    print("3. EELT CLUSTER - is EELT a tradeable sub-index?")
    print("=" * 70)
    ie = idx('EELT', tickers)
    print(f"  Testing EELT against its {len(EELT_CLUSTER)} cointegrated partners:")
    cluster_pnl = np.zeros(n_days)
    cluster_results = []
    for tb in EELT_CLUSTER:
        try:
            ib = idx(tb, tickers)
        except ValueError:
            continue
        # use best zw=20, band=0.5 as starting point
        pnl = pair_backtest(prc, ie, ib, zw=20, band=0.5)
        sb = objective(pnl[d500:dB])
        sc = objective(pnl[dB:])
        cluster_results.append((sb, tb, pnl))
        mu = pnl[20:].mean(); sr = np.sqrt(250)*mu/max(pnl[20:].std(),1e-9)
        print(f"  EELT/{tb:<6}: B={sb:>6,.0f}  C={sc:>6,.0f}  SR={sr:.2f}")

    # combine positive-B pairs only
    pos_pairs = [(tb, pnl) for b, tb, pnl in cluster_results if b > 0]
    if pos_pairs:
        combo = sum(p for _, p in pos_pairs)
        print(f"\n  Combined {len(pos_pairs)} positive-B EELT pairs:")
        seg_print(combo, "EELT cluster combo", start=20)
    else:
        print("\n  No EELT cluster pairs scored positive on B.")

    # ============================================================
    print()
    print("=" * 70)
    print("4. FULL PORTFOLIO: structural pairs + resid signal")
    print("=" * 70)
    print("  Combining best structural pairs with the factor-resid")
    print("  signal from teamName_final.py:")

    # structural pairs PnL
    struct_pnl = np.zeros(n_days)
    for (ta, tb), (_, pnl) in best.items():
        struct_pnl += pnl

    # add positive EELT cluster pairs
    if pos_pairs:
        for _, p in pos_pairs:
            struct_pnl += p

    seg_print(struct_pnl, "Structural pairs alone", start=30)

    print()
    print("  To test the full combined strategy:")
    print("  1. Add structural pair positions to teamName_final.py")
    print("  2. The pairs use assets 25/37 and 33/42 - within $10k caps")
    print("  3. No conflict with the resid signal's ALGO exclusion")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pairs_backtest.py prices.txt")
        sys.exit(1)
    main(sys.argv[1])
