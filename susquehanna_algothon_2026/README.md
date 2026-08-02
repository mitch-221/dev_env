# Susquehanna x UNSW FinTech Society Algothon 2026

My submission and research for the 2026 Algothon — a student algorithmic trading
hackathon where teams write a position-sizing function that's backtested against
daily commissions and per-asset position caps, scored on realised PnL/Sharpe over
held-out data segments.

**Result:** [fill in — final score / rank / placement if applicable]

## Final strategy — `MTAlpha.py`

Two independent, additively-combined layers trading a 51-asset universe:

- **Layer 1 — cointegrated pairs (33 pairs, 39 assets).** Each pair's log-price
  ratio is tracked as a mean-reverting spread; positions are sized by
  `kappa * |z| * sigma` (proportional for the index asset ALGO, winner-takes-all
  for the rest), with beta-scaled sizing and an expanding mean / ring-buffer
  volatility estimate.
- **Layer 2 — cointegrated triplets (20 triplets, 12 "stranded" assets not used
  in Layer 1).** Spread defined as `log(P_i) - b1*log(P_j) - b2*log(P_k) - alpha`,
  with position caps scaled down to respect each asset's individual cap.

Backtested score: ~788 on days 1–1000 in-sample, ~820 expected on the finals
segment (days 1001+). Parameters (beta/alpha OLS estimates, band width,
aggression, z-window) were tuned across all four quarters of the price history.

## How to run it

```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
python eval.py
```

`eval.py` imports `getMyPosition` from `MTAlpha.py` and scores it on the last
250 days of `prices.txt` using the official competition mechanics (daily
position clipping, per-asset commissions).

## `iterations/` — earlier strategy versions

Kept to show how the approach developed rather than just the final answer:

- **`teamName_v2.py`** — first working version: 17 disjoint cointegrated pairs
  only, asset-disjoint (no shared exposure). Score ~678, Sharpe ~8.4. Adding a
  factor-residual overlay on top of this was tried and *dropped* — it cost 186
  score points by fighting the pair positions on shared assets, which is what
  motivated the disjoint-triplet design in the final version instead.
- **`teamName_v1.py`** — added an ALGO return-residual fade (ALGO's return
  minus its fitted basket return has Hurst ≈ 0.011, i.e. strongly same-day
  mean-reverting) on top of two structurally cointegrated pairs.

## `research/` — analysis behind the strategy

- **`algo_index_test.py`** — tests whether ALGO is just an index of the other
  50 assets (out-of-sample regression R², weight stability across halves,
  equal-weight/price-weight baselines), then simulates trading the deviation
  if the fit holds up.
- **`algo_r.py`** — digs into the ~17% of ALGO's variance the basket doesn't
  explain: residual autocorrelation, lagged-basket predictors, and per-asset
  lagged predictors of next-day ALGO residual.
- **`arb_lab.py`** — hedged ALGO index-arbitrage lab: rolling-OLS basket
  weights, z-scored spread, linear vs. tanh position sizing, grid-searched
  over fit window / z-window and scored per segment.
- **`arb_diag.py`** — diagnostic companion to the above: confirms which
  strategy module was actually wired in, characterises the spread
  distribution (std, AR(1), implied half-life), and reports live arb activity
  (trade count, turnover, PnL) in isolation.
- **`range_analysis.py`** — screens the universe for range-bound assets and
  cointegration candidates (ADF, Engle-Granger) — this is where the pairs used
  in the final strategy were first identified.
- **`universe_exp.py`** — backtests two structurally-designed pairs
  (CTGI/EELT, MTNS/BENI) confirmed by both cointegration and ratio-stationarity
  tests, including EELT as a sub-index against each partner.
- **`attribution.py`** — component attribution: isolates resid-only, market-only,
  and combined performance to confirm each piece is pulling its weight rather
  than one component carrying the score.
- **`eval_strategy.py`** — standalone evaluator for any strategy module under
  the exact competition protocol (segments, clipping, commissions), used to
  sanity-check components in isolation before wiring them into the final file.
