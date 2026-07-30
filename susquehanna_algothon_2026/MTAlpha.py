"""
MTAlpha v8 — 16 pairs, per-pair adaptive state.
Sim score: ~943 full 1000 days. All 4 windows positive (766/1078/994/933).

Per-pair configuration:
  - OLS hedge ratio (beta, alpha) where it improves cointegration
  - Expanding or rolling mean — chosen per pair by cross-validation
  - Per-pair vol compression params (vlo, vhi) tuned on all 4 windows
  - All parameters state held across calls in _state dict

How it works per pair:
  spread[t] = log(P_i[t]) - beta*log(P_j[t]) - alpha
  sm = expanding mean of spread (or rolling, pair-specific)
  ss = rolling std of spread over zw days
  z  = (spread[t] - sm) / ss
  vol_mult = clip(long_run_vol / ss, vlo, vhi)
  position = tanh(aggr * z) * vol_mult * cap
"""
import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# (i, j, zw, band, aggr, vlo, vhi, expanding, beta, alpha)
PAIRS = [
    (18,35,24,0.50,20.0, 0.0,5.0, True, 1.0000, 0.000000),  # RTTH/NAYO
    (25,37,22,0.50,15.0, 0.0,4.0, True, 1.0000, 0.000000),  # CTGI/EELT
    ( 7,40,17,0.50,20.0, 0.0,5.0, True, 1.0000, 0.000000),  # HETT/ULXY
    ( 1,20,12,0.50,20.0, 0.0,3.0, True, 1.0000, 0.000000),  # AENO/NWIG
    (13,45,16,0.50, 3.0, 0.0,5.0, True, 1.0000, 0.000000),  # EORC/NGTE
    (10,46,14,0.50, 3.0, 0.0,4.0, True, 1.0000, 0.000000),  # SMAH/ILVX
    (49,50,23,0.50,20.0, 0.0,3.0, True, 1.0000, 0.000000),  # MHRM/EAFC
    ( 8,27,19,0.50,20.0, 0.0,5.0, True, 1.0000, 0.000000),  # HUXZ/ACAC
    (31,43,30,0.50,20.0, 0.0,4.0, True, 1.0000, 0.000000),  # ACIX/ITPA
    (26,32,46,0.50, 2.0, 0.0,4.0, True, 0.8401, 0.225163),  # ALUT/CCNS
    (36,41,28,0.50, 8.0, 0.0,4.0, True, 1.0000, 0.000000),  # FWWG/BLBT
    ( 2,42,60,0.50,20.0, 0.1,3.0,False, 1.0000, 0.000000),  # LSST/BENI
    (39,47,60,0.00,20.0, 0.1,3.0,False, 0.5770, 1.395205),  # AETS/FCSG
    (15,22,30,0.20,20.0, 0.0,5.0, True, 0.4212, 1.634643),  # HRET/MDGI
    (28,38,10,0.30,12.0, 0.1,3.0, True, 0.4078, 4.322803),  # SRTX/HRND
    ( 9,11,30,1.00,20.0, 0.0,5.0, True, 1.0000, 0.000000),  # DUCT/NPCK
]

REBAL_BAND  = 0.05
MIN_HISTORY = max(int(zw) for _,_,zw,_,_,_,_,_,_,_ in PAIRS) + 10

# Per-pair running state — updated every call
# Each pair gets: spread_sum, spread_sum_sq, spread_count (for expanding mean/var)
# Plus last zw spread values (ring buffer for rolling std)
_state   = None
_pos     = np.zeros(N_INST, dtype=int)
_last_n  = 0

N_PAIRS = len(PAIRS)


def _init_state():
    """Initialise per-pair state buffers."""
    return {
        'sum1':    np.zeros(N_PAIRS),   # running sum of spread
        'sum2':    np.zeros(N_PAIRS),   # running sum of spread^2
        'count':   np.zeros(N_PAIRS, dtype=int),
        'buf':     [np.zeros(zw) for _,_,zw,_,_,_,_,_,_,_ in PAIRS],
        'buf_idx': np.zeros(N_PAIRS, dtype=int),
        'buf_full':np.zeros(N_PAIRS, dtype=bool),
    }


def getMyPosition(prcSoFar):
    global _state, _pos, _last_n

    prc     = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh  = np.floor(CAPS / p_today).astype(int)

    # Reset on restart
    if n_days <= _last_n:
        _state = _init_state()
        _pos   = np.zeros(N_INST, dtype=int)
    _last_n = n_days

    if _state is None:
        _state = _init_state()

    log_today = np.log(p_today)
    target_dollars = np.zeros(N_INST)

    for pidx, (i, j, zw, band, aggr, vlo, vhi, expanding, beta, alpha) in enumerate(PAIRS):
        if max(i, j) >= n_inst:
            continue

        # Current spread value
        s_now = log_today[i] - beta * log_today[j] - alpha

        # Update ring buffer (rolling window of zw values)
        buf     = _state['buf'][pidx]
        idx     = _state['buf_idx'][pidx]
        old_val = buf[idx] if _state['buf_full'][pidx] else 0.0
        buf[idx] = s_now
        _state['buf_idx'][pidx] = (idx + 1) % zw
        if not _state['buf_full'][pidx] and idx == zw - 1:
            _state['buf_full'][pidx] = True

        # Need full buffer before trading
        if not _state['buf_full'][pidx]:
            _state['sum1'][pidx] += s_now
            _state['sum2'][pidx] += s_now * s_now
            _state['count'][pidx] += 1
            continue

        # Update expanding statistics
        _state['sum1'][pidx] += s_now
        _state['sum2'][pidx] += s_now * s_now
        _state['count'][pidx] += 1
        n = _state['count'][pidx]

        # Short-window rolling std from ring buffer
        ss_mean = buf.mean()
        ss_var  = buf.var()
        if ss_var < 1e-14:
            continue
        ss = np.sqrt(ss_var)

        # Mean: expanding or rolling
        if expanding:
            sm = _state['sum1'][pidx] / n
        else:
            sm = ss_mean

        # Volatility regime: long-run vol vs short-run vol
        # Long-run: expanding std from running sum of squares
        exp_mean = _state['sum1'][pidx] / n
        exp_var  = max(_state['sum2'][pidx] / n - exp_mean ** 2, 0.0)
        long_vol = np.sqrt(exp_var) if exp_var > 0 else ss

        vol_mult = np.clip(long_vol / ss, vlo, vhi)

        z = (s_now - sm) / ss
        if abs(z) <= band:
            continue

        f = np.tanh(aggr * z) * vol_mult
        target_dollars[i] -= f * CAPS[i]
        target_dollars[j] += f * CAPS[j] * min(beta, 1.0)

    target_dollars = np.clip(target_dollars, -CAPS, CAPS)
    target_sh = np.round(target_dollars / p_today).astype(int)
    target_sh = np.clip(target_sh, -max_sh, max_sh)

    drift   = np.abs(target_sh - _pos) * p_today
    trade   = drift > REBAL_BAND * CAPS
    new_pos = _pos.copy()
    new_pos[trade] = target_sh[trade]
    new_pos = np.clip(new_pos, -max_sh, max_sh)

    _pos = new_pos
    return new_pos.copy()
