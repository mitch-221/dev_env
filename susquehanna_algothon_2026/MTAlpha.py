"""
MTAlpha — 23 pairs, ERP-based dynamic capital allocation.
Score: ~630 full 1000 days. All 4 windows positive.

DYNAMIC ALLOCATION:
  6 extra "conflict" pairs compete with primary pairs for shared assets.
  On each day t, for each asset A:
    ERP_pair = kappa * |z| * sigma  (expected 1-day reversion profit per dollar)
  The pair with highest ERP wins full $10k for that asset.
  The loser still trades its OTHER (free) asset at full cap.

  Example: HRET is used by HRET/MDGI (primary) and ALGO/HRET (conflict).
  On days when ALGO/HRET has stronger signal → ALGO gets $100k, HRET goes
  to ALGO/HRET. On days when HRET/MDGI is stronger → HRET/MDGI keeps HRET.
  Either way, MDGI and ALGO are each trading independently.

PAIRS (17 primary + 6 conflict):
  Primary: RTTH/NAYO CTGI/EELT HETT/ULXY AENO/NWIG EORC/NGTE SMAH/ILVX
           MHRM/EAFC HUXZ/ACAC ACIX/ITPA ALUT/CCNS FWWG/BLBT LSST/BENI
           AETS/FCSG HRET/MDGI SRTX/HRND DUCT/NPCK RRES/ANSO
  Conflict: ANSO/EELT ALGO/HRET RCRI/NAYO HTRK/HRET MTNS/BENI MTNS/ULXY

PRIOR_STATE: baked-in 1000-day statistics. Zero warmup on finals day 1001.
"""

import numpy as np

N_INST = 51
CAPS   = np.full(N_INST, 10_000.0)
CAPS[0] = 100_000.0

# (i, j, zw, band, aggr, expanding, beta, alpha)
PAIRS = [
    (18,35,12,0.50,12.0, True, 1.0000, 0.000000),  # RTTH/NAYO
    (25,37,16,0.50,20.0, True, 1.0000, 0.000000),  # CTGI/EELT
    ( 7,40,16,0.50,20.0, True, 1.0000, 0.000000),  # HETT/ULXY
    ( 1,20,30,0.50,20.0, True, 1.0000, 0.000000),  # AENO/NWIG
    (13,45,10,0.50, 5.0, True, 1.0000, 0.000000),  # EORC/NGTE
    (10,46,14,0.50,20.0, True, 1.0000, 0.000000),  # SMAH/ILVX
    (49,50, 8,0.50,20.0, True, 1.0000, 0.000000),  # MHRM/EAFC
    ( 8,27, 8,0.50,20.0, True, 1.0000, 0.000000),  # HUXZ/ACAC
    (31,43,12,0.50, 5.0, True, 1.0000, 0.000000),  # ACIX/ITPA
    (26,32,30,0.50, 2.0, True, 0.8401, 0.225163),  # ALUT/CCNS
    (36,41,40,0.50, 5.0, True, 1.0000, 0.000000),  # FWWG/BLBT
    ( 2,42,46,0.50, 3.0,False, 1.0000, 0.000000),  # LSST/BENI
    (39,47, 8,0.00,20.0,False, 0.5770, 1.395205),  # AETS/FCSG
    (15,22,24,0.20,20.0, True, 0.4212, 1.634643),  # HRET/MDGI
    (28,38,30,0.30, 1.5, True, 0.4078, 4.322803),  # SRTX/HRND
    ( 9,11,24,1.00,20.0, True, 1.0000, 0.000000),  # DUCT/NPCK
    (24,16,30,1.00, 3.0, True, 1.0000, 0.000000),  # RRES/ANSO
    (16,37,16,0.50,20.0, True, 1.0000, 0.000000),  # ANSO/EELT  ← conflict
    ( 0,15,12,1.00,20.0, True, 1.0000, 0.000000),  # ALGO/HRET  ← conflict
    (30,35,12,1.00, 3.0, True, 1.0000, 0.000000),  # RCRI/NAYO  ← conflict
    (44,15, 8,0.20,20.0, True, 1.0000, 0.000000),  # HTRK/HRET  ← conflict
    (33,42,12,0.50,20.0, True, 1.0000, 0.000000),  # MTNS/BENI  ← conflict
    (33,40,16,0.70,20.0, True, 1.0000, 0.000000),  # MTNS/ULXY  ← conflict
]

# Precomputed mean-reversion rates kappa = ln(2)/HL
KAPPAS = [
    0.086244, 0.092865, 0.120842, 0.160824, 0.123326, 0.148045,
    0.090155, 0.109141, 0.068962, 0.048232, 0.073412, 0.018091,
    0.014108, 0.018287, 0.017535, 0.012384, 0.018208, 0.023997,
    0.018224, 0.024823, 0.030713, 0.066462, 0.035352,
]

# Baked-in 1000-day statistics: (sum1, sum2, count, ring_buffer)
PRIOR_STATE = [
    (66.57384478,14.31938634,1000,[0.21730128,0.26129854,0.19596862,0.16540963,0.21792042,0.20230002,0.18596131,0.20760445,0.20546759,0.19117481,0.21332810,0.20930415]),
    (53.79742560,11.64158670,1000,[0.05810979,0.05367831,0.04751578,0.00523040,0.01591228,-0.00283722,0.00749424,0.06550137,0.00229095,0.03628567,0.02816162,0.01911766,0.05436001,0.00908434,0.06476536,0.06911886]),
    (287.09991992,86.50787308,1000,[0.16808215,0.15768201,0.18955529,0.21759436,0.24412647,0.24567849,0.22864364,0.23484302,0.24186507,0.26381676,0.25424647,0.20163223,0.23028687,0.22290285,0.22526601,0.26421873]),
    (-65.16673088,6.16336197,1000,[-0.07619157,-0.04830062,-0.08354213,-0.08030701,-0.07695161,-0.07002798,-0.06452225,-0.03107142,-0.00367153,-0.05094689,-0.03453068,-0.00631151,0.00222525,-0.04845412,-0.09759600,-0.05632802,-0.04884434,-0.04060918,-0.02611999,-0.01752235,-0.02018162,-0.01213145,-0.02731726,-0.02106360,-0.01412075,-0.02031418,-0.04839450,-0.02921214,-0.01341787,-0.04961399]),
    (-500.92487593,254.19244728,1000,[-0.40677059,-0.39113461,-0.39023807,-0.42998707,-0.41448706,-0.44255987,-0.47228237,-0.43385093,-0.44745471,-0.42562708]),
    (307.06491240,95.99124105,1000,[0.38632680,0.35678471,0.35868686,0.36255889,0.38531433,0.38708224,0.37212359,0.38095927,0.37277629,0.40482624,0.39822352,0.38060981,0.35785263,0.36647815]),
    (-682.81184128,469.99482481,1000,[-0.75141609,-0.77817018,-0.80488367,-0.80135353,-0.80745788,-0.82437708,-0.81650555,-0.82843406]),
    (100.68615865,12.94542898,1000,[-0.00268354,-0.00268508,0.02126739,0.03714659,0.04512607,0.05993394,0.02886094,0.03132165]),
    (-190.08310275,38.28357277,1000,[-0.14932629,-0.17655310,-0.17552572,-0.16503450,-0.17220823,-0.15775169,-0.17160334,-0.17453617,-0.15260601,-0.16157061,-0.12996633,-0.13442527]),
    (0.03929738,6.25983118,1000,[-0.20002586,-0.18769917,-0.18509123,-0.17696878,-0.16611221,-0.12883135,-0.14082752,-0.09193791,-0.03419529,-0.02763411,-0.00723525,0.02091510,0.07738862,0.06996586,0.05424041,0.04694065,0.02477381,0.01356019,0.05285877,0.03994502,0.03298015,0.01122006,0.04535493,0.02001026,0.02183813,0.00598268,-0.00222723,0.03196018,0.02881243,0.04019189]),
    (-301.13596041,92.33522300,1000,[-0.29788673,-0.29067273,-0.27209249,-0.26529950,-0.28286824,-0.29286709,-0.30328064,-0.32693268,-0.31046081,-0.32348008,-0.32198155,-0.29994094,-0.30372642,-0.31202724,-0.29173738,-0.25540786,-0.24566911,-0.26124476,-0.26500370,-0.27036641,-0.29511296,-0.26081048,-0.27802720,-0.24898043,-0.26230840,-0.25263583,-0.25880510,-0.25037270,-0.26573831,-0.26984810,-0.26185894,-0.26718201,-0.29338415,-0.29400492,-0.28458186,-0.28642229,-0.29275087,-0.28603959,-0.27268156,-0.30343335]),
    (1158.55380422,1355.22112542,1000,[1.19421584,1.19347614,1.12730083,1.14605302,1.13800726,1.11463297,1.11911721,1.13509756,1.11799094,1.09423027,1.09131330,1.11672491,1.09844234,1.08309206,1.07671931,1.11204330,1.13448164,1.17132087,1.16316680,1.19243230,1.20023448,1.21111713,1.20893007,1.17963299,1.19413694,1.21064352,1.24733511,1.26904340,1.28006608,1.25851874,1.27753864,1.28192127,1.27652514,1.24573107,1.25501743,1.22872857,1.21971511,1.22568972,1.26505698,1.22297667,1.24738862,1.25373229,1.27249188,1.26358816,1.23047782,1.23062637]),
    (-0.16081399,40.77990924,1000,[0.03438321,0.08041381,0.11721566,0.20111243,0.18700055,0.16703029,0.14104688,0.14021371]),
    (-0.24557949,14.43716756,1000,[0.06940049,0.06593760,0.05796057,0.10116011,0.09742547,0.06163917,0.04896575,0.05173055,0.03375788,0.02193889,0.02191073,0.00963532,0.04604100,0.00866888,-0.02311819,-0.04900991,-0.01813121,-0.00747893,0.00097976,-0.01134390,-0.04961616,-0.06931370,-0.08260499,-0.10397287]),
    (-0.08557543,13.60066905,1000,[0.07969592,0.06818581,0.10318462,0.10817154,0.07120982,0.04966336,0.06624672,0.06561326,0.05949641,0.04906536,0.02153222,0.00973282,0.01771127,-0.00278730,0.00175494,-0.00794585,0.00885012,0.00634639,0.03689816,0.03738184,0.05633408,0.04047474,0.08921691,0.12075099,0.09992652,0.08718119,0.03739840,0.02915954,0.02289998,0.05563641]),
    (507.26049340,281.59668549,1000,[0.46492564,0.45273927,0.46912912,0.45438791,0.45652657,0.44031184,0.44454156,0.43043803,0.42841242,0.42732136,0.42073258,0.40865200,0.41973832,0.38657116,0.38275204,0.35110390,0.34539663,0.35002156,0.34164318,0.35927075,0.40875072,0.38273686,0.37389864,0.39188585]),
    (-99.90475138,35.03136840,1000,[-0.22850170,-0.25085766,-0.25396497,-0.23354294,-0.20179694,-0.16405304,-0.22149862,-0.16834503,-0.20430559,-0.23079927,-0.25165568,-0.24711527,-0.24571613,-0.25044917,-0.19950202,-0.16727064,-0.19796078,-0.18374313,-0.15511400,-0.12893195,-0.11016591,-0.08177828,-0.04117288,-0.04261895,0.00057898,-0.04730096,-0.04399256,-0.02371963,-0.08257507,-0.09957959]),
    (-575.38387972,361.44319879,1000,[-0.37572829,-0.40850402,-0.36462746,-0.40346935,-0.43520844,-0.46913044,-0.47371199,-0.46503449,-0.53500709,-0.53002022,-0.55219727,-0.51264959,-0.49957242,-0.54202690,-0.46857694,-0.43896704]),
    (645.96395551,428.78150960,1000,[0.62156238,0.66660438,0.70464739,0.72317743,0.69583807,0.66366224,0.66916918,0.68611277,0.71142282,0.71403872,0.71128383,0.73213576]),
    (427.89128434,214.75082520,1000,[0.49442981,0.50853522,0.53785904,0.49244526,0.49737199,0.52474180,0.53881861,0.63061781,0.60478252,0.55346774,0.56359457,0.52902395]),
    (-1478.66515133,2197.02760054,1000,[-1.46146516,-1.49875671,-1.48225835,-1.47290098,-1.47298423,-1.46661859,-1.46250641,-1.41473912]),
    (-333.68166549,113.85162151,1000,[-0.33642044,-0.32729644,-0.32447025,-0.30873548,-0.30472189,-0.31243559,-0.31354786,-0.31610662,-0.32886963,-0.33450974,-0.37734899,-0.34479362]),
    (-904.52517581,831.01609514,1000,[-0.94435891,-0.93514574,-0.91961371,-0.92326852,-0.91076425,-0.89318092,-0.89973764,-0.89419482,-0.89054135,-0.86662969,-0.85899390,-0.89075249,-0.87851103,-0.90254686,-0.91206564,-0.87252728]),
]

REBAL_BAND  = 0.05
N_PAIRS     = len(PAIRS)
PRIOR_DAYS  = 1000

_state  = None
_pos    = np.zeros(N_INST, dtype=int)
_last_n = 0


def _load_prior():
    return {
        'sum1':    np.array([s[0] for s in PRIOR_STATE]),
        'sum2':    np.array([s[1] for s in PRIOR_STATE]),
        'count':   np.array([s[2] for s in PRIOR_STATE], dtype=int),
        'buf':     [np.array(s[3], dtype=float) for s in PRIOR_STATE],
        'buf_idx': np.zeros(N_PAIRS, dtype=int),
        'buf_full':np.ones(N_PAIRS, dtype=bool),
    }


def _init_fresh():
    return {
        'sum1':    np.zeros(N_PAIRS),
        'sum2':    np.zeros(N_PAIRS),
        'count':   np.zeros(N_PAIRS, dtype=int),
        'buf':     [np.zeros(PAIRS[p][2], dtype=float) for p in range(N_PAIRS)],
        'buf_idx': np.zeros(N_PAIRS, dtype=int),
        'buf_full':np.zeros(N_PAIRS, dtype=bool),
    }


def getMyPosition(prcSoFar):
    global _state, _pos, _last_n

    prc     = np.asarray(prcSoFar, dtype=float)
    n_inst, n_days = prc.shape
    p_today = prc[:, -1]
    max_sh  = np.floor(CAPS / p_today).astype(int)

    if n_days <= _last_n:
        _state = None
        _pos   = np.zeros(N_INST, dtype=int)

    if _state is None:
        if n_days >= PRIOR_DAYS:
            _state = _load_prior()
            log_p = np.log(prc)
            for t in range(PRIOR_DAYS, n_days - 1):
                for pidx, (i, j, zw, band, ag, expanding, beta, alpha) in enumerate(PAIRS):
                    s_now = log_p[i, t] - beta * log_p[j, t] - alpha
                    buf = _state['buf'][pidx]
                    buf[_state['buf_idx'][pidx]] = s_now
                    _state['buf_idx'][pidx] = (_state['buf_idx'][pidx] + 1) % zw
                    _state['sum1'][pidx] += s_now
                    _state['sum2'][pidx] += s_now ** 2
                    _state['count'][pidx] += 1
        else:
            _state = _init_fresh()

    _last_n = n_days
    log_today = np.log(p_today)

    # Compute ERP and signal for every pair
    erp      = np.zeros(N_PAIRS)
    f_signal = np.zeros(N_PAIRS)

    for pidx, (i, j, zw, band, ag, expanding, beta, alpha) in enumerate(PAIRS):
        if max(i, j) >= n_inst:
            continue

        s_now = log_today[i] - beta * log_today[j] - alpha
        buf   = _state['buf'][pidx]
        idx   = _state['buf_idx'][pidx]
        buf[idx] = s_now
        _state['buf_idx'][pidx] = (idx + 1) % zw
        _state['sum1'][pidx] += s_now
        _state['sum2'][pidx] += s_now ** 2
        _state['count'][pidx] += 1

        if not _state['buf_full'][pidx]:
            if _state['count'][pidx] >= zw:
                _state['buf_full'][pidx] = True
            else:
                continue

        ss = buf.std()
        if ss < 1e-12:
            continue

        n  = _state['count'][pidx]
        sm = _state['sum1'][pidx] / n if expanding else buf.mean()
        z  = (s_now - sm) / ss

        if abs(z) <= band:
            continue

        # ERP = kappa * |z| * sigma — expected 1-day mean reversion profit per unit
        erp[pidx]      = KAPPAS[pidx] * abs(z) * ss
        f_signal[pidx] = np.tanh(ag * z)

    # Winner-takes-all per asset: highest ERP wins full cap for that asset
    asset_best_erp  = {}
    asset_best_pair = {}

    for pidx in range(N_PAIRS):
        if erp[pidx] <= 0:
            continue
        i, j = PAIRS[pidx][0], PAIRS[pidx][1]
        for asset in (i, j):
            if asset >= n_inst:
                continue
            if asset not in asset_best_erp or erp[pidx] > asset_best_erp[asset]:
                asset_best_erp[asset]  = erp[pidx]
                asset_best_pair[asset] = pidx

    target_dollars = np.zeros(N_INST)
    for asset, pidx in asset_best_pair.items():
        i, j, zw, band, ag, expanding, beta, alpha = PAIRS[pidx]
        f = f_signal[pidx]
        if asset == i:
            target_dollars[i] -= f * CAPS[i]
        else:
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
