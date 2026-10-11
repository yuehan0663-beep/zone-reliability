from __future__ import annotations
import math,random
from collections import defaultdict
import numpy as np
import pandas as pd
from .constants import ZONE_EN
FEATURES_C4=["GR","AC","DEN","CNL","RT","pressure_coeff","depth_md","porosity_pct"]
METHODS=["C0_proxy","C0b_zone_mean","C1_offset","C2_affine","C3_isotonic","C4_small_ridge"]
REJECT_PCTS=[0,5,10,15,20,25,30,35,40]
RNG=random.Random(20260613)
# Source: run_within_zone_recalibration.py:54
def fnum(x):
    """fnum: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    try:
        if x is None or str(x) == '':
            return None
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None

# Source: run_within_zone_recalibration.py:60
def percentile(vals, q):
    """percentile: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    vals = sorted([v for v in vals if v is not None and math.isfinite(v)])
    if not vals:
        return None
    pos = (len(vals) - 1) * q / 100
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

# Source: run_within_zone_recalibration.py:66
def mean(vals):
    """mean: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float('nan')

# Source: run_within_zone_recalibration.py:110
def mae(y, p):
    """mae: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    return float(np.mean(np.abs(np.array(y) - np.array(p)))) if y else float('nan')

# Source: run_within_zone_recalibration.py:111
def r2(y, p):
    """r2: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    y = np.array(y, float)
    p = np.array(p, float)
    if len(y) < 2:
        return float('nan')
    ss = float(np.sum((y - p) ** 2))
    st = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - ss / st if st > 0 else float('nan')

# Source: run_within_zone_recalibration.py:115
def rankdata(a):
    """rankdata: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    pairs = sorted(((v, i) for i, v in enumerate(a)))
    ranks = [0] * len(a)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rk = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = rk
        i = j + 1
    return ranks

# Source: run_within_zone_recalibration.py:124
def spear(y, p):
    """spear: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    if len(y) < 3:
        return float('nan')
    ry = np.array(rankdata(y))
    rp = np.array(rankdata(p))
    return float(np.corrcoef(ry, rp)[0, 1]) if np.std(ry) > 0 and np.std(rp) > 0 else float('nan')

# Source: run_within_zone_recalibration.py:128
def mdict(y, p):
    """mdict: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    return {'pooled_R2': r2(y, p), 'weighted_MAE': mae(y, p), 'Spearman': spear(y, p), 'n': len(y)}

# Source: run_within_zone_recalibration.py:129
def fit_offset(p, y):
    """fit_offset: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    return mean([yy - pp for pp, yy in zip(p, y)])

# Source: run_within_zone_recalibration.py:130
def fit_affine(p, y):
    """fit_affine: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    x = np.array(p, float)
    yy = np.array(y, float)
    if len(x) < 2 or np.var(x) == 0:
        return (0.0, float(np.mean(yy)) if len(yy) else 0.0)
    a, b = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, yy, rcond=None)[0]
    return (float(a), float(b))

# Source: run_within_zone_recalibration.py:134
def pava_fit(x, y):
    """pava_fit: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    order = np.argsort(np.array(x))
    blocks = []
    for i in order:
        blocks.append([x[i], x[i], y[i], 1])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            b2 = blocks.pop()
            b1 = blocks.pop()
            n = b1[3] + b2[3]
            avg = (b1[2] * b1[3] + b2[2] * b2[3]) / n
            blocks.append([b1[0], b2[1], avg, n])
    bx = []
    by = []
    for lo, hi, v, n in blocks:
        bx += [lo, hi]
        by += [v, v]
    return (bx, by)

# Source: run_within_zone_recalibration.py:143
def pred_iso(p, model):
    """pred_iso: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    bx, by = model
    out = []
    for v in p:
        if v <= bx[0]:
            out.append(by[0])
            continue
        if v >= bx[-1]:
            out.append(by[-1])
            continue
        k = 0
        while k < len(bx) - 1 and bx[k + 1] < v:
            k += 1
        t = 0 if bx[k + 1] == bx[k] else (v - bx[k]) / (bx[k + 1] - bx[k])
        out.append(by[k] * (1 - t) + by[k + 1] * t)
    return out

# Source: run_within_zone_recalibration.py:152
def mat_builder(train, features):
    """mat_builder: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    meds = []
    mus = []
    sds = []
    for c in features:
        vals = [fnum(r.get(c)) for r in train if fnum(r.get(c)) is not None]
        med = percentile(vals, 50) if vals else 0.0
        arr = np.array(vals or [med], float)
        meds.append(med)
        mus.append(float(np.mean(arr)))
        sds.append(float(np.std(arr)) or 1.0)

    def mat(rs, proxy):
        """mat: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
        X = []
        for r in rs:
            row = [1.0, float(r[proxy])]
            for c, med, mu, sd in zip(features, meds, mus, sds):
                v = fnum(r.get(c))
                v = med if v is None else v
                row.append((v - mu) / sd)
            X.append(row)
        return np.array(X, float)
    return mat

# Source: run_within_zone_recalibration.py:167
def fit_c4(train, y, proxy):
    """fit_c4: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    mat = mat_builder(train, FEATURES_C4)
    X = mat(train, proxy)
    yy = np.array(y, float)
    lam = np.eye(X.shape[1]) * 2.0
    lam[0, 0] = 0
    return (mat, np.linalg.pinv(X.T @ X + lam) @ X.T @ yy)

# Source: run_within_zone_recalibration.py:170
def eval_zone(zrows, target, proxy, method):
    """eval_zone: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    obs = []
    pred = []
    wells = []
    samples = []
    risks = []
    zmeans = []
    for w in sorted(set((r['well'] for r in zrows))):
        tr = [r for r in zrows if r['well'] != w]
        te = [r for r in zrows if r['well'] == w]
        if not tr or not te:
            continue
        tp = [r[proxy] for r in tr]
        ty = [fnum(r[target]) for r in tr]
        ep = [r[proxy] for r in te]
        ey = [fnum(r[target]) for r in te]
        zm = mean(ty)
        if method == 'C0_proxy':
            yh = ep
        elif method == 'C0b_zone_mean':
            yh = [zm] * len(te)
        elif method == 'C1_offset':
            yh = [v + fit_offset(tp, ty) for v in ep]
        elif method == 'C2_affine':
            a, b = fit_affine(tp, ty)
            yh = [a * v + b for v in ep]
        elif method == 'C3_isotonic':
            yh = pred_iso(ep, pava_fit(tp, ty))
        elif method == 'C4_small_ridge':
            mat, beta = fit_c4(tr, ty, proxy)
            yh = list(mat(te, proxy) @ beta)
        med = percentile(tp, 50)
        iqr = max(1e-09, percentile(tp, 75) - percentile(tp, 25) if percentile(tp, 25) is not None else np.std(tp) or 1)
        obs += ey
        pred += [float(v) for v in yh]
        wells += [w] * len(te)
        samples += [r['sample_id'] for r in te]
        risks += [abs(v - med) / iqr for v in ep]
        zmeans += [zm] * len(te)
    return (obs, pred, wells, samples, risks, zmeans)

# Source: run_within_zone_recalibration.py:188
def bootstrap(y, p, wells, B=400):
    """bootstrap: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    uniq = sorted(set(wells))
    idx_by = {w: [i for i, a in enumerate(wells) if a == w] for w in uniq}
    res = defaultdict(list)
    if len(uniq) < 3:
        return {k: (float('nan'), float('nan')) for k in ['pooled_R2', 'weighted_MAE', 'Spearman', 'MAE_improvement_vs_zonemean']}
    for _ in range(B):
        idx = []
        for w in [RNG.choice(uniq) for _ in uniq]:
            idx += idx_by[w]
        yy = [y[i] for i in idx]
        pp = [p[i] for i in idx]
        m = mdict(yy, pp)
        for k in ['pooled_R2', 'weighted_MAE', 'Spearman']:
            if math.isfinite(m[k]):
                res[k].append(m[k])
    return {k: (percentile(v, 2.5), percentile(v, 97.5)) if len(v) >= 20 else (float('nan'), float('nan')) for k, v in res.items()}

# Source: run_within_zone_recalibration.py:199
def bootstrap_improvement(y, p, zm, wells, B=500):
    """bootstrap_improvement: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    uniq = sorted(set(wells))
    idx_by = {w: [i for i, a in enumerate(wells) if a == w] for w in uniq}
    vals = []
    if len(uniq) < 3:
        return (float('nan'), float('nan'))
    for _ in range(B):
        idx = []
        for w in [RNG.choice(uniq) for _ in uniq]:
            idx += idx_by[w]
        yy = [y[i] for i in idx]
        pp = [p[i] for i in idx]
        zz = [zm[i] for i in idx]
        base = mae(yy, zz)
        corr = mae(yy, pp)
        if base > 0:
            vals.append((base - corr) / base * 100)
    return (percentile(vals, 2.5), percentile(vals, 97.5))

# Source: run_within_zone_recalibration.py:209
def fmt(x):
    """fmt: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    try:
        return '' if not math.isfinite(float(x)) else f'{float(x):.6g}'
    except Exception:
        return ''

# Source: run_within_zone_recalibration.py:215
def drift(p, y):
    """drift: retained research implementation; paper §3.6. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    a, b = fit_affine(p, y)
    c0 = mae(y, p)
    c1 = mae(y, [v + fit_offset(p, y) for v in p])
    c2 = mae(y, [a * v + b for v in p])
    r = spear(y, p)
    if abs(a - 1) < 0.25 and c1 < c0 * 0.85:
        return 'location_shift'
    if c2 < c1 * 0.9 and abs(a - 1) >= 0.25:
        return 'scale_shift'
    if math.isfinite(r) and r > 0.4:
        return 'monotonic_shape_shift'
    return 'relationship_shift_or_weak_proxy'
# Source: run_gas_proxy_reproduction.py:214
def add_proxies(rows, params):
    """add_proxies: retained research implementation; paper §3.6, Methods-aligned gas correction. Inputs and outputs follow the named arguments and return statement; see module provenance."""
    byz = defaultdict(list)
    for r in rows:
        byz[r['zone']].append(r)
    baselines = {}
    for z, zs in byz.items():
        den_base = percentile([fnum(r.get('DEN')) for r in zs], 90)
        gr_cut = percentile([fnum(r.get('GR')) for r in zs], 25)
        ref = [r for r in zs if fnum(r.get('GR')) is not None and gr_cut is not None and (fnum(r.get('GR')) <= gr_cut) and (fnum(r.get('RT')) is not None) and (fnum(r.get('AC')) is not None)]
        if len(ref) < 5:
            ref = [r for r in zs if fnum(r.get('RT')) is not None and fnum(r.get('AC')) is not None]
        baselines[z] = {'DEN_baseline_p90': den_base, 'lowGR_cut_p25': gr_cut, 'RT_baseline_lowGR_median': percentile([fnum(r.get('RT')) for r in ref], 50), 'AC_baseline_lowGR_median': percentile([fnum(r.get('AC')) for r in ref], 50), 'porosity_median': percentile([fnum(r.get('porosity_pct')) for r in zs], 50), 'n_zone_rows': len(zs), 'n_lowGR_ref': len(ref)}
    for r in rows:
        z = r['zone']
        b = baselines[z]
        den = fnum(r.get('DEN'))
        rt = fnum(r.get('RT'))
        ac = fnum(r.get('AC'))
        depth = fnum(r.get('depth_md'))
        pc = fnum(r.get('pressure_coeff'))
        por = fnum(r.get('porosity_pct'))
        r['schmoker_toc_proxy'] = None
        if den is not None and b['DEN_baseline_p90'] is not None:
            r['schmoker_toc_proxy'] = max(0, (b['DEN_baseline_p90'] - den) * 1.378)
        r['passey_toc_proxy'] = None
        if rt and ac is not None and b['RT_baseline_lowGR_median'] and b['AC_baseline_lowGR_median']:
            dlogr = math.log10(rt / b['RT_baseline_lowGR_median']) + 0.02 * (ac - b['AC_baseline_lowGR_median'])
            r['passey_toc_proxy'] = max(0, dlogr * 10 ** (2.297 - 0.1688 * 11.5))
        r['gas_total_proxy_quant'] = None
        r['gas_ads_proxy'] = None
        r['gas_free_proxy'] = None
        r['porosity_imputed_flag'] = '0'
        if depth is not None and pc is not None:
            pressure_mpa = max(0.001, pc * depth * 0.0098)
            ql_mgg = params['q_L_mg_g']
            ql_m3t = ql_mgg * 1.3975
            g_ads = ql_m3t * pressure_mpa / (pressure_mpa + params['P_L_MPa'])
            phi = por
            if phi is None:
                phi = b['porosity_median']
                r['porosity_imputed_flag'] = '1'
            if phi is not None:
                phi_frac = phi / 100.0
                temp_k = 293.15 + params['geothermal_gradient_C_per_m'] * depth
                bg = 0.00035096 * params['Z'] * temp_k / pressure_mpa
                g_free = phi_frac * params['S_g'] / (params['rho_rock_t_m3'] * bg)
                r['gas_ads_proxy'] = g_ads
                r['gas_free_proxy'] = g_free
                r['gas_total_proxy_quant'] = g_ads + g_free
    return baselines

def load_gas_params(path=None):
    """Load fixed Methods parameters from JSON; allow declared sensitivity inputs."""
    import json
    from pathlib import Path
    path=Path(path) if path else Path(__file__).with_name('gas_proxy.json')
    return json.loads(path.read_text(encoding='utf-8'))

def evaluate_zone(rows,target,proxy,method):
    """Evaluate one zone only under within-zone LOWO (§3.6).

    Rejects cross-zone labels rather than borrowing them. C4 has intercept,
    unstandardized proxy, eight original covariates, and lambda=2 penalty.
    """
    if len({r['zone'] for r in rows})!=1:raise ValueError('Exactly one zone is required; cross-zone borrowing is prohibited')
    if method not in METHODS:raise ValueError(method)
    return eval_zone(rows,target,proxy,method)

def empty_recalibration():
    """Header-preserving outputs when physical recalibration is skipped."""
    common=['target','proxy','zone','zone_en']
    return {
        'metrics':pd.DataFrame(columns=common+['method','n','wells','pooled_R2','weighted_MAE','Spearman',
            'pooled_R2_ci_low','pooled_R2_ci_high','weighted_MAE_ci_low','weighted_MAE_ci_high','Spearman_ci_low','Spearman_ci_high']),
        'benefits':pd.DataFrame(columns=common+['wells','best_method','best_MAE','zone_mean_MAE',
            'relative_MAE_improvement_vs_zonemean_pct','improvement_ci_low_pct','improvement_ci_high_pct','benefit_class','small_sample_note']),
        'drift':pd.DataFrame(columns=common+['n','wells','spearman_proxy_gold','affine_slope','affine_intercept','drift_type']),
        'predictions':pd.DataFrame(columns=common+['well','sample_id','method','observed','predicted',
            'zone_mean_prediction','rejection_score','abs_error','zone_mean_abs_error']),
        'rejection':pd.DataFrame(columns=common+['best_method','reject_pct','retain_pct','retained_n','retained_wells',
            'corrected_MAE','zone_mean_MAE','relative_improvement_pct','same_retained_subset','risk_score'])}


def recalibration_ladder(df,params=None):
    """Run original C0–C4 order and continuous bootstrap stream (§3.6).

    Returns metrics, selected benefits, drift typing, predictions and rejection.
    Density and porosity baselines use full-zone inputs, never target labels.
    """
    import warnings
    missing=[c for c in ['pressure_coeff','porosity_pct'] if c not in df]
    if missing and 'total_gas_m3_t' in df and df.total_gas_m3_t.notna().any():
        warnings.warn('Gas physical proxy skipped: missing '+', '.join(missing)+
                      '; gas local_correction_robust is False.', UserWarning, stacklevel=2)
    global RNG
    RNG=random.Random(20260613)
    rows=[{k:(None if pd.isna(v) else v) for k,v in r.items()} for r in df.to_dict("records")]
    add_proxies(rows,params or load_gas_params())
    targets = [('TOC', 'toc_pct', ['schmoker_toc_proxy', 'passey_toc_proxy']), ('total_gas', 'total_gas_m3_t', ['gas_total_proxy_quant'])]
    results = []
    preds = []
    drifts = []
    rej = []
    benefit = []
    for tname, tcol, proxies in targets:
        if missing and tname == 'total_gas':
            continue
        for proxy in proxies:
            for z in sorted(set((r['zone'] for r in rows))):
                zrows = [r for r in rows if r['zone'] == z and fnum(r.get(tcol)) is not None and (r.get(proxy) is not None) and math.isfinite(r.get(proxy))]
                wells = sorted(set((r['well'] for r in zrows)))
                if len(wells) < 3 or len(zrows) < 5:
                    continue
                p = [r[proxy] for r in zrows]
                y = [fnum(r[tcol]) for r in zrows]
                a, b = fit_affine(p, y)
                drifts.append({'target': tname, 'proxy': proxy, 'zone': z, 'zone_en': ZONE_EN.get(z, z), 'n': len(zrows), 'wells': len(wells), 'spearman_proxy_gold': fmt(spear(y, p)), 'affine_slope': fmt(a), 'affine_intercept': fmt(b), 'drift_type': drift(p, y)})
                cache = {}
                for m in METHODS:
                    yy, pp, ww, ss, rr, zm = eval_zone(zrows, tcol, proxy, m)
                    cache[m] = (yy, pp, ww, ss, rr, zm)
                    md = mdict(yy, pp)
                    ci = bootstrap(yy, pp, ww)
                    rec = {'target': tname, 'proxy': proxy, 'zone': z, 'zone_en': ZONE_EN.get(z, z), 'method': m, 'n': len(yy), 'wells': len(set(ww))}
                    for k, v in md.items():
                        rec[k] = fmt(v)
                    for k, (lo, hi) in ci.items():
                        rec[k + '_ci_low'] = fmt(lo)
                        rec[k + '_ci_high'] = fmt(hi)
                    results.append(rec)
                    for sid, w, obs, pr, ris, zmp in zip(ss, ww, yy, pp, rr, zm):
                        preds.append({'target': tname, 'proxy': proxy, 'zone': z, 'zone_en': ZONE_EN.get(z, z), 'well': w, 'sample_id': sid, 'method': m, 'observed': fmt(obs), 'predicted': fmt(pr), 'zone_mean_prediction': fmt(zmp), 'rejection_score': fmt(ris), 'abs_error': fmt(abs(obs - pr)), 'zone_mean_abs_error': fmt(abs(obs - zmp))})
                cand = [r for r in results if r['target'] == tname and r['proxy'] == proxy and (r['zone'] == z) and (r['method'] in ['C1_offset', 'C2_affine', 'C3_isotonic', 'C4_small_ridge'])]
                best = min(cand, key=lambda r: float(r['weighted_MAE']))
                yy, pp, ww, ss, rr, zm = cache[best['method']]
                base = mae(yy, zm)
                corr = mae(yy, pp)
                imp = (base - corr) / base * 100 if base > 0 else float('nan')
                ilo, ihi = bootstrap_improvement(yy, pp, zm, ww)
                benefit_class = 'CI-robust' if imp > 1 and ilo > 0 else 'directional only' if imp > 1 else 'no directional benefit'
                benefit.append({'target': tname, 'proxy': proxy, 'zone': z, 'zone_en': ZONE_EN.get(z, z), 'wells': len(set(ww)), 'best_method': best['method'], 'best_MAE': fmt(corr), 'zone_mean_MAE': fmt(base), 'relative_MAE_improvement_vs_zonemean_pct': fmt(imp), 'improvement_ci_low_pct': fmt(ilo), 'improvement_ci_high_pct': fmt(ihi), 'benefit_class': benefit_class, 'small_sample_note': 'North gas has 5 wells; interpret cautiously' if tname == 'total_gas' and z == '北' else ''})
                order = sorted(range(len(yy)), key=lambda i: rr[i])
                for pct in REJECT_PCTS:
                    keep = max(1, int(round(len(order) * (100 - pct) / 100)))
                    idx = order[:keep]
                    cy = [yy[i] for i in idx]
                    cp = [pp[i] for i in idx]
                    cz = [zm[i] for i in idx]
                    bmae = mae(cy, cz)
                    cmae = mae(cy, cp)
                    ri = (bmae - cmae) / bmae * 100 if bmae > 0 else float('nan')
                    rej.append({'target': tname, 'proxy': proxy, 'zone': z, 'zone_en': ZONE_EN.get(z, z), 'best_method': best['method'], 'reject_pct': pct, 'retain_pct': 100 - pct, 'retained_n': len(idx), 'retained_wells': len(set((ww[i] for i in idx))), 'corrected_MAE': fmt(cmae), 'zone_mean_MAE': fmt(bmae), 'relative_improvement_pct': fmt(ri), 'same_retained_subset': 'yes', 'risk_score': 'proxy_distance_from_training_distribution'})
    tables={"metrics":pd.DataFrame(results),"benefits":pd.DataFrame(benefit),"drift":pd.DataFrame(drifts),"predictions":pd.DataFrame(preds),"rejection":pd.DataFrame(rej)}
    if missing:
        for key,empty in empty_recalibration().items():
            if tables[key].empty:
                tables[key]=empty
    return tables
