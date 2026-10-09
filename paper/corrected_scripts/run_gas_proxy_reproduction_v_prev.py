# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"PROJECT_ROOT_PLACEHOLDER/evo_workspace")
gas_proxy_sensitivity = ROOT / "\u5199\u4f5c_\u88689" / "supplementary_analysis" / "gas_proxy_sensitivity"
PRIVATE = gas_proxy_sensitivity / "internal_review"
SRC = ROOT / "within_zone_recalibration_analysis_derived_dataset"
DATA = ROOT / "Input Program" / "derived_dataset.xlsx"
SHEET = "\u6570\u636e"

ZONE_EN = {"\u5317": "North", "\u4e2d": "Central", "\u5357": "South"}
FEATURES_C4 = ["GR", "AC", "DEN", "CNL", "RT", "pressure_coeff", "depth_md", "porosity_pct"]
METHODS = ["C0_proxy", "C0b_zone_mean", "C1_offset", "C2_affine", "C3_isotonic", "C4_small_ridge"]

BASE_PARAMS = {
    "q_L_mg_g": 0.39,
    "P_L_MPa": 7.60,
    "S_g": 0.50,
    "Z": 0.90,
    "geothermal_gradient_C_per_m": 0.03,
    "rho_rock_t_m3": 2.65,
    "free_gas_conservative_scale": 0.01,
}

GATE_EXPECTED = [
    ("total_gas", "North", "gas_total_proxy_quant", "C4_small_ridge", 22.2451, "acquisition_guidance"),
    ("total_gas", "Central", "gas_total_proxy_quant", "C2_affine", -1.60144, "acquisition_guidance"),
    ("total_gas", "South", "gas_total_proxy_quant", "C3_isotonic", 1.86839, "acquisition_guidance"),
    ("TOC", "North", "schmoker_toc_proxy", "C1_offset", 7.3887, "acquisition_guidance"),
]


def ensure_dirs() -> None:
    gas_proxy_sensitivity.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)


def backup_existing() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PRIVATE / f"deprecated_old_gas_proxy_sensitivity_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for item in gas_proxy_sensitivity.iterdir():
        if item.name == "internal_review":
            continue
        if item.is_file():
            shutil.copy2(item, backup / item.name)
    for item in PRIVATE.iterdir():
        if item.name.startswith("deprecated_old_gas_proxy_sensitivity"):
            continue
        if item.name == Path(__file__).name:
            continue
        dest = backup / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
    return backup


def fnum(x):
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def percentile(vals, q):
    vals = sorted([v for v in vals if v is not None and math.isfinite(v)])
    if not vals:
        return None
    pos = (len(vals) - 1) * q / 100
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return vals[lo] if lo == hi else vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def rankdata(a):
    pairs = sorted((v, i) for i, v in enumerate(a))
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


def spearman(y, p):
    if len(y) < 3:
        return float("nan")
    ry = np.array(rankdata(y), float)
    rp = np.array(rankdata(p), float)
    return float(np.corrcoef(ry, rp)[0, 1]) if np.std(ry) > 0 and np.std(rp) > 0 else float("nan")


def r2(y, p):
    y = np.array(y, float)
    p = np.array(p, float)
    if len(y) < 2:
        return float("nan")
    ss = float(np.sum((y - p) ** 2))
    st = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - ss / st if st > 0 else float("nan")


def mae(y, p):
    return float(np.mean(np.abs(np.array(y, float) - np.array(p, float)))) if y else float("nan")


def fit_offset(p, y):
    return mean([yy - pp for pp, yy in zip(p, y)])


def fit_affine(p, y):
    x = np.array(p, float)
    yy = np.array(y, float)
    if len(x) < 2 or np.var(x) == 0:
        return 0.0, float(np.mean(yy)) if len(yy) else 0.0
    a, b = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, yy, rcond=None)[0]
    return float(a), float(b)


def pava_fit(x, y):
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
    bx, by = [], []
    for lo, hi, v, _n in blocks:
        bx += [lo, hi]
        by += [v, v]
    return bx, by


def pred_iso(p, model):
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


def mat_builder(train, features):
    meds, mus, sds = [], [], []
    for c in features:
        vals = [fnum(r.get(c)) for r in train if fnum(r.get(c)) is not None]
        med = percentile(vals, 50) if vals else 0.0
        arr = np.array(vals or [med], float)
        meds.append(med)
        mus.append(float(np.mean(arr)))
        sds.append(float(np.std(arr)) or 1.0)

    def mat(rs, proxy):
        x = []
        for r in rs:
            row = [1.0, float(r[proxy])]
            for c, med, mu, sd in zip(features, meds, mus, sds):
                v = fnum(r.get(c))
                v = med if v is None else v
                row.append((v - mu) / sd)
            x.append(row)
        return np.array(x, float)

    return mat


def fit_c4(train, y, proxy):
    mat = mat_builder(train, FEATURES_C4)
    x = mat(train, proxy)
    yy = np.array(y, float)
    lam = np.eye(x.shape[1]) * 2.0
    lam[0, 0] = 0
    return mat, np.linalg.pinv(x.T @ x + lam) @ x.T @ yy


def add_proxies(rows, params):
    byz = defaultdict(list)
    for r in rows:
        byz[r["zone"]].append(r)
    baselines = {}
    for z, zs in byz.items():
        den_base = percentile([fnum(r.get("DEN")) for r in zs], 90)
        gr_cut = percentile([fnum(r.get("GR")) for r in zs], 25)
        ref = [
            r
            for r in zs
            if fnum(r.get("GR")) is not None
            and gr_cut is not None
            and fnum(r.get("GR")) <= gr_cut
            and fnum(r.get("RT")) is not None
            and fnum(r.get("AC")) is not None
        ]
        if len(ref) < 5:
            ref = [r for r in zs if fnum(r.get("RT")) is not None and fnum(r.get("AC")) is not None]
        baselines[z] = {
            "DEN_baseline_p90": den_base,
            "lowGR_cut_p25": gr_cut,
            "RT_baseline_lowGR_median": percentile([fnum(r.get("RT")) for r in ref], 50),
            "AC_baseline_lowGR_median": percentile([fnum(r.get("AC")) for r in ref], 50),
            "porosity_median": percentile([fnum(r.get("porosity_pct")) for r in zs], 50),
            "n_zone_rows": len(zs),
            "n_lowGR_ref": len(ref),
        }
    for r in rows:
        z = r["zone"]
        b = baselines[z]
        den = fnum(r.get("DEN"))
        rt = fnum(r.get("RT"))
        ac = fnum(r.get("AC"))
        depth = fnum(r.get("depth_md"))
        pc = fnum(r.get("pressure_coeff"))
        por = fnum(r.get("porosity_pct"))
        toc = fnum(r.get("toc_pct"))
        r["schmoker_toc_proxy"] = None
        if den is not None and b["DEN_baseline_p90"] is not None:
            r["schmoker_toc_proxy"] = max(0, (b["DEN_baseline_p90"] - den) * 1.378)
        r["passey_toc_proxy"] = None
        if rt and ac is not None and b["RT_baseline_lowGR_median"] and b["AC_baseline_lowGR_median"]:
            dlogr = math.log10(rt / b["RT_baseline_lowGR_median"]) + 0.02 * (ac - b["AC_baseline_lowGR_median"])
            r["passey_toc_proxy"] = max(0, dlogr * (10 ** (2.297 - 0.1688 * 11.5)))
        r["gas_total_proxy_quant"] = None
        r["gas_ads_proxy"] = None
        r["gas_free_proxy"] = None
        r["porosity_imputed_flag"] = "0"
        if depth is not None and pc is not None:
            pressure_mpa = max(0.001, pc * depth * 0.0098)
            toc_for_q = toc if toc is not None else r["schmoker_toc_proxy"]
            if toc_for_q is None:
                toc_for_q = 2.0
            ql_mgg = max(0.12, min(0.80, params["q_L_mg_g"] * (toc_for_q / 2.0)))
            ql_m3t = ql_mgg * 1.3975
            g_ads = ql_m3t * pressure_mpa / (pressure_mpa + params["P_L_MPa"])
            phi = por
            if phi is None:
                phi = b["porosity_median"]
                r["porosity_imputed_flag"] = "1"
            if phi is not None:
                phi_frac = max(0.0001, phi / 100.0)
                temp_k = 293.15 + params["geothermal_gradient_C_per_m"] * depth
                bg = 0.00035096 * params["Z"] * temp_k / pressure_mpa
                g_free = (phi_frac * params["S_g"]) / (params["rho_rock_t_m3"] * bg)
                g_free = g_free * params["free_gas_conservative_scale"]
                r["gas_ads_proxy"] = g_ads
                r["gas_free_proxy"] = g_free
                r["gas_total_proxy_quant"] = g_ads + g_free
    return baselines


def eval_zone(zrows, target, proxy, method):
    obs, pred, wells, samples, risks, zmeans = [], [], [], [], [], []
    for w in sorted(set(r["well"] for r in zrows)):
        tr = [r for r in zrows if r["well"] != w]
        te = [r for r in zrows if r["well"] == w]
        if not tr or not te:
            continue
        tp = [r[proxy] for r in tr]
        ty = [fnum(r[target]) for r in tr]
        ep = [r[proxy] for r in te]
        ey = [fnum(r[target]) for r in te]
        zm = mean(ty)
        if method == "C0_proxy":
            yh = ep
        elif method == "C0b_zone_mean":
            yh = [zm] * len(te)
        elif method == "C1_offset":
            yh = [v + fit_offset(tp, ty) for v in ep]
        elif method == "C2_affine":
            a, b = fit_affine(tp, ty)
            yh = [a * v + b for v in ep]
        elif method == "C3_isotonic":
            yh = pred_iso(ep, pava_fit(tp, ty))
        elif method == "C4_small_ridge":
            mat, beta = fit_c4(tr, ty, proxy)
            yh = list(mat(te, proxy) @ beta)
        else:
            raise ValueError(method)
        med = percentile(tp, 50)
        iqr = max(1e-9, (percentile(tp, 75) - percentile(tp, 25)) if percentile(tp, 25) is not None else np.std(tp) or 1)
        obs += ey
        pred += [float(v) for v in yh]
        wells += [w] * len(te)
        samples += [r.get("sample_id") for r in te]
        risks += [abs(v - med) / iqr for v in ep]
        zmeans += [zm] * len(te)
    return obs, pred, wells, samples, risks, zmeans


def load_rows():
    df = pd.read_excel(DATA, sheet_name=SHEET)
    rows = []
    for rec in df.to_dict(orient="records"):
        out = {}
        for k, v in rec.items():
            out[str(k)] = None if pd.isna(v) else v
        rows.append(out)
    return rows


def compute_all(params, include_predictions=False):
    rows = load_rows()
    baselines = add_proxies(rows, params)
    targets = [("TOC", "toc_pct", "schmoker_toc_proxy"), ("total_gas", "total_gas_m3_t", "gas_total_proxy_quant")]
    metrics = []
    predictions = []
    for tname, tcol, proxy in targets:
        for z in sorted(set(r["zone"] for r in rows)):
            zrows = [
                r
                for r in rows
                if r["zone"] == z
                and fnum(r.get(tcol)) is not None
                and r.get(proxy) is not None
                and math.isfinite(float(r.get(proxy)))
            ]
            wells = sorted(set(r["well"] for r in zrows))
            if len(wells) < 3 or len(zrows) < 5:
                continue
            for method in METHODS:
                yy, pp, ww, ss, rr, zm = eval_zone(zrows, tcol, proxy, method)
                base = mae(yy, zm)
                corr = mae(yy, pp)
                imp = (base - corr) / base * 100 if base > 0 else float("nan")
                metrics.append(
                    {
                        "target": tname,
                        "zone": z,
                        "zone_en": ZONE_EN.get(z, z),
                        "proxy": proxy,
                        "method": method,
                        "n_points": len(yy),
                        "n_wells": len(set(ww)),
                        "baseline_zone_mean_MAE": base,
                        "method_MAE": corr,
                        "relative_MAE_improvement_vs_zonemean_pct": imp,
                        "pooled_R2": r2(yy, pp),
                        "Spearman": spearman(yy, pp),
                    }
                )
                if include_predictions:
                    for sid, w, obs, pred, zmp, risk in zip(ss, ww, yy, pp, zm, rr):
                        predictions.append(
                            {
                                "target": tname,
                                "zone": z,
                                "zone_en": ZONE_EN.get(z, z),
                                "proxy": proxy,
                                "method": method,
                                "well": w,
                                "sample_id": sid,
                                "observed": obs,
                                "predicted": pred,
                                "zone_mean_prediction": zmp,
                                "abs_error": abs(obs - pred),
                                "zone_mean_abs_error": abs(obs - zmp),
                                "rejection_score": risk,
                            }
                        )
    return pd.DataFrame(metrics), pd.DataFrame(predictions), baselines


def load_expected_from_sources():
    acq = pd.read_csv(SRC / "acquisition_guidance.csv", encoding="utf-8-sig")
    benefit = pd.read_csv(SRC / "benefit_summary.csv", encoding="utf-8-sig")
    out = []
    for target, zone_en, proxy, method, expected, source in GATE_EXPECTED:
        if source == "acquisition_guidance":
            row = acq[
                (acq["target"].eq(target))
                & (acq["zone_en"].eq(zone_en))
                & (acq["proxy"].eq(proxy))
                & (acq["best_method"].eq(method))
            ]
            source_value = float(row.iloc[0]["relative_MAE_improvement_vs_zonemean_pct"]) if len(row) else float("nan")
        else:
            row = benefit[
                (benefit["target"].eq(target))
                & (benefit["zone_en"].eq(zone_en))
                & (benefit["proxy"].eq(proxy))
                & (benefit["method"].eq(method))
            ]
            source_value = float(row.iloc[0]["relative_MAE_improvement_vs_zonemean_pct"]) if len(row) else float("nan")
        out.append((target, zone_en, proxy, method, expected, source_value, source))
    return out


def gate_check(metrics):
    source_expected = load_expected_from_sources()
    rows = []
    for target, zone_en, proxy, method, expected, source_value, source in source_expected:
        row = metrics[
            (metrics["target"].eq(target))
            & (metrics["zone_en"].eq(zone_en))
            & (metrics["proxy"].eq(proxy))
            & (metrics["method"].eq(method))
        ]
        reproduced = float(row.iloc[0]["relative_MAE_improvement_vs_zonemean_pct"]) if len(row) else float("nan")
        delta_expected = reproduced - expected
        delta_source = reproduced - source_value
        rows.append(
            {
                "check": f"{zone_en} {target} / {method}",
                "target": target,
                "zone_en": zone_en,
                "proxy": proxy,
                "method": method,
                "expected_anchor_pct": expected,
                "source_pct": source_value,
                "reproduced_pct": reproduced,
                "delta_vs_anchor_pct": delta_expected,
                "delta_vs_source_pct": delta_source,
                "source_file": source,
                "status": "PASS" if abs(delta_expected) <= 0.02 and abs(delta_source) <= 0.02 else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def write_params():
    rows = [
        {
            "parameter": "q_L",
            "value": BASE_PARAMS["q_L_mg_g"],
            "unit": "mg/g",
            "source": "W148 moisture-equilibrated 30 C Langmuir anchor; literature/DOI pending author confirmation",
            "gold_label_used": "N",
            "note": "TOC only scales adsorption capacity during proxy construction; gold total gas is not used for parameter fitting.",
        },
        {
            "parameter": "P_L",
            "value": BASE_PARAMS["P_L_MPa"],
            "unit": "MPa",
            "source": "W148 moisture-equilibrated 30 C Langmuir anchor; literature/DOI pending author confirmation",
            "gold_label_used": "N",
            "note": "Used in Langmuir adsorption term.",
        },
        {
            "parameter": "pressure",
            "value": "pressure_coeff * depth_md * 0.0098",
            "unit": "MPa",
            "source": "locked proxy construction script",
            "gold_label_used": "N",
            "note": "pressure_coeff is the corrected present-day pressure coefficient.",
        },
        {
            "parameter": "S_g",
            "value": BASE_PARAMS["S_g"],
            "unit": "fraction",
            "source": "locked free-gas proxy assumption",
            "gold_label_used": "N",
            "note": "Conservative gas saturation proxy.",
        },
        {
            "parameter": "Z",
            "value": BASE_PARAMS["Z"],
            "unit": "dimensionless",
            "source": "locked free-gas proxy assumption",
            "gold_label_used": "N",
            "note": "Gas compressibility factor for rough formation-volume conversion.",
        },
        {
            "parameter": "geothermal_gradient",
            "value": BASE_PARAMS["geothermal_gradient_C_per_m"],
            "unit": "C/m",
            "source": "locked free-gas proxy assumption",
            "gold_label_used": "N",
            "note": "Temperature model uses 293.15 K plus gradient times depth.",
        },
        {
            "parameter": "porosity_missing",
            "value": "zone median",
            "unit": "rule",
            "source": "locked proxy construction script",
            "gold_label_used": "N",
            "note": "Imputed internally and flagged; not tuned on gas labels.",
        },
        {
            "parameter": "G_total_proxy",
            "value": "G_ads + G_free",
            "unit": "m3/t proxy",
            "source": "locked proxy construction script",
            "gold_label_used": "N",
            "note": "C1-C4 recalibration aligns proxy to measured total gas inside training folds only.",
        },
    ]
    pd.DataFrame(rows).to_csv(gas_proxy_sensitivity / "gas_proxy_params.csv", index=False, encoding="utf-8-sig")


def sensitivity():
    multipliers = [0.8, 1.0, 1.2]
    rows = []
    for qmul in multipliers:
        for plmul in multipliers:
            for sgmul in multipliers:
                for zmul in multipliers:
                    params = dict(BASE_PARAMS)
                    params["q_L_mg_g"] = BASE_PARAMS["q_L_mg_g"] * qmul
                    params["P_L_MPa"] = BASE_PARAMS["P_L_MPa"] * plmul
                    params["S_g"] = BASE_PARAMS["S_g"] * sgmul
                    params["Z"] = BASE_PARAMS["Z"] * zmul
                    metrics, _preds, _bases = compute_all(params, include_predictions=False)
                    gas = metrics[metrics["target"].eq("total_gas") & metrics["method"].isin(["C1_offset", "C2_affine", "C3_isotonic", "C4_small_ridge"])]
                    for _, r in gas.iterrows():
                        rows.append(
                            {
                                "q_L_mg_g": params["q_L_mg_g"],
                                "P_L_MPa": params["P_L_MPa"],
                                "S_g": params["S_g"],
                                "Z": params["Z"],
                                "q_L_multiplier": qmul,
                                "P_L_multiplier": plmul,
                                "S_g_multiplier": sgmul,
                                "Z_multiplier": zmul,
                                "target": r["target"],
                                "zone_en": r["zone_en"],
                                "proxy": r["proxy"],
                                "method": r["method"],
                                "n_points": int(r["n_points"]),
                                "n_wells": int(r["n_wells"]),
                                "baseline_zone_mean_MAE": r["baseline_zone_mean_MAE"],
                                "method_MAE": r["method_MAE"],
                                "benefit_pct": r["relative_MAE_improvement_vs_zonemean_pct"],
                                "is_anchor_combo": r["zone_en"] == "North" and r["method"] == "C4_small_ridge",
                            }
                        )
    sens = pd.DataFrame(rows)
    sens.to_csv(gas_proxy_sensitivity / "gas_proxy_sensitivity.csv", index=False, encoding="utf-8-sig")
    focus = sens[sens["is_anchor_combo"]].copy()
    return sens, focus


def fmt(x, digits=4):
    if x is None or not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def write_reports(gate, metrics, sens, focus, backup, baselines):
    internal = [
        "# gas_proxy_sensitivity internal traceability",
        "",
        f"- Workspace: `{ROOT}`",
        f"- Data workbook: `{DATA}`",
        f"- Canonical proxy implementation inspected: `{ROOT / 'run_within_zone_recalibration.py'}`",
        f"- Canonical quant-recal outputs: `{SRC}`",
        f"- Previous gas_proxy_sensitivity files copied to: `{backup}`",
        "- Current `benefit_summary.csv` is TOC-only because it was overwritten by the later real-library full-condition script; gas anchors are preserved in `acquisition_guidance.csv` and `zone_recalibration_results.csv`.",
        "",
        "## Zone baselines",
    ]
    for z, b in baselines.items():
        internal.append(f"- {ZONE_EN.get(z,z)}: {b}")
    (PRIVATE / "gas_proxy_sensitivity_internal_traceability.md").write_text("\n".join(internal) + "\n", encoding="utf-8")

    gate.to_csv(PRIVATE / "gas_proxy_sensitivity_reproduction_gate.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(PRIVATE / "gas_proxy_sensitivity_recomputed_all_method_metrics.csv", index=False, encoding="utf-8-sig")

    focus_min = float(focus["benefit_pct"].min())
    focus_max = float(focus["benefit_pct"].max())
    focus_pos = int((focus["benefit_pct"] > 0).sum())
    focus_total = int(len(focus))
    gate_status = "PASS" if gate["status"].eq("PASS").all() else "FAIL"

    public_lines = [
        "# gas_proxy_sensitivity report - gas proxy specification and sensitivity",
        "",
        f"Gate status: **{gate_status}**.",
        "",
        "## Reproduction gate",
        "",
        "| Check | Expected % | Reproduced % | Delta % | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in gate.iterrows():
        public_lines.append(
            f"| {r['check']} | {fmt(r['expected_anchor_pct'])} | {fmt(r['reproduced_pct'])} | {fmt(r['delta_vs_anchor_pct'], 6)} | {r['status']} |"
        )
    public_lines += [
        "",
        "The gate requires the three gas anchors and the North TOC sanity check to pass together. The TOC sanity check confirms that the gas-proxy rerun did not disturb the locked TOC pipeline.",
        "",
        "## Locked gas proxy specification",
        "",
        "- Adsorption anchor: W148 moisture-equilibrated 30 C Langmuir anchor, q_L=0.39 mg/g and P_L=7.60 MPa; literature/DOI remains pending author confirmation.",
        "- Pressure: pressure_coeff x depth_md x 0.0098 MPa.",
        "- Free gas: porosity-based term with S_g=0.5, Z=0.90, geothermal gradient 0.03 C/m, and zone-median porosity imputation when missing.",
        "- Total proxy: G_total_proxy = G_ads + G_free.",
        "- Gold total gas was not used to set proxy parameters; gold labels enter only within training folds for C1-C4 recalibration.",
        "",
        "## Sensitivity result",
        "",
        f"- Sensitivity grid: q_L, P_L, S_g, and Z each at 0.8x, 1.0x, and 1.2x baseline; {len(sens)} zone-method rows.",
        f"- North gas / C4 small ridge stayed positive in {focus_pos}/{focus_total} grid points.",
        f"- North gas / C4 benefit range across the +/-20% grid: {focus_min:.4f}% to {focus_max:.4f}%.",
        "- The sensitivity table reports benefit% for every gas zone x C1-C4 method at every grid point; it is not an enrichment ranking.",
        "",
        "## Claim boundary",
        "",
        "- North gas C4 remains directional only and small-N; it should not be framed as CI-robust.",
        "- Central gas C2 remains unsupported at the baseline anchor.",
        "- South gas C3 is weak directional evidence only.",
        "- Author confirmation is still required for the W148 literature source/DOI/page metadata before public submission.",
        "",
        "## Outputs",
        "",
        "- `gas_proxy_params.csv`",
        "- `gas_proxy_sensitivity.csv`",
        "- `internal_review/gas_proxy_sensitivity_reproduction_gate.csv`",
        "- `internal_review/gas_proxy_sensitivity_recomputed_all_method_metrics.csv`",
        "- `internal_review/gas_proxy_sensitivity_internal_traceability.md`",
    ]
    if gate_status != "PASS":
        public_lines.insert(4, "gas_proxy_sensitivity stopped before accepting sensitivity conclusions because the reproduction gate failed.")
    (gas_proxy_sensitivity / "gas_proxy_sensitivity_report.md").write_text("\n".join(public_lines) + "\n", encoding="utf-8")


def anonymization_scan():
    banned = []  # public package: true-name deny-list removed after de-identification
    rows = []
    for path in [gas_proxy_sensitivity / "gas_proxy_params.csv", gas_proxy_sensitivity / "gas_proxy_sensitivity.csv", gas_proxy_sensitivity / "gas_proxy_sensitivity_report.md"]:
        text = path.read_text(encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8")
        text_lower = text.lower()
        hits = [b for b in banned if b.lower() in text_lower]
        rows.append({"file": path.name, "hit_count": len(hits), "hits": ";".join(hits), "status": "PASS" if not hits else "FAIL"})
    pd.DataFrame(rows).to_csv(PRIVATE / "gas_proxy_sensitivity_public_anonymization_scan.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    backup = backup_existing()
    write_params()
    metrics, preds, baselines = compute_all(BASE_PARAMS, include_predictions=True)
    preds.to_csv(PRIVATE / "gas_proxy_sensitivity_recomputed_predictionsinternal_review.csv", index=False, encoding="utf-8-sig")
    gate = gate_check(metrics)
    if not gate["status"].eq("PASS").all():
        gate.to_csv(PRIVATE / "gas_proxy_sensitivity_reproduction_gate.csv", index=False, encoding="utf-8-sig")
        (gas_proxy_sensitivity / "gas_proxy_sensitivity_report.md").write_text(
            "# gas_proxy_sensitivity report - stopped at reproduction gate\n\n"
            "The gas_proxy_sensitivity rerun did not reproduce the required anchors, so no sensitivity conclusion is accepted. "
            "See `internal_review/gas_proxy_sensitivity_reproduction_gate.csv` and `internal_review/gas_proxy_sensitivity_recomputed_all_method_metrics.csv`.\n",
            encoding="utf-8",
        )
        metrics.to_csv(PRIVATE / "gas_proxy_sensitivity_recomputed_all_method_metrics.csv", index=False, encoding="utf-8-sig")
        raise SystemExit("gas_proxy_sensitivity reproduction gate failed")
    sens, focus = sensitivity()
    write_reports(gate, metrics, sens, focus, backup, baselines)
    scan = anonymization_scan()
    if not scan["status"].eq("PASS").all():
        raise SystemExit("gas_proxy_sensitivity public anonymization scan failed")
    print("gas_proxy_sensitivity PASS", str(gas_proxy_sensitivity))
    print(gate[["check", "reproduced_pct", "status"]].to_string(index=False))
    print("North gas C4 sensitivity range", float(focus["benefit_pct"].min()), float(focus["benefit_pct"].max()))


if __name__ == "__main__":
    main()
