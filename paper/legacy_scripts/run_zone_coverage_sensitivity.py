# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from run_protocol_sensitivity import (
    ALPHA,
    CANON_DIR,
    OUT_DIR,
    PROJECT_ROOT,
    SEED,
    TARGET_LABEL,
    ZONE_EN,
    choose_models_from_canonical,
    interval_metrics,
    load_datasets,
    load_raw,
    make_pipe,
    quantile_empirical,
    quantile_finite,
    regressor_pool,
    split_iter,
    transformed,
)

SCRIPT = Path(__file__).resolve()
ANCHOR_FILE = CANON_DIR / "zone_reliability_summary.csv"
OUT_CSV = OUT_DIR / "zone_coverage_sensitivity_zone_coverage_sensitivity.csv"
OUT_REPORT = OUT_DIR / "report_zone_coverage_sensitivity.md"


def canonical_calibration_split(df: pd.DataFrame, tr_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, set[str]]:
    train = df.iloc[tr_all]
    wells = np.array(sorted(train["well"].dropna().unique()))
    rng = np.random.default_rng(SEED)
    if len(wells) >= 4:
        cal_wells = set(rng.choice(wells, size=max(1, int(len(wells) * 0.25)), replace=False))
        cal = np.array([i for i in tr_all if df.iloc[i]["well"] in cal_wells])
        tr = np.array([i for i in tr_all if df.iloc[i]["well"] not in cal_wells])
    else:
        cut = int(len(tr_all) * 0.75)
        tr = tr_all[:cut]
        cal = tr_all[cut:]
        cal_wells = set(df.iloc[cal]["well"].dropna().unique())
    return tr, cal, {str(w) for w in cal_wells}


def add_records(records: list[dict[str, object]], target: str, zones: np.ndarray, covered: np.ndarray, variant: str) -> None:
    for zone, ok in zip(zones, covered):
        records.append(
            {
                "target": target,
                "zone_en": ZONE_EN.get(str(zone), str(zone)),
                "variant": variant,
                "covered": bool(ok),
            }
        )


def median_scenario(scenarios: list[tuple[float, np.ndarray]]) -> tuple[float, np.ndarray]:
    ordered = sorted(scenarios, key=lambda item: item[0])
    return ordered[len(ordered) // 2]


def run_zone_coverage_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw()
    models = regressor_pool()
    records: list[dict[str, object]] = []

    for dataset, target_col, df in load_datasets(raw):
        target = TARGET_LABEL[target_col]
        model_names = choose_models_from_canonical(dataset)
        num = [c for c in ["GR", "AC", "DEN", "CNL", "RT", "pressure_coeff", "formation_top_md", "formation_bottom_md", "max_burial_top"] if c in df.columns]
        cat = [c for c in ["layer", "lithology", "diagenesis_category"] if c in df.columns]
        x = df[num + cat].copy()
        y = df[target_col].astype(float).to_numpy()

        for model_name in model_names:
            if model_name not in models:
                raise RuntimeError(f"Missing canonical model {model_name} for {dataset}")
            for scheme in ["LOWO", "LOZO"]:
                for fold, tr_all, te in split_iter(df, scheme):
                    tr, cal, cal_wells = canonical_calibration_split(df, tr_all)
                    if len(tr) < 10 or len(cal) < 5:
                        continue
                    pipe = make_pipe(model_name, models[model_name], num, cat)
                    pipe.fit(x.iloc[tr], y[tr])
                    cal_pred = np.asarray(pipe.predict(x.iloc[cal]), dtype=float)
                    pred = np.asarray(pipe.predict(x.iloc[te]), dtype=float)
                    y_te = y[te]
                    zones = df.iloc[te]["zone"].astype(str).to_numpy()
                    resid = np.abs(y[cal] - cal_pred)

                    q_emp = quantile_empirical(resid, ALPHA)
                    q_fin = quantile_finite(resid, ALPHA)
                    covered_base, _, _ = interval_metrics(y_te, pred, np.full(len(te), q_emp))
                    covered_a1, _, _ = interval_metrics(y_te, pred, np.full(len(te), q_fin))
                    add_records(records, target, zones, covered_base, "baseline")
                    add_records(records, target, zones, covered_a1, "A1_finite_sample_quantile")

                    scenarios_by_zone: dict[str, list[tuple[float, np.ndarray]]] = {}
                    cal_well_values = df.iloc[cal]["well"].astype(str).to_numpy()
                    for left_well in sorted(cal_wells):
                        keep = cal_well_values != left_well
                        if int(keep.sum()) < 1:
                            continue
                        q_leave = quantile_empirical(resid[keep], ALPHA)
                        covered_leave, _, _ = interval_metrics(y_te, pred, np.full(len(te), q_leave))
                        for zone in sorted(set(zones)):
                            mask = zones == zone
                            if not mask.any():
                                continue
                            cov = float(np.mean(covered_leave[mask]))
                            scenarios_by_zone.setdefault(zone, []).append((cov, covered_leave.copy()))

                    for zone, scenarios in scenarios_by_zone.items():
                        if not scenarios:
                            continue
                        mask = zones == zone
                        min_scenario = min(scenarios, key=lambda item: item[0])
                        med_scenario = median_scenario(scenarios)
                        max_scenario = max(scenarios, key=lambda item: item[0])
                        for variant, (_, covered) in [
                            ("A2min_leave_one_calibration_well", min_scenario),
                            ("A2median_leave_one_calibration_well", med_scenario),
                            ("A2max_leave_one_calibration_well", max_scenario),
                        ]:
                            add_records(records, target, zones[mask], covered[mask], variant)

    rec = pd.DataFrame(records)
    coverage = (
        rec.groupby(["target", "zone_en", "variant"], dropna=False)
        .agg(coverage=("covered", "mean"), n_records=("covered", "size"))
        .reset_index()
    )
    return coverage, rec


def sign(value: float) -> int:
    delta = value - 0.90
    if delta > 1e-12:
        return 1
    if delta < -1e-12:
        return -1
    return 0


def build_public_table(coverage: pd.DataFrame) -> pd.DataFrame:
    anchors = pd.read_csv(ANCHOR_FILE, encoding="utf-8-sig")
    anchors["target_label"] = anchors["target"].map({"toc_pct": "TOC", "total_gas_m3_t": "total_gas"})
    anchors["zone_en"] = anchors["zone"].map(ZONE_EN).fillna(anchors["zone"])
    anchor_map = anchors.set_index(["target_label", "zone_en"])["coverage"].to_dict()

    pivot = coverage.pivot(index=["target", "zone_en"], columns="variant", values="coverage").reset_index()
    rename = {
        "baseline": "baseline",
        "A1_finite_sample_quantile": "A1",
        "A2min_leave_one_calibration_well": "A2min",
        "A2median_leave_one_calibration_well": "A2median",
        "A2max_leave_one_calibration_well": "A2max",
    }
    pivot = pivot.rename(columns=rename)
    for col in ["baseline", "A1", "A2min", "A2median", "A2max"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["anchor"] = [
        anchor_map[(str(target), str(zone_en))]
        for target, zone_en in zip(pivot["target"], pivot["zone_en"])
    ]
    pivot["anchor_delta"] = pivot["baseline"] - pivot["anchor"]

    target_lowest: dict[str, bool] = {}
    variants = ["baseline", "A1", "A2min", "A2median", "A2max"]
    for target, group in pivot.groupby("target"):
        ok_all = True
        north = group[group["zone_en"].eq("North")]
        if north.empty:
            ok_all = False
        else:
            for variant in variants:
                north_cov = float(north[variant].iloc[0])
                min_cov = float(group[variant].min())
                if north_cov > min_cov + 1e-12:
                    ok_all = False
        target_lowest[str(target)] = ok_all

    flip_values = []
    for _, row in pivot.iterrows():
        base_sign = sign(float(row["anchor"]))
        flips = []
        for variant in variants:
            flips.append(sign(float(row[variant])) != base_sign)
        flip_values.append(any(flips))
    pivot["north_lowest_all_variants"] = pivot["target"].map(target_lowest)
    pivot["nominal_gap_sign_flip"] = flip_values
    cols = [
        "target",
        "zone_en",
        "anchor",
        "baseline",
        "A1",
        "A2min",
        "A2median",
        "A2max",
        "north_lowest_all_variants",
        "nominal_gap_sign_flip",
        "anchor_delta",
    ]
    return pivot[cols].sort_values(["target", "zone_en"]).reset_index(drop=True)


def check_anchors(public: pd.DataFrame) -> list[str]:
    checks = []
    for _, row in public.iterrows():
        ok = abs(float(row["baseline"]) - float(row["anchor"])) <= 0.005
        checks.append(
            f"{'PASS' if ok else 'FAIL'} {row['target']} {row['zone_en']} baseline={float(row['baseline']):.6f} anchor={float(row['anchor']):.6f}"
        )
    return checks


def write_report(public: pd.DataFrame, checks: list[str]) -> None:
    north_answers = []
    flip_answers = []
    for target, group in public.groupby("target"):
        north_answers.append(
            f"- {target}: North lowest under all variants = {bool(group['north_lowest_all_variants'].iloc[0])}."
        )
        flipped = group[group["nominal_gap_sign_flip"].astype(bool)]
        if flipped.empty:
            flip_answers.append(f"- {target}: no zone flips the sign of coverage - 0.90.")
        else:
            items = ", ".join(flipped["zone_en"].tolist())
            flip_answers.append(f"- {target}: sign flips occur in {items}.")

    lines = [
        "# zone_coverage_sensitivity zone coverage sensitivity",
        "",
        "## Canonical Scope Replication",
        "",
        "The corrected zone_coverage_sensitivity scope follows the canonical `zone_reliability_summary.csv` generation logic, not the previous zone-conditional slice. In `EVO\u8f93\u51fa/full_condition_analysis_derived_dataset/run_full_condition_evaluation.py`, `conformal_rejection()` builds conformal predictions for the `choose_models()` top-three model pool and the LOWO/LOZO outer folds; the interval rows are appended at lines 203-225. The map-level zone summary is then produced by `reliability(cpreds)`: line 236 filters `cpreds[cpreds.method=='global']`, and lines 239-241 aggregate coverage by `dataset,target,zone`. zone_coverage_sensitivity therefore uses the same top-three model pool, both LOWO and LOZO, global empirical q90 intervals, and sample-level zone aggregation.",
        "",
        "Only the quantile convention is changed after the baseline anchor is reproduced: A1 uses the finite-sample conformal order statistic, and A2 leaves out one calibration well at a time and recomputes the empirical global q90 before aggregating zone coverage.",
        "",
        "## Anchor Checks",
        "",
    ]
    lines.extend(f"- {item}" for item in checks)
    lines += [
        "",
        "## Fact Answers",
        "",
        "### Is North still the lowest-coverage zone for each target under all variants?",
        "",
    ]
    lines.extend(north_answers)
    lines += [
        "",
        "### Does any zone flip the sign of coverage minus the 0.90 nominal level?",
        "",
    ]
    lines.extend(flip_answers)
    lines += [
        "",
        "## Values",
        "",
        "| target | zone | anchor | baseline | A1 | A2min | A2median | A2max | North lowest all variants | nominal gap sign flip |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, row in public.iterrows():
        lines.append(
            f"| {row['target']} | {row['zone_en']} | {float(row['anchor']):.6f} | {float(row['baseline']):.6f} | "
            f"{float(row['A1']):.6f} | {float(row['A2min']):.6f} | {float(row['A2median']):.6f} | {float(row['A2max']):.6f} | "
            f"{bool(row['north_lowest_all_variants'])} | {bool(row['nominal_gap_sign_flip'])} |"
        )
    lines += [
        "",
        "## Reproduction",
        "",
        f"`py -3.11 \"{SCRIPT}\"`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    coverage, _ = run_zone_coverage_sensitivity()
    public = build_public_table(coverage)
    public.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    checks = check_anchors(public)
    write_report(public, checks)
    if any(item.startswith("FAIL") for item in checks):
        print("\n".join(checks))
        raise SystemExit(2)
    print("PASS")
    print("\n".join(checks))
    print(public.to_string(index=False))


if __name__ == "__main__":
    main()
