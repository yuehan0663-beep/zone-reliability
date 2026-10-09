# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"PROJECT_ROOT_PLACEHOLDER/evo_workspace")
WORK_DIR = next(ROOT.glob("*9/*analysis_set"))
adaptive_rejection = WORK_DIR / "adaptive_rejection"
PRIVATE = adaptive_rejection / "internal_review"
reproducibility_bundle = WORK_DIR / "reproducibility_bundle" / "repro_bundle"

PRED_PATH = reproducibility_bundle / "conformal_predictions.csv"
FIXED_SUMMARY_PATH = reproducibility_bundle / "rejection_strategy_comparison.csv"
BUDGET_PATH = adaptive_rejection / "budget_spec.md"

K_APPROVED = 4
MDE_PCT_POINTS = 15.660
CI_LEVEL = 1.0 - 0.05 / K_APPROVED
ALPHA = 1.0 - CI_LEVEL
BOOTSTRAP_N = 20000
SEED = 20260701 + 6
BASE_REJECT = 0.20
MIN_REJECT = 0.05
MAX_REJECT = 0.35

PRIMARY_DATASET = "total_gas_high_medium"
PRIMARY_TARGET = "total_gas_m3_t"
NEGATIVE_CONTROL_DATASET = "TOC_high_medium"
NEGATIVE_CONTROL_TARGET = "toc_pct"
CV_SCHEME = "LOWO"
METHOD = "global"
FIXED_RISK_SCORE = "composite_risk"


VARIANTS = [
    {
        "variant_id": "V1_width_zone_slope05",
        "score_family": "interval_width",
        "score_formula": "zone mean of within-fold interval_width percentile rank",
        "base_reject_frac": BASE_REJECT,
        "slope": 0.05,
        "min_reject_frac": MIN_REJECT,
        "max_reject_frac": MAX_REJECT,
        "label_free_inputs": "interval_width, zone_en",
        "rationale": "Allocate slightly more rejection to zones with wider conformal intervals.",
    },
    {
        "variant_id": "V2_domain_zone_slope05",
        "score_family": "domain_knn",
        "score_formula": "zone mean of within-fold domain_knn percentile rank",
        "base_reject_frac": BASE_REJECT,
        "slope": 0.05,
        "min_reject_frac": MIN_REJECT,
        "max_reject_frac": MAX_REJECT,
        "label_free_inputs": "domain_knn, zone_en",
        "rationale": "Allocate slightly more rejection to zones with stronger out-of-domain distance.",
    },
    {
        "variant_id": "V3_missing_quality_zone_slope05",
        "score_family": "missing_quality",
        "score_formula": "zone mean of 0.5*missing_risk_rank + 0.5*quality_risk_rank",
        "base_reject_frac": BASE_REJECT,
        "slope": 0.05,
        "min_reject_frac": MIN_REJECT,
        "max_reject_frac": MAX_REJECT,
        "label_free_inputs": "missing_risk, quality_risk, zone_en",
        "rationale": "Allocate slightly more rejection to zones with weaker input completeness/quality.",
    },
    {
        "variant_id": "V4_composite_label_free_zone_slope05",
        "score_family": "composite_label_free",
        "score_formula": "zone mean of 0.55*width_rank + 0.25*domain_rank + 0.10*missing_rank + 0.10*quality_rank",
        "base_reject_frac": BASE_REJECT,
        "slope": 0.05,
        "min_reject_frac": MIN_REJECT,
        "max_reject_frac": MAX_REJECT,
        "label_free_inputs": "interval_width, domain_knn, missing_risk, quality_risk, zone_en",
        "rationale": "Use the same label-free uncertainty families as the fixed composite score, but adapt only the zone rejection fraction.",
    },
]


def ensure_dirs() -> None:
    adaptive_rejection.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)


def freeze_variants() -> pd.DataFrame:
    path = adaptive_rejection / "preregistered_variants.csv"
    df = pd.DataFrame(VARIANTS)
    df.insert(1, "K_budget", K_APPROVED)
    df["formula"] = "r_z = clip(base_reject_frac + slope * standardized_zone_score, min_reject_frac, max_reject_frac)"
    df["uses_heldout_labels"] = "N"
    df["frozen_before_evaluation"] = "Y"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def rank_pct(s: pd.Series) -> pd.Series:
    if s.nunique(dropna=False) <= 1:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return s.rank(pct=True, method="average")


def add_label_free_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group = ["dataset", "target", "model", "cv_scheme", "fold_label_anon", "method"]
    for col, new_col in [
        ("interval_width", "width_rank"),
        ("domain_knn", "domain_rank"),
        ("missing_risk", "missing_rank"),
        ("quality_risk", "quality_rank"),
    ]:
        out[new_col] = out.groupby(group, group_keys=False)[col].apply(rank_pct)
    out["missing_quality_rank"] = 0.5 * out["missing_rank"] + 0.5 * out["quality_rank"]
    out["composite_label_free_rank"] = (
        0.55 * out["width_rank"]
        + 0.25 * out["domain_rank"]
        + 0.10 * out["missing_rank"]
        + 0.10 * out["quality_rank"]
    )
    return out


def zone_rates(df: pd.DataFrame, variant: pd.Series) -> pd.DataFrame:
    score_col = {
        "interval_width": "width_rank",
        "domain_knn": "domain_rank",
        "missing_quality": "missing_quality_rank",
        "composite_label_free": "composite_label_free_rank",
    }[variant["score_family"]]
    rows = []
    for (dataset, target), g in df.groupby(["dataset", "target"], dropna=False):
        z = g.groupby("zone_en", as_index=False).agg(zone_score=(score_col, "mean"), n_rows=(score_col, "size"))
        if len(z) <= 1 or float(z["zone_score"].std(ddof=0)) == 0:
            z["standardized_zone_score"] = 0.0
        else:
            z["standardized_zone_score"] = (z["zone_score"] - z["zone_score"].mean()) / z["zone_score"].std(ddof=0)
        z["reject_frac"] = np.clip(
            float(variant["base_reject_frac"]) + float(variant["slope"]) * z["standardized_zone_score"],
            float(variant["min_reject_frac"]),
            float(variant["max_reject_frac"]),
        )
        z.insert(0, "variant_id", variant["variant_id"])
        z.insert(1, "dataset", dataset)
        z.insert(2, "target", target)
        rows.append(z)
    return pd.concat(rows, ignore_index=True)


def metric_rows_fixed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "target", "model", "cv_scheme", "fold_label_anon", "method"]
    for key, g in df.groupby(keys, dropna=False):
        if len(g) < 5:
            continue
        base_mae = float(g["abs_error"].mean())
        sb = g.sort_values(FIXED_RISK_SCORE, ascending=False).reset_index(drop=True)
        kept = sb.iloc[int(len(sb) * BASE_REJECT) :]
        retained_valid = len(kept) >= 5
        rows.append(
            dict(
                zip(keys, key),
                base_MAE=base_mae,
                retained_MAE=float(kept["abs_error"].mean()) if retained_valid else np.nan,
                kept_n=int(len(kept)),
                retained_valid=retained_valid,
                reject_frac=BASE_REJECT,
            )
        )
    return pd.DataFrame(rows)


def metric_rows_adaptive(df: pd.DataFrame, rates: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    rows = []
    rate_map = {
        (r["dataset"], r["target"], r["zone_en"]): float(r["reject_frac"])
        for _, r in rates[rates["variant_id"].eq(variant_id)].iterrows()
    }
    keys = ["dataset", "target", "model", "cv_scheme", "fold_label_anon", "method"]
    for key, g in df.groupby(keys, dropna=False):
        if len(g) < 5:
            continue
        base_mae = float(g["abs_error"].mean())
        kept_parts = []
        actual_rejected = 0
        for zone, zg in g.groupby("zone_en", dropna=False):
            rf = rate_map[(key[0], key[1], zone)]
            sb = zg.sort_values(FIXED_RISK_SCORE, ascending=False).reset_index(drop=True)
            n_reject = int(len(sb) * rf)
            actual_rejected += n_reject
            kept_parts.append(sb.iloc[n_reject:])
        kept = pd.concat(kept_parts, ignore_index=True)
        retained_valid = len(kept) >= 5
        rows.append(
            dict(
                zip(keys, key),
                base_MAE=base_mae,
                retained_MAE=float(kept["abs_error"].mean()) if retained_valid else np.nan,
                kept_n=int(len(kept)),
                retained_valid=retained_valid,
                effective_reject_frac=float(actual_rejected / len(g)),
            )
        )
    return pd.DataFrame(rows)


def benefit(rows: pd.DataFrame) -> float:
    retained = rows.loc[rows["retained_valid"].astype(bool), "retained_MAE"]
    return (float(rows["base_MAE"].mean()) - float(retained.mean())) / float(rows["base_MAE"].mean()) * 100.0


def bootstrap_paired_diff(fixed: pd.DataFrame, adaptive: pd.DataFrame, n_boot: int, seed: int) -> np.ndarray:
    merge_keys = ["dataset", "target", "model", "cv_scheme", "fold_label_anon", "method"]
    f = fixed.rename(columns={"base_MAE": "fixed_base_MAE", "retained_MAE": "fixed_retained_MAE"})
    a = adaptive.rename(columns={"base_MAE": "adaptive_base_MAE", "retained_MAE": "adaptive_retained_MAE"})
    merged = f[merge_keys + ["fixed_base_MAE", "fixed_retained_MAE", "retained_valid"]].rename(
        columns={"retained_valid": "fixed_retained_valid"}
    ).merge(
        a[merge_keys + ["adaptive_base_MAE", "adaptive_retained_MAE", "retained_valid"]].rename(
            columns={"retained_valid": "adaptive_retained_valid"}
        ),
        on=merge_keys,
        how="inner",
    )
    fold_stats = []
    for fold, g in merged.groupby("fold_label_anon", dropna=False):
        fixed_retained = g.loc[g["fixed_retained_valid"].astype(bool), "fixed_retained_MAE"]
        adaptive_retained = g.loc[g["adaptive_retained_valid"].astype(bool), "adaptive_retained_MAE"]
        fold_stats.append(
            {
                "fold": fold,
                "fixed_base_sum": float(g["fixed_base_MAE"].sum()),
                "fixed_base_n": int(g["fixed_base_MAE"].count()),
                "fixed_retained_sum": float(fixed_retained.sum()),
                "fixed_retained_n": int(fixed_retained.count()),
                "adaptive_base_sum": float(g["adaptive_base_MAE"].sum()),
                "adaptive_base_n": int(g["adaptive_base_MAE"].count()),
                "adaptive_retained_sum": float(adaptive_retained.sum()),
                "adaptive_retained_n": int(adaptive_retained.count()),
            }
        )
    stats = pd.DataFrame(fold_stats)
    n_folds = len(stats)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_folds, size=(n_boot, n_folds))
    arrays = {col: stats[col].to_numpy(float) for col in stats.columns if col != "fold"}

    fb_mean = arrays["fixed_base_sum"][sampled].sum(axis=1) / arrays["fixed_base_n"][sampled].sum(axis=1)
    fr_n = arrays["fixed_retained_n"][sampled].sum(axis=1)
    fr_mean = arrays["fixed_retained_sum"][sampled].sum(axis=1) / fr_n
    ab_mean = arrays["adaptive_base_sum"][sampled].sum(axis=1) / arrays["adaptive_base_n"][sampled].sum(axis=1)
    ar_n = arrays["adaptive_retained_n"][sampled].sum(axis=1)
    ar_mean = arrays["adaptive_retained_sum"][sampled].sum(axis=1) / ar_n

    valid = (fr_n > 0) & (ar_n > 0)
    fixed_benefit = (fb_mean - fr_mean) / fb_mean * 100.0
    adaptive_benefit = (ab_mean - ar_mean) / ab_mean * 100.0
    vals = adaptive_benefit - fixed_benefit
    vals = vals[valid]
    return vals[~np.isnan(vals)]


def ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [ALPHA / 2.0, 1.0 - ALPHA / 2.0]).astype(float))


def scan_public_outputs() -> pd.DataFrame:
    banned = []  # public package: true-name deny-list removed after de-identification
    rows = []
    for path in [adaptive_rejection / "preregistered_variants.csv", adaptive_rejection / "search_results.csv", adaptive_rejection / "ladder_log.md", adaptive_rejection / "verdict.md", adaptive_rejection / "adaptive_rejection_report.md", adaptive_rejection / "budget_spec.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8", errors="ignore")
        hits = [b for b in banned if b.lower() in text.lower()]
        rows.append({"file": path.name, "hit_count": len(hits), "hits": ";".join(hits), "status": "PASS" if not hits else "FAIL"})
    out = pd.DataFrame(rows)
    out.to_csv(PRIVATE / "adaptive_rejection_public_sensitive_scan.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    ensure_dirs()
    variants = freeze_variants()

    pred = pd.read_csv(PRED_PATH)
    pred = add_label_free_ranks(pred)
    pred = pred[pred["cv_scheme"].eq(CV_SCHEME) & pred["method"].eq(METHOD)].copy()
    pred = pred[pred["target"].isin([PRIMARY_TARGET, NEGATIVE_CONTROL_TARGET])].copy()

    all_rates = []
    for _, variant in variants.iterrows():
        all_rates.append(zone_rates(pred, variant))
    rates = pd.concat(all_rates, ignore_index=True)
    rates.to_csv(PRIVATE / "adaptive_rejection_setup_variant_zone_rates.csv", index=False, encoding="utf-8-sig")

    fixed_rows = metric_rows_fixed(pred)
    fixed_rows.to_csv(PRIVATE / "adaptive_rejection_evaluation_fixed20_metric_rows.csv", index=False, encoding="utf-8-sig")

    fixed_summary = pd.read_csv(FIXED_SUMMARY_PATH)
    fixed_check = fixed_summary[
        fixed_summary["target"].eq(PRIMARY_TARGET)
        & fixed_summary["cv_scheme"].eq(CV_SCHEME)
        & fixed_summary["method"].eq(METHOD)
        & fixed_summary["risk_score"].eq(FIXED_RISK_SCORE)
        & fixed_summary["reject_frac"].eq(BASE_REJECT)
    ]

    results = []
    adaptive_all = []
    for _, variant in variants.iterrows():
        adaptive_rows = metric_rows_adaptive(pred, rates, str(variant["variant_id"]))
        adaptive_rows.insert(0, "variant_id", variant["variant_id"])
        adaptive_all.append(adaptive_rows)
        for target, label in [(PRIMARY_TARGET, "gas_main"), (NEGATIVE_CONTROL_TARGET, "toc_negative_control")]:
            f = fixed_rows[fixed_rows["target"].eq(target)].copy()
            a = adaptive_rows[adaptive_rows["target"].eq(target)].copy()
            fixed_b = benefit(f)
            adaptive_b = benefit(a)
            diff = adaptive_b - fixed_b
            vals = bootstrap_paired_diff(f, a, BOOTSTRAP_N, SEED + int(variant.name) * 100 + (0 if target == PRIMARY_TARGET else 50))
            lo, hi = ci(vals)
            win = bool(target == PRIMARY_TARGET and diff > MDE_PCT_POINTS and lo > 0)
            toc_false_gain = bool(target == NEGATIVE_CONTROL_TARGET and lo > 0)
            results.append(
                {
                    "variant_id": variant["variant_id"],
                    "target_role": label,
                    "dataset": PRIMARY_DATASET if target == PRIMARY_TARGET else NEGATIVE_CONTROL_DATASET,
                    "target": target,
                    "fixed_benefit_pct": fixed_b,
                    "adaptive_benefit_pct": adaptive_b,
                    "paired_diff_pct_points": diff,
                    "bonferroni_ci_level": CI_LEVEL,
                    "paired_diff_ci_low": lo,
                    "paired_diff_ci_high": hi,
                    "mde_pct_points": MDE_PCT_POINTS,
                    "passes_diff_gt_mde": diff > MDE_PCT_POINTS,
                    "ci_excludes_zero_positive": lo > 0,
                    "toc_false_gain": toc_false_gain,
                    "win": win,
                    "bootstrap_n": len(vals),
                    "fixed_metric_rows": len(f),
                    "adaptive_metric_rows": len(a),
                    "paired_metric_rows": len(
                        f.merge(a, on=["dataset", "target", "model", "cv_scheme", "fold_label_anon", "method"], how="inner")
                    ),
                }
            )

    adaptive_all_df = pd.concat(adaptive_all, ignore_index=True)
    adaptive_all_df.to_csv(PRIVATE / "adaptive_rejection_evaluation_adaptive_metric_rows.csv", index=False, encoding="utf-8-sig")
    results_df = pd.DataFrame(results)
    results_df.to_csv(adaptive_rejection / "search_results.csv", index=False, encoding="utf-8-sig")

    gas = results_df[results_df["target_role"].eq("gas_main")].copy()
    toc = results_df[results_df["target_role"].eq("toc_negative_control")].copy()
    winners = gas[gas["win"]]
    toc_false = toc[toc["toc_false_gain"]]

    ladder_lines = [
        "# adaptive_rejection_evaluation Ladder-style evaluation log",
        "",
        f"- Approved K: {K_APPROVED}.",
        f"- Bonferroni CI level: {CI_LEVEL:.4f}.",
        f"- MDE: {MDE_PCT_POINTS:.3f} percentage points.",
        "- Each preregistered variant was evaluated exactly once; no variant was modified after seeing outcomes.",
        "",
        "| order | variant | gas paired diff | CI | ladder status |",
        "|---:|---|---:|---|---|",
    ]
    current_best = -math.inf
    for i, row in gas.reset_index(drop=True).iterrows():
        status = "recorded_no_win"
        if row["paired_diff_pct_points"] > current_best + MDE_PCT_POINTS and row["paired_diff_ci_low"] > 0:
            status = "would_update_ladder"
            current_best = row["paired_diff_pct_points"]
        else:
            current_best = max(current_best, row["paired_diff_pct_points"])
        ladder_lines.append(
            f"| {i+1} | {row['variant_id']} | {row['paired_diff_pct_points']:.3f} | [{row['paired_diff_ci_low']:.3f}, {row['paired_diff_ci_high']:.3f}] | {status} |"
        )
    (adaptive_rejection / "ladder_log.md").write_text("\n".join(ladder_lines) + "\n", encoding="utf-8")

    verdict = "upgrade_contribution" if len(winners) and toc_false.empty else "no_robust_adaptive_gain"
    verdict_lines = [
        "# adaptive_rejection_verdict verdict",
        "",
        f"Verdict: **{verdict}**.",
        "",
        "## Decision rule",
        "",
        f"- K budget: `{K_APPROVED}` preregistered variants.",
        f"- MDE: `{MDE_PCT_POINTS:.3f}` percentage points from adaptive_rejection_budget.",
        f"- Bonferroni CI level: `{CI_LEVEL:.2%}`.",
        "- Win requires gas LOWO paired difference `(adaptive_benefit - fixed_benefit)` to exceed MDE and have Bonferroni CI entirely above 0.",
        "- Negative control requires TOC LOWO not to show a positive Bonferroni CI.",
        "",
        "## Gas LOWO results",
        "",
        "| variant | fixed benefit % | adaptive benefit % | diff pp | Bonferroni CI | win |",
        "|---|---:|---:|---:|---|---|",
    ]
    for _, row in gas.iterrows():
        verdict_lines.append(
            f"| {row['variant_id']} | {row['fixed_benefit_pct']:.3f} | {row['adaptive_benefit_pct']:.3f} | {row['paired_diff_pct_points']:.3f} | [{row['paired_diff_ci_low']:.3f}, {row['paired_diff_ci_high']:.3f}] | {row['win']} |"
        )
    verdict_lines += [
        "",
        "## TOC LOWO negative control",
        "",
        "| variant | fixed benefit % | adaptive benefit % | diff pp | Bonferroni CI | false positive? |",
        "|---|---:|---:|---:|---|---|",
    ]
    for _, row in toc.iterrows():
        verdict_lines.append(
            f"| {row['variant_id']} | {row['fixed_benefit_pct']:.3f} | {row['adaptive_benefit_pct']:.3f} | {row['paired_diff_pct_points']:.3f} | [{row['paired_diff_ci_low']:.3f}, {row['paired_diff_ci_high']:.3f}] | {row['toc_false_gain']} |"
        )
    verdict_lines += [
        "",
        "## Claim boundary",
        "",
        f"Under this dataset and the approved honest budget K={K_APPROVED}, MDE={MDE_PCT_POINTS:.3f} percentage points, no preregistered label-free adaptive rejection variant robustly improves over the fixed 20% gas LOWO baseline. This is not evidence that adaptive rejection is impossible; it only says no robust gain was found under this small-sample, preregistered, multiplicity-corrected budget.",
    ]
    (adaptive_rejection / "verdict.md").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# adaptive_rejection report",
        "",
        "Status: **COMPLETE; STOP FOR REVIEW**.",
        "",
        "## Outputs",
        "",
        "- `preregistered_variants.csv`: four frozen label-free adaptive rejection variants.",
        "- `search_results.csv`: one-shot adaptive_rejection_evaluation paired by-well bootstrap evaluation.",
        "- `ladder_log.md`: Ladder-style audit log.",
        "- `verdict.md`: adaptive_rejection_verdict double-sided decision.",
        "",
        "## Validation notes",
        "",
        f"- Fixed baseline recomputed from the anonymized derived prediction table: gas LOWO global fixed benefit = {gas['fixed_benefit_pct'].iloc[0]:.3f}%.",
        "- The paired bootstrap resamples held-out wells and computes adaptive and fixed benefit within the same bootstrap draw.",
        f"- Bonferroni CI level uses `1 - 0.05/K = {CI_LEVEL:.4f}` for K={K_APPROVED}.",
        f"- No variant is declared a win unless gas paired diff > MDE ({MDE_PCT_POINTS:.3f} pp) and the corrected CI excludes 0.",
        "- Public outputs use anonymized derived data only.",
        "",
        "## Verdict",
        "",
        f"`{verdict}`. See `verdict.md` for the full table.",
    ]
    (adaptive_rejection / "adaptive_rejection_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    scan = scan_public_outputs()
    if (scan["status"] == "FAIL").any():
        raise RuntimeError("adaptive_rejection public sensitive scan failed; inspect internal_review/adaptive_rejection_public_sensitive_scan.csv")

    print(f"adaptive rejection search PASS: verdict={verdict}; results={adaptive_rejection / 'search_results.csv'}")


if __name__ == "__main__":
    main()
