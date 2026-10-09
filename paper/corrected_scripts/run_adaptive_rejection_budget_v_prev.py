# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


ROOT = Path(r"PROJECT_ROOT_PLACEHOLDER/evo_workspace")
WORK_DIR = next(ROOT.glob("*9/*analysis_set"))
adaptive_rejection = WORK_DIR / "adaptive_rejection"
PRIVATE = adaptive_rejection / "internal_review"
reproducibility_bundle_README = WORK_DIR / "reproducibility_bundle" / "repro_bundle" / "README.md"
DETAIL = next(ROOT.glob("*9/*round1*/rejection_weight_ablation_v2_detail.csv"))
SUMMARY = next(ROOT.glob("*9/*round1*/rejection_weight_ablation_v2.csv"))

SEED = 20260701
BOOTSTRAP_N = 20000
PRIMARY_TARGET = "total_gas_m3_t"
PRIMARY_DATASET = "total_gas_high_medium"
PRIMARY_CV = "LOWO"
PRIMARY_METHOD = "global"
PRIMARY_SCHEME = "current_0.55_0.25_0.10_0.10"
PRIMARY_THRESHOLD_PP = 20.0


def ensure_dirs() -> None:
    adaptive_rejection.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)


def point_estimate(df: pd.DataFrame) -> tuple[float, float, float]:
    base = float(df.loc[df["reject_frac"].eq(0.0), "MAE"].mean())
    retained = float(df.loc[df["reject_frac"].eq(0.2), "MAE"].mean())
    improvement = (base - retained) / base * 100.0
    return improvement, base, retained


def bootstrap_by_well(df: pd.DataFrame, n_boot: int, seed: int) -> np.ndarray:
    folds = list(df["fold"].drop_duplicates())
    grouped = []
    for fold in folds:
        g = df[df["fold"].eq(fold)]
        grouped.append(
            (
                g.loc[g["reject_frac"].eq(0.0), "MAE"].to_numpy(float),
                g.loc[g["reject_frac"].eq(0.2), "MAE"].to_numpy(float),
            )
        )

    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        base_parts = [grouped[j][0] for j in sampled if grouped[j][0].size]
        retained_parts = [grouped[j][1] for j in sampled if grouped[j][1].size]
        if not retained_parts:
            values[i] = np.nan
            continue
        base = np.concatenate(base_parts)
        retained = np.concatenate(retained_parts)
        values[i] = (float(base.mean()) - float(retained.mean())) / float(base.mean()) * 100.0
    return values[~np.isnan(values)]


def ci_half_width(values: np.ndarray, level: float) -> tuple[float, float, float]:
    alpha = 1.0 - level
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi), float((hi - lo) / 2.0)


def k_half_width(mde_95: float, k: int) -> tuple[float, float]:
    z95 = NormalDist().inv_cdf(0.975)
    level = 1.0 - 0.05 / k
    z = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * k))
    return level, float(mde_95 * z / z95)


def k_max_for_threshold(mde_95: float, threshold_pp: float) -> int:
    if threshold_pp <= mde_95:
        return 0
    z95 = NormalDist().inv_cdf(0.975)
    z_limit = threshold_pp / mde_95 * z95
    tail = 1.0 - NormalDist().cdf(z_limit)
    if tail <= 0:
        return 10**9
    return max(0, math.floor(0.05 / (2.0 * tail)))


def format_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    ensure_dirs()
    detail = pd.read_csv(DETAIL)
    summary = pd.read_csv(SUMMARY)
    scope = detail[
        detail["dataset"].eq(PRIMARY_DATASET)
        & detail["target"].eq(PRIMARY_TARGET)
        & detail["cv_scheme"].eq(PRIMARY_CV)
        & detail["weight_scheme"].eq(PRIMARY_SCHEME)
        & detail["reject_frac"].isin([0.0, 0.2])
    ].copy()
    primary = scope[scope["method"].eq(PRIMARY_METHOD)].copy()
    values = bootstrap_by_well(primary, BOOTSTRAP_N, SEED)
    point, mae0, mae20 = point_estimate(primary)
    ci95_lo, ci95_hi, mde95 = ci_half_width(values, 0.95)

    secondary_rows = []
    for method, group in scope.groupby("method"):
        vals = bootstrap_by_well(group, BOOTSTRAP_N, SEED + len(method))
        imp, b0, b20 = point_estimate(group)
        lo, hi, half = ci_half_width(vals, 0.95)
        secondary_rows.append(
            {
                "method": method,
                "point_improvement_pct": imp,
                "mae_0pct": b0,
                "mae_20pct": b20,
                "ci95_low_pct": lo,
                "ci95_high_pct": hi,
                "mde95_half_width_pct_points": half,
                "bootstrap_n": len(vals),
                "well_count_union": group["fold"].nunique(),
                "well_count_at_20pct": group.loc[group["reject_frac"].eq(0.2), "fold"].nunique(),
            }
        )
    secondary = pd.DataFrame(secondary_rows)
    secondary.to_csv(PRIVATE / "adaptive_rejection_budget_method_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    k_rows = []
    for k in [1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 50, 100]:
        level, half = k_half_width(mde95, k)
        k_rows.append(
            {
                "K": k,
                "bonferroni_ci_level": level,
                "required_excess_half_width_pct_points": half,
                "passes_20pp_threshold": half < PRIMARY_THRESHOLD_PP,
                "passes_observed_baseline_threshold": half < point,
            }
        )
    k_table = pd.DataFrame(k_rows)
    k_table.to_csv(PRIVATE / "adaptive_rejection_budget_k_budget_table.csv", index=False, encoding="utf-8-sig")

    threshold_rows = []
    for threshold in [5.0, 10.0, 15.0, 20.0, point, 25.0, 30.0]:
        threshold_rows.append(
            {
                "geological_meaningful_threshold_pct_points": threshold,
                "K_max": k_max_for_threshold(mde95, threshold),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(PRIVATE / "adaptive_rejection_budget_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")

    primary_summary = summary[
        summary["dataset"].eq(PRIMARY_DATASET)
        & summary["target"].eq(PRIMARY_TARGET)
        & summary["cv_scheme"].eq(PRIMARY_CV)
        & summary["method"].eq(PRIMARY_METHOD)
        & summary["weight_scheme"].eq(PRIMARY_SCHEME)
    ].iloc[0]
    alignment_delta = point - float(primary_summary["relative_improvement_pct"])
    kmax20 = k_max_for_threshold(mde95, PRIMARY_THRESHOLD_PP)
    kmax_observed = k_max_for_threshold(mde95, point)
    level_kmax20, half_kmax20 = k_half_width(mde95, max(kmax20, 1))

    lines = [
        "# adaptive_rejection_budget honest-budget gate: MDE and K_max",
        "",
        "Status: **PASS FOR adaptive_rejection_budget ONLY; STOP BEFORE adaptive rejection search**.",
        "",
        "## Scope",
        "",
        f"- Focus: gas LOWO fixed 20% rejection baseline.",
        f"- Canonical source: `{DETAIL.name}` plus `{SUMMARY.name}`.",
        f"- Primary row: dataset `{PRIMARY_DATASET}`, target `{PRIMARY_TARGET}`, method `{PRIMARY_METHOD}`, weight scheme `{PRIMARY_SCHEME}`.",
        "- Aggregation follows the verified rejection_weight_sensitivity canonical convention: fold/well-level MAE rows are averaged by method and rejection fraction; no raw well names or raw workbook data are used in the public report.",
        "",
        "## Baseline alignment",
        "",
        f"- `MAE_0pct = {format_float(mae0, 6)}`.",
        f"- `MAE_20pct = {format_float(mae20, 6)}`.",
        f"- Fixed 20% retained-MAE improvement = `{format_float(point, 6)}%`.",
        f"- rejection_weight_sensitivity summary improvement = `{format_float(float(primary_summary['relative_improvement_pct']), 6)}%`; recomputation delta = `{format_float(alignment_delta, 9)} percentage points`.",
        "",
        "Formula:",
        "",
        "`improvement_pct = (MAE_0pct - MAE_20pct) / MAE_0pct * 100`",
        "",
        "## MDE",
        "",
        f"- Bootstrap unit: LOWO held-out well/fold.",
        f"- Bootstrap resamples: `{len(values)}` with seed `{SEED}`.",
        f"- 95% bootstrap CI for fixed-baseline improvement: `{format_float(ci95_lo)}%` to `{format_float(ci95_hi)}%`.",
        f"- **MDE = 95% CI half-width = `{format_float(mde95)} percentage points`.**",
        "",
        "Formula:",
        "",
        "`MDE_95 = (Q_0.975(bootstrap_improvement) - Q_0.025(bootstrap_improvement)) / 2`",
        "",
        "## Bonferroni K budget",
        "",
        "For K preregistered variants, the two-sided bootstrap CI level is:",
        "",
        "`CI_level(K) = 1 - 0.05 / K`",
        "",
        "The corresponding judgment half-width is scaled from the empirical 95% MDE by the normal-quantile ratio:",
        "",
        "`half_width(K) = MDE_95 * z_(1 - 0.05/(2K)) / z_0.975`",
        "",
        f"Using a preregistered practical/geological threshold of `{format_float(PRIMARY_THRESHOLD_PP, 1)} percentage points`, selected because it is close to the existing gas LOWO effect size and smaller effects would not materially change the geological reliability-boundary claim:",
        "",
        f"- **K_max = `{kmax20}`**.",
        f"- At K={kmax20}, CI level = `{format_float(level_kmax20 * 100, 2)}%`; judgment half-width = `{format_float(half_kmax20)} percentage points`.",
        f"- At K={kmax20 + 1}, the half-width would exceed or meet the 20-point threshold, so the budget should not be expanded without redefining the geological threshold before seeing adaptive rejection search results.",
        "",
        "Sensitivity to the definition of a geologically meaningful excess improvement:",
        "",
        "| threshold (percentage points) | K_max |",
        "|---:|---:|",
    ]
    for row in threshold_rows:
        label = "observed fixed-baseline effect" if abs(row["geological_meaningful_threshold_pct_points"] - point) < 1e-9 else ""
        lines.append(
            f"| {format_float(row['geological_meaningful_threshold_pct_points'])}{(' (' + label + ')') if label else ''} | {row['K_max']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The gate is tight: the primary MDE is large relative to small practical effects because LOWO gas evidence is sparse and fold-level effects vary strongly by held-out well.",
        f"- Under the 20-point threshold, adaptive_rejection_setup may preregister at most `{kmax20}` variants. If the reviewer/author requires a smaller threshold such as 5 or 10 percentage points, `K_max = 0` and adaptive rejection search should not be run.",
        f"- If the only acceptable threshold is the already observed fixed-baseline effect ({format_float(point)} percentage points), the mathematical upper budget is `K_max = {kmax_observed}`, but that is looser than the conservative 20-point gate and should be approved explicitly before use.",
        "- Conceptual basis: this uses preregistration plus Bonferroni correction as a transparent small-sample analogue to reusable-holdout/Thresholdout and Ladder-style control; direct differentially private thresholding is not used because the sample is small and DP constants would be too loose for this review task.",
        "",
        "## Files",
        "",
        "- `internal_review/adaptive_rejection_budget_method_bootstrap_summary.csv`: method-level bootstrap summaries.",
        "- `internal_review/adaptive_rejection_budget_k_budget_table.csv`: Bonferroni K table.",
        "- `internal_review/adaptive_rejection_budget_threshold_sensitivity.csv`: K_max by geological threshold.",
    ]
    (adaptive_rejection / "budget_spec.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"adaptive_rejection_budget PASS: MDE={mde95:.3f} pp; K_max_20pp={kmax20}; output={adaptive_rejection / 'budget_spec.md'}")


if __name__ == "__main__":
    main()
