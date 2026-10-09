# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
"""Compute zone-level LOWO model skill against the zone-mean baseline."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


BOOTSTRAP_SEED = 20260613
BOOTSTRAP_REPLICATES = 500
ZONES = ("North", "Central", "South")
KEY_FIELDS = (
    "sample_anon",
    "well_anon",
    "zone_en",
    "target",
    "cv_scheme",
    "fold_label_anon",
)
OUTPUT_FIELDS = (
    "target",
    "zone_en",
    "predictor_group",
    "model",
    "n_points",
    "n_wells",
    "MAE_model",
    "MAE_zone_mean",
    "delta_MAE",
    "relative_improvement_pct",
    "ci_low_pct",
    "ci_high_pct",
    "spearman_model",
    "skill_class",
)
DEFAULT_RESOURCE_ROOT = Path(
    'PROJECT_ROOT_PLACEHOLDER'
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return float("nan")
    pos = (len(values) - 1) * q / 100
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end + 2) / 2
        for position in range(cursor, end + 1):
            ranks[order[position]] = rank
        cursor = end + 1
    return ranks


def spearman(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) < 3:
        return float("nan")
    rank_true = np.asarray(rankdata(y_true), dtype=float)
    rank_pred = np.asarray(rankdata(y_pred), dtype=float)
    if np.std(rank_true) == 0 or np.std(rank_pred) == 0:
        return float("nan")
    return float(np.corrcoef(rank_true, rank_pred)[0, 1])


def mae(y_true: list[float], y_pred: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in KEY_FIELDS)


def paired_rows(
    lowo_rows: list[dict[str, str]], target: str, model: str, zone: str | None = None
) -> list[dict[str, object]]:
    baseline: dict[tuple[str, ...], dict[str, str]] = {}
    for row in lowo_rows:
        if row["target"] == target and row["model"] == "zone_mean":
            if zone is None or row["zone_en"] == zone:
                if key(row) in baseline:
                    raise ValueError(f"Duplicate zone_mean row for key {key(row)}")
                baseline[key(row)] = row

    paired: list[dict[str, object]] = []
    for row in lowo_rows:
        if row["target"] != target or row["model"] != model:
            continue
        if zone is not None and row["zone_en"] != zone:
            continue
        base = baseline.get(key(row))
        if base is None:
            raise ValueError(f"Missing zone_mean match for key {key(row)}")
        if not math.isclose(float(row["y_true"]), float(base["y_true"]), abs_tol=1e-12):
            raise ValueError(f"y_true mismatch for key {key(row)}")
        paired.append(
            {
                "sample_anon": row["sample_anon"],
                "well_anon": row["well_anon"],
                "zone_en": row["zone_en"],
                "fold_label_anon": row["fold_label_anon"],
                "y_true": float(row["y_true"]),
                "y_pred_model": float(row["y_pred"]),
                "y_pred_baseline": float(base["y_pred"]),
            }
        )
    if len(paired) != len(baseline):
        raise ValueError(
            f"Model/baseline row count mismatch for {target}/{model}/{zone}: "
            f"{len(paired)} vs {len(baseline)}"
        )
    return paired


def bootstrap_improvement(
    rows: list[dict[str, object]], rng: random.Random, replicates: int = BOOTSTRAP_REPLICATES
) -> tuple[float, float]:
    # This follows run_within_zone_recalibration.py:205-214 exactly.
    wells = [str(row["well_anon"]) for row in rows]
    unique_wells = sorted(set(wells))
    indices_by_well = {
        well: [index for index, value in enumerate(wells) if value == well]
        for well in unique_wells
    }
    values: list[float] = []
    for _ in range(replicates):
        indices: list[int] = []
        for well in [rng.choice(unique_wells) for _ in unique_wells]:
            indices.extend(indices_by_well[well])
        y_true = [float(rows[index]["y_true"]) for index in indices]
        model = [float(rows[index]["y_pred_model"]) for index in indices]
        baseline = [float(rows[index]["y_pred_baseline"]) for index in indices]
        baseline_mae = mae(y_true, baseline)
        model_mae = mae(y_true, model)
        if baseline_mae > 0:
            values.append((baseline_mae - model_mae) / baseline_mae * 100)
    return percentile(values, 2.5), percentile(values, 97.5)


def classify(improvement: float, ci_low: float) -> str:
    if improvement > 0 and ci_low > 0:
        return "skill_CI_robust"
    if improvement > 0:
        return "skill_directional"
    return "no_skill"


def calculate_row(
    rows: list[dict[str, object]],
    target: str,
    zone: str,
    predictor_group: str,
    model: str,
    rng: random.Random,
) -> tuple[dict[str, object], float]:
    y_true = [float(row["y_true"]) for row in rows]
    model_pred = [float(row["y_pred_model"]) for row in rows]
    baseline_pred = [float(row["y_pred_baseline"]) for row in rows]
    model_mae = mae(y_true, model_pred)
    baseline_mae = mae(y_true, baseline_pred)
    improvement = (baseline_mae - model_mae) / baseline_mae * 100
    ci_low, ci_high = bootstrap_improvement(rows, rng)

    folds: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        folds[str(row["fold_label_anon"])].append(index)
    weighted_fold_mae = sum(
        mae([y_true[index] for index in indices], [model_pred[index] for index in indices])
        * len(indices)
        for indices in folds.values()
    ) / len(rows)
    equivalence_error = abs(weighted_fold_mae - model_mae)

    result = {
        "target": target,
        "zone_en": zone,
        "predictor_group": predictor_group,
        "model": model,
        "n_points": len(rows),
        "n_wells": len({str(row["well_anon"]) for row in rows}),
        "MAE_model": f"{model_mae:.6f}",
        "MAE_zone_mean": f"{baseline_mae:.6f}",
        "delta_MAE": f"{model_mae - baseline_mae:.6f}",
        "relative_improvement_pct": f"{improvement:.6f}",
        "ci_low_pct": f"{ci_low:.6f}",
        "ci_high_pct": f"{ci_high:.6f}",
        "spearman_model": f"{spearman(y_true, model_pred):.6f}",
        "skill_class": classify(improvement, ci_low),
    }
    return result, equivalence_error


def combined_metrics(
    lowo_rows: list[dict[str, str]], target: str, model: str
) -> dict[str, float | int]:
    rows = paired_rows(lowo_rows, target, model)
    y_true = [float(row["y_true"]) for row in rows]
    model_pred = [float(row["y_pred_model"]) for row in rows]
    baseline_pred = [float(row["y_pred_baseline"]) for row in rows]
    model_mae = mae(y_true, model_pred)
    baseline_mae = mae(y_true, baseline_pred)
    return {
        "n_points": len(rows),
        "n_wells": len({str(row["well_anon"]) for row in rows}),
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "improvement": (baseline_mae - model_mae) / baseline_mae * 100,
    }


def validate_anchors(
    lowo_rows: list[dict[str, str]], s18_rows: list[dict[str, str]], s9_rows: list[dict[str, str]]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    toc = combined_metrics(lowo_rows, "toc_pct", "ExtraTrees")
    gas = combined_metrics(lowo_rows, "total_gas_m3_t", "RF")
    s18_lookup = {
        (row["target"], row["model"], row["cv_scheme"]): float(row["MAE_w_full"])
        for row in s18_rows
    }
    checks = [
        {
            "name": "TOC combined LOWO MAE",
            "passed": abs(float(toc["model_mae"]) - 0.3069) <= 0.0005
            and abs(float(toc["baseline_mae"]) - 0.256) <= 0.001
            and abs(float(toc["model_mae"]) - s18_lookup[("TOC", "ExtraTrees", "LOWO")]) <= 1e-6,
            "actual": f"ExtraTrees={float(toc['model_mae']):.6f}; zone_mean={float(toc['baseline_mae']):.6f}",
            "target": "ExtraTrees≈0.3069; zone_mean≈0.256; S18 exact match",
        },
        {
            "name": "Gas combined LOWO MAE",
            "passed": abs(float(gas["model_mae"]) - 0.390) <= 0.001
            and abs(float(gas["baseline_mae"]) - 0.426) <= 0.001
            and abs(float(gas["model_mae"]) - s18_lookup[("total_gas", "RF", "LOWO")]) <= 1e-6,
            "actual": f"RF={float(gas['model_mae']):.6f}; zone_mean={float(gas['baseline_mae']):.6f}",
            "target": "RF≈0.390; zone_mean≈0.426; S18 exact match",
        },
        {
            "name": "Sample closure",
            "passed": toc["n_points"] == 291
            and toc["n_wells"] == 32
            and gas["n_points"] == 155
            and gas["n_wells"] == 26,
            "actual": f"TOC={toc['n_points']} points/{toc['n_wells']} wells; gas={gas['n_points']} points/{gas['n_wells']} wells",
            "target": "TOC=291/32; gas=155/26",
        },
        {
            "name": "Combined improvement signs",
            "passed": float(toc["improvement"]) < 0 and float(gas["improvement"]) > 0,
            "actual": f"TOC={float(toc['improvement']):+.6f}%; gas={float(gas['improvement']):+.6f}%",
            "target": "TOC<0; gas>0",
        },
    ]
    inventory = {
        "toc_points": sum(int(float(row["toc_points"])) for row in s9_rows),
        "toc_wells": sum(int(float(row["toc_points"])) > 0 for row in s9_rows),
        "gas_points": sum(int(float(row["gas_points"])) for row in s9_rows),
        "gas_wells": sum(int(float(row["gas_points"])) > 0 for row in s9_rows),
    }
    return checks, inventory


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    checks: list[dict[str, object]],
    inventory: dict[str, int],
    equivalence_error: float,
) -> None:
    main_rows = [
        row
        for row in rows
        if row["predictor_group"]
        in {"A_fixed_backbone", "A_fixed_backbone_and_B_table3_best"}
    ]
    comparison_rows = [row for row in rows if row["predictor_group"] == "B_table3_best_envelope"]
    lines = [
        "# Within-zone LOWO model versus zone-mean skill criterion",
        "",
        "## 1. Calculation convention",
        "",
        "- Input is oof_predictions.csv filtered to LOWO; no model refitting, tuning or feature changes.",
        "- Group A fixed backbones: TOC ExtraTrees and gas RF. Group B Table 3 envelope: TOC SVR_RBF. Gas RF is shared by both groups and computed once.",
        f"- Well-bootstrap replicates B={BOOTSTRAP_REPLICATES}, random seed={BOOTSTRAP_SEED}, percentile interval 2.5%--97.5%.",
        "- zone_mean is constant within each zone; its Spearman is undefined. S30 reports model Spearman only.",
        f"- Sample-weighted LOWO fold MAE equals within-zone pooled MAE; maximum absolute difference {equivalence_error:.3e}.",
        "",
        "## 2. Anchor checks",
        "",
        "| Check | Status | Observed | Expected |",
        "|---|---|---|---|",
    ]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"| {check['name']} | {status} | {check['actual']} | {check['target']} |")
    lines.extend(
        [
            "",
            f"S9 inventory: TOC={inventory['toc_points']} points/{inventory['toc_wells']} wells, gas={inventory['gas_points']} points/{inventory['gas_wells']} wells; LOWO analysis has 291/32 and 155/26 points/wells. Well counts agree; inventory and model-ready sample counts are reported separately.",
            "",
            "## 3. Group A fixed-backbone results",
            "",
            "| Target | Zone | n_points | n_wells | MAE_model | MAE_zone_mean | Relative improvement % (95% CI) | Spearman | Skill class |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in main_rows:
        lines.append(
            f"| {row['target']} | {row['zone_en']} | {row['n_points']} | {row['n_wells']} | "
            f"{float(row['MAE_model']):.3f} | {float(row['MAE_zone_mean']):.3f} | "
            f"{float(row['relative_improvement_pct']):+.2f} [{float(row['ci_low_pct']):+.2f}, {float(row['ci_high_pct']):+.2f}] | "
            f"{float(row['spearman_model']):+.3f} | {row['skill_class']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Group B model-envelope comparison",
            "",
            "Group B is a comparison only and does not set Fig. 11 action classes. Gas uses RF in both groups and appears in one shared row.",
            "",
            "| Target | Zone | Model | Relative improvement % (95% CI) | Descriptive skill class |",
            "|---|---:|---|---:|---|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['target']} | {row['zone_en']} | {row['model']} | "
            f"{float(row['relative_improvement_pct']):+.2f} [{float(row['ci_low_pct']):+.2f}, {float(row['ci_high_pct']):+.2f}] | {row['skill_class']} |"
        )
    lines.extend(
        [
            "",
            "## 5. Fig. 11 action-class interpretation",
            "",
            "### (a) TOC panel",
            "",
            "- North remains yellow (locally calibratable). Coverage has priority; the CI-robust Schmoker C1 TOC offset gain of 7.39% supports local recalibration, independently of group A no_skill.",
            "- Central remains green (deployable with interval): coverage passes and group A has directional skill.",
            "- South is steel blue (#6B8EAE; zone-level only). Coverage passes but group A has no_skill; use a zone-level value and interval, without a depth-resolved log prediction.",
            "",
            "### (b) Gas panel",
            "",
            "- North remains red with hatching (data acquisition). Coverage has priority and no CI-robust local correction exists; group A CI-robust model skill cannot override failed coverage.",
            "- Central remains green (deployable with interval): coverage passes and group A has directional skill.",
            "- South is steel blue (#6B8EAE; zone-level only). Coverage passes but group A has no_skill; use a zone-level value and interval, without a depth-resolved log prediction.",
            "",
            "### Dependence on intervals crossing zero",
            "",
            "Central TOC and gas group A point estimates are positive but intervals cross zero (directional skill). The predefined rule keeps both green; a stricter CI-robust-only interpretation would make both steel blue.",
            "",
            "## 6. Conclusion",
            "",
            "With the skill condition, two Fig. 11 cells change: South TOC and gas become steel blue (zone-level only).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-resource-root", type=Path, default=DEFAULT_RESOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    oof_path = args.online_resource_root / "code" / "data_derived" / "oof_predictions.csv"
    s18_path = args.online_resource_root / "code" / "data_derived" / "Table_S18_logs_only_vs_full.csv"
    s9_path = args.online_resource_root / "supplementary_tables" / "Table_S9_well_inventory.csv"
    for path in (oof_path, s18_path, s9_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    oof_rows = read_csv(oof_path)
    actual_models = sorted({row["model"] for row in oof_rows})
    for required in ("ExtraTrees", "RF", "SVR_RBF", "zone_mean"):
        if required not in actual_models:
            raise ValueError(f"Required model {required!r} not found; actual={actual_models}")
    lowo_rows = [row for row in oof_rows if row["cv_scheme"] == "LOWO"]

    checks, inventory = validate_anchors(lowo_rows, read_csv(s18_path), read_csv(s9_path))
    failed = [check for check in checks if not check["passed"]]
    if failed:
        for check in failed:
            print(f"ANCHOR FAIL: {check['name']}: {check['actual']} vs {check['target']}")
        return 2

    rng = random.Random(BOOTSTRAP_SEED)
    specifications = (
        ("toc_pct", "A_fixed_backbone", "ExtraTrees"),
        ("toc_pct", "B_table3_best_envelope", "SVR_RBF"),
        ("total_gas_m3_t", "A_fixed_backbone_and_B_table3_best", "RF"),
    )
    results: list[dict[str, object]] = []
    equivalence_errors: list[float] = []
    for target, predictor_group, model in specifications:
        for zone in ZONES:
            row, error = calculate_row(
                paired_rows(lowo_rows, target, model, zone),
                target,
                zone,
                predictor_group,
                model,
                rng,
            )
            results.append(row)
            equivalence_errors.append(error)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "Table_S30_zone_level_skill_criterion.csv", results)
    write_report(
        args.output_dir / "report_zone_skill_criterion.md",
        results,
        checks,
        inventory,
        max(equivalence_errors),
    )
    print(f"PASS: wrote {len(results)} rows; models={actual_models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
