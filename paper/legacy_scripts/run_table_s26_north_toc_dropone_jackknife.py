"""Build anonymized Table S26 from the locked North TOC C1 robustness output.

The public per-sample prediction CSV is intentionally rounded and cannot meet
the instruction's 1e-6 full-sample gate on its own. Therefore this script uses
the canonical robustness result produced by the same LOWO C1 protocol as the
numerical source, while checking public prediction coverage and aliases.
It only creates new files and refuses to overwrite existing outputs.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


ANCHOR_IMPROVEMENT = 7.388704586280177
ANCHOR_BASELINE_MAE = 0.242636019723279
ANCHOR_CORRECTED_MAE = 0.2247083610060175
CLAIM_MIN = 5.19
CLAIM_MAX = 21.01
OUTPUT_COLUMNS = [
    "zone_en",
    "dropped_well_anon",
    "n_wells_remaining",
    "n_points_remaining",
    "baseline_MAE",
    "corrected_MAE",
    "relative_improvement_pct",
    "mean_offset",
]


def require_close(label: str, observed: float, expected: float, tolerance: float = 1e-6) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise RuntimeError(
            f"Reproduction gate failed for {label}: expected {expected:.15f}, got {observed:.15f}."
        )


def check_public_prediction_coverage(path: Path) -> None:
    predictions = pd.read_csv(path, encoding="utf-8-sig")
    north_c1 = predictions[
        predictions["target"].eq("TOC")
        & predictions["proxy"].eq("schmoker_toc_proxy")
        & predictions["zone_en"].eq("North")
        & predictions["method"].eq("C1_offset")
    ]
    aliases = north_c1["well_anon"].astype(str)
    if len(north_c1) != 93 or aliases.nunique() != 8:
        raise RuntimeError("Public prediction coverage is not the expected 93 North samples across 8 wells.")
    if not aliases.str.fullmatch(r"W\d{4}").all():
        raise RuntimeError("Public prediction anonymization check failed.")


def find_s24c_anchor(path: Path) -> pd.Series:
    table = pd.read_csv(path, encoding="utf-8-sig")
    rows = table[
        table["zone_en"].eq("North")
        & table["variant"].eq("baseline")
        & table["density_quantile"].eq(0.90)
    ]
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one North baseline row in Table S24c.")
    return rows.iloc[0]


def find_s4_anchor(path: Path) -> pd.Series:
    table = pd.read_csv(path, encoding="utf-8-sig")
    rows = table[
        table["target"].eq("TOC")
        & table["zone"].eq("North")
        & table["proxy"].eq("schmoker_toc_proxy")
        & table["method"].eq("C1_offset")
    ]
    if len(rows) != 1:
        raise RuntimeError("Expected exactly one North TOC Schmoker C1 row in Table S4.")
    return rows.iloc[0]


def normalize_alias(case_name: str) -> str:
    if not str(case_name).startswith("drop_W"):
        raise RuntimeError(f"Unexpected drop-one case name: {case_name}")
    digits = str(case_name).removeprefix("drop_W")
    if not digits.isdigit():
        raise RuntimeError(f"Malformed anonymous well alias: {case_name}")
    return f"W{int(digits):04d}"


def rows_from_canonical(path: Path) -> pd.DataFrame:
    canonical = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "case_anon",
        "n_points",
        "n_wells",
        "baseline_MAE",
        "corrected_MAE",
        "relative_improvement_pct",
        "mean_offset",
    }
    if not required.issubset(canonical.columns):
        raise RuntimeError("Canonical robustness input has an unexpected schema.")
    original = canonical[canonical["case_anon"].eq("original")]
    drops = canonical[canonical["case_anon"].astype(str).str.startswith("drop_W")].copy()
    if len(original) != 1 or len(drops) != 8:
        raise RuntimeError("Expected one original case and eight canonical drop-one cases.")
    output_rows = [
        {
            "zone_en": "North",
            "dropped_well_anon": "none_full_sample",
            "n_wells_remaining": int(original["n_wells"].iloc[0]),
            "n_points_remaining": int(original["n_points"].iloc[0]),
            "baseline_MAE": float(original["baseline_MAE"].iloc[0]),
            "corrected_MAE": float(original["corrected_MAE"].iloc[0]),
            "relative_improvement_pct": float(original["relative_improvement_pct"].iloc[0]),
            "mean_offset": float(original["mean_offset"].iloc[0]),
        }
    ]
    for _, row in drops.sort_values("case_anon").iterrows():
        output_rows.append(
            {
                "zone_en": "North",
                "dropped_well_anon": normalize_alias(str(row["case_anon"])),
                "n_wells_remaining": int(row["n_wells"]),
                "n_points_remaining": int(row["n_points"]),
                "baseline_MAE": float(row["baseline_MAE"]),
                "corrected_MAE": float(row["corrected_MAE"]),
                "relative_improvement_pct": float(row["relative_improvement_pct"]),
                "mean_offset": float(row["mean_offset"]),
            }
        )
    return pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)


def build_report(rows: pd.DataFrame, source_key: str) -> str:
    full = rows[rows["dropped_well_anon"].eq("none_full_sample")].iloc[0]
    jackknife = rows[rows["dropped_well_anon"].ne("none_full_sample")].copy()
    minimum = float(jackknife["relative_improvement_pct"].min())
    maximum = float(jackknife["relative_improvement_pct"].max())
    all_positive = bool((jackknife["relative_improvement_pct"] > 0).all())
    matches = round(minimum, 2) == CLAIM_MIN and round(maximum, 2) == CLAIM_MAX
    comparison = "matches" if matches else f"does not match; observed {minimum:.6f}% to {maximum:.6f}%"
    return "\n".join(
        [
            "# Table S26: North TOC drop-one-well jackknife",
            "",
            "## Reproduction gate",
            "",
            f"PASS. The canonical full-sample North TOC / Schmoker / C1 offset relative improvement is {float(full['relative_improvement_pct']):.15f}%, matching the required {ANCHOR_IMPROVEMENT:.15f}% within 1e-6.",
            f"The canonical full-sample baseline and corrected MAEs are {float(full['baseline_MAE']):.15f} and {float(full['corrected_MAE']):.15f}.",
            "",
            "## Source and precision note",
            "",
            f"The numerical source is `{source_key}`, the locked output of the canonical North TOC Schmoker C1 LOWO protocol. The public per-sample prediction CSV was used for the independent 93-sample/8-well and alias-format check, but its exported rounding is insufficient for the required 1e-6 numerical gate. No post hoc fitting or parameter adjustment was performed.",
            "",
            "## Jackknife result",
            "",
            f"- Eight drop-one-well estimates range from {minimum:.6f}% to {maximum:.6f}%.",
            f"- The manuscript range 5.19% to 21.01% {comparison}.",
            f"- All eight drop-one estimates are {'positive' if all_positive else 'not all positive'}.",
            "",
            "## Protocol",
            "",
            "Each omitted-well case retains seven North wells. The C1 additive offset and the zone-mean baseline were recomputed on the retained training wells, rather than reusing the full-sample MAE.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-resource-root", type=Path, required=True)
    parser.add_argument("--canonical-robustness", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.online_resource_root.resolve()
    data_dir = root / "code" / "data_derived"
    table_dir = root / "supplementary_tables"
    output_name = "Table_S26_north_toc_dropone_jackknife.csv"
    data_output = data_dir / output_name
    table_output = table_dir / output_name
    report_output = table_dir / "Table_S26_north_toc_dropone_jackknife_report.md"

    check_public_prediction_coverage(data_dir / "recalibration_predictions.csv")
    s24c = find_s24c_anchor(table_dir / "Table_S24c_schmoker_baseline_sensitivity.csv")
    s4 = find_s4_anchor(table_dir / "Table_S4_recalibration_all_methods.csv")
    output = rows_from_canonical(args.canonical_robustness)
    full = output[output["dropped_well_anon"].eq("none_full_sample")].iloc[0]
    require_close("canonical full-sample improvement", float(full["relative_improvement_pct"]), ANCHOR_IMPROVEMENT)
    require_close("Table S24c improvement", float(s24c["relative_improvement_pct"]), ANCHOR_IMPROVEMENT)
    require_close("Table S4 improvement", float(s4["relative_MAE_improvement_vs_zonemean_pct"]), ANCHOR_IMPROVEMENT)
    require_close("canonical full-sample baseline MAE", float(full["baseline_MAE"]), ANCHOR_BASELINE_MAE)
    require_close("canonical full-sample corrected MAE", float(full["corrected_MAE"]), ANCHOR_CORRECTED_MAE)
    if len(output) != 9 or output["dropped_well_anon"].duplicated().any():
        raise RuntimeError("Expected exactly one full-sample row and eight unique jackknife rows.")
    aliases = output.loc[output["dropped_well_anon"].ne("none_full_sample"), "dropped_well_anon"]
    if not aliases.str.fullmatch(r"W\d{4}").all():
        raise RuntimeError("Output anonymization validation failed.")
    if args.dry_run:
        print("PASS: reproduction and anonymization gates passed; no outputs written.")
        return
    for output_path in (data_output, table_output, report_output):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output.to_csv(data_output, index=False, encoding="utf-8-sig")
    shutil.copyfile(data_output, table_output)
    report_output.write_text(build_report(output, args.canonical_robustness.name), encoding="utf-8")
    print(f"PASS: wrote {data_output}")
    print(f"PASS: wrote {table_output}")
    print(f"PASS: wrote {report_output}")


if __name__ == "__main__":
    main()
