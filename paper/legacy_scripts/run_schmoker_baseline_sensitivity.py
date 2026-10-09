# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from __future__ import annotations

import csv
import difflib
import importlib.util
import math
import random
from pathlib import Path

import pandas as pd


BOOTSTRAP_SEED = 20260613
BOOTSTRAP_N = 500


def percentile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mae(y: list[float], pred: list[float]) -> float:
    return sum(abs(float(a) - float(b)) for a, b in zip(y, pred)) / len(y)


def advance_rng(rng: random.Random, unique_count: int, n_boot: int) -> None:
    dummy = list(range(unique_count))
    for _ in range(n_boot):
        for _ in dummy:
            rng.choice(dummy)


class CanonicalBootstrap:
    """Legacy canonical by-well percentile bootstrap used by benefit_summary.csv."""

    def __init__(self) -> None:
        self.rng = random.Random(BOOTSTRAP_SEED)
        self.calls = 0
        self.pre_advance_description = (
            "Advance RNG through canonical run_within_zone_recalibration.py sequence before "
            "North TOC/Schmoker benefit: Central metrics 6x400x7, Central benefit "
            "500x7, and North metrics 6x400x8."
        )
        for unique_count, n_boot in ([(7, 400)] * 6 + [(7, 500)] + [(8, 400)] * 6):
            advance_rng(self.rng, unique_count, n_boot)

    def __call__(
        self,
        y: list[float],
        pred: list[float],
        baseline: list[float],
        wells: list[str],
        n_boot: int = BOOTSTRAP_N,
        seed: int = BOOTSTRAP_SEED,
    ) -> tuple[float, float]:
        self.calls += 1
        unique = sorted(set(wells))
        if len(unique) < 3:
            return float("nan"), float("nan")
        idx_by_well = {well: [i for i, value in enumerate(wells) if value == well] for well in unique}
        values: list[float] = []
        for _ in range(n_boot):
            idx: list[int] = []
            for sampled in [self.rng.choice(unique) for _ in unique]:
                idx.extend(idx_by_well[sampled])
            yy = [y[i] for i in idx]
            pp = [pred[i] for i in idx]
            zz = [baseline[i] for i in idx]
            base = mae(yy, zz)
            corr = mae(yy, pp)
            if base > 0:
                values.append((base - corr) / base * 100.0)
        if len(values) < 20:
            return float("nan"), float("nan")
        return percentile(values, 2.5), percentile(values, 97.5)


def find_stage_si() -> Path:
    matches = [
        p.parent
        for p in Path("PROJECT_ROOT_PLACEHOLDER").glob("*/manscript/SI/Table_S24c_schmoker_baseline_sensitivity.csv")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one Stage-2 SI directory, found {len(matches)}: {matches}")
    return matches[0]


def find_protocol_script(root: Path) -> Path:
    matches = list(root.glob("EVO*/**/protocol_sensitivity/run_protocol_sensitivity.py"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one protocol sensitivity script, found {len(matches)}: {matches}")
    return matches[0]


def find_canonical_benefit(root: Path) -> Path:
    candidates = list(root.glob("EVO*/**/benefit_summary.csv"))
    accepted: list[Path] = []
    for path in candidates:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        required = {
            "target",
            "proxy",
            "zone_en",
            "best_method",
            "relative_MAE_improvement_vs_zonemean_pct",
            "improvement_ci_low_pct",
            "improvement_ci_high_pct",
        }
        if not required.issubset(df.columns):
            continue
        row = df[
            df["target"].astype(str).eq("TOC")
            & df["proxy"].astype(str).eq("schmoker_toc_proxy")
            & df["zone_en"].astype(str).eq("North")
            & df["best_method"].astype(str).eq("C1_offset")
        ]
        if row.empty:
            continue
        lo = float(row["improvement_ci_low_pct"].iloc[0])
        hi = float(row["improvement_ci_high_pct"].iloc[0])
        if abs(lo - 1.17358) < 1e-3 and abs(hi - 25.225) < 1e-3:
            accepted.append(path)
    if len(accepted) != 1:
        raise RuntimeError(f"Expected exactly one canonical benefit_summary.csv, found {len(accepted)}: {accepted}")
    return accepted[0]


def import_protocol_module(script: Path):
    spec = importlib.util.spec_from_file_location("protocol_sensitivity", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def to_text_csv(df: pd.DataFrame) -> list[str]:
    return df.to_csv(index=False, encoding=None, lineterminator="\n").splitlines(True)


def write_report(
    audit_dir: Path,
    protocol_script: Path,
    canonical_benefit: Path,
    source_si: Path,
    recomputed_path: Path,
    diff_path: Path,
    original: pd.DataFrame,
    recomputed: pd.DataFrame,
    bootstrapper: CanonicalBootstrap,
) -> None:
    canonical = pd.read_csv(canonical_benefit, encoding="utf-8-sig")
    c_north = canonical[
        canonical["target"].astype(str).eq("TOC")
        & canonical["proxy"].astype(str).eq("schmoker_toc_proxy")
        & canonical["zone_en"].astype(str).eq("North")
        & canonical["best_method"].astype(str).eq("C1_offset")
    ].iloc[0]
    north = recomputed[
        recomputed["zone_en"].astype(str).eq("North") & recomputed["variant"].astype(str).eq("baseline")
    ].iloc[0]
    protected = [
        c
        for c in original.columns
        if c not in {"improvement_ci_low_pct", "improvement_ci_high_pct"}
    ]
    changed_protected = []
    for col in protected:
        for i, (old, new) in enumerate(zip(original[col], recomputed[col]), start=1):
            if str(old) != str(new):
                try:
                    if close(float(old), float(new), 1e-12):
                        continue
                except Exception:
                    pass
                changed_protected.append((i, col, old, new))

    north_rows = recomputed[recomputed["zone_en"].astype(str).eq("North")]
    central_south_rows = recomputed[recomputed["zone_en"].astype(str).isin(["Central", "South"])]
    point_targets = {
        "B1_strict_fold_density_baseline": 7.365,
        "B2_density_quantile_0.85": 7.302,
        "B2_density_quantile_0.95": 7.553,
    }
    point_checks = []
    for variant, target in point_targets.items():
        row = north_rows[north_rows["variant"].astype(str).eq(variant)].iloc[0]
        point_checks.append((variant, float(row["relative_improvement_pct"]), target))

    checks = []
    checks.append(("row_count", len(recomputed) == 12, f"{len(recomputed)} rows"))
    checks.append(("columns_unchanged", list(original.columns) == list(recomputed.columns), "same columns"))
    checks.append(("non_ci_values_unchanged", len(changed_protected) == 0, f"{len(changed_protected)} protected changes"))
    checks.append(
        (
            "north_baseline_ci",
            abs(float(north["improvement_ci_low_pct"]) - float(c_north["improvement_ci_low_pct"])) < 1e-5
            and abs(float(north["improvement_ci_high_pct"]) - float(c_north["improvement_ci_high_pct"])) < 1e-3,
            f"{float(north['improvement_ci_low_pct']):.6f}, {float(north['improvement_ci_high_pct']):.6f}",
        )
    )
    checks.append(
        (
            "north_baseline_point_and_mae",
            abs(float(north["relative_improvement_pct"]) - 7.3887) < 1e-3
            and abs(float(north["corrected_MAE"]) - 0.224708) < 1e-6
            and abs(float(north["baseline_MAE"]) - 0.242636) < 1e-6,
            f"improvement={float(north['relative_improvement_pct']):.6f}",
        )
    )
    for variant, observed, target in point_checks:
        checks.append((f"north_point_{variant}", abs(observed - target) < 0.001, f"{observed:.6f} vs {target:.3f}"))
    checks.append(
        (
            "north_all_ci_positive",
            bool((north_rows["improvement_ci_low_pct"].astype(float) > 0).all()),
            "all North lower bounds > 0",
        )
    )
    checks.append(
        (
            "central_south_cross_zero",
            bool(
                (
                    (central_south_rows["improvement_ci_low_pct"].astype(float) < 0)
                    & (central_south_rows["improvement_ci_high_pct"].astype(float) > 0)
                ).all()
            ),
            "all Central/South intervals cross zero",
        )
    )

    status = "PASS" if all(ok for _, ok, _ in checks) else "FAIL"
    lines = [
        "# S24c CI Reconcile Report",
        "",
        f"Status: **{status}**",
        "",
        "## Sources",
        "",
        f"- Protocol sensitivity script: `{protocol_script}`",
        f"- Canonical benefit source: `{canonical_benefit}`",
        f"- Current authoritative SI source: `{source_si}`",
        "",
        "## Canonical Bootstrap Protocol",
        "",
        "- Unit: by-well resampling.",
        f"- Bootstrap repetitions: {BOOTSTRAP_N}.",
        f"- Seed: {BOOTSTRAP_SEED}.",
        "- RNG: Python `random.Random`, matching `run_within_zone_recalibration.py`.",
        "- CI method: empirical percentile, 2.5% and 97.5%.",
        f"- RNG alignment: {bootstrapper.pre_advance_description}",
        "",
        "## North Baseline Anchor",
        "",
        "| field | canonical target | recomputed |",
        "| --- | ---: | ---: |",
        f"| relative improvement % | {float(c_north['relative_MAE_improvement_vs_zonemean_pct']):.6f} | {float(north['relative_improvement_pct']):.6f} |",
        f"| CI low % | {float(c_north['improvement_ci_low_pct']):.6f} | {float(north['improvement_ci_low_pct']):.6f} |",
        f"| CI high % | {float(c_north['improvement_ci_high_pct']):.6f} | {float(north['improvement_ci_high_pct']):.6f} |",
        f"| corrected MAE | {float(c_north['best_MAE']):.6f} | {float(north['corrected_MAE']):.6f} |",
        f"| zone-mean MAE | {float(c_north['zone_mean_MAE']):.6f} | {float(north['baseline_MAE']):.6f} |",
        "",
        "## Validation Checks",
        "",
        "| check | result | evidence |",
        "| --- | --- | --- |",
    ]
    for name, ok, evidence in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {evidence} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Recomputed S24c: `{recomputed_path}`",
            f"- Unified diff vs current SI: `{diff_path}`",
            "",
            "No authoritative SI file was overwritten in this step.",
        ]
    )
    (audit_dir / "S24c_ci_reconcile_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (audit_dir / "S24c_ci_validation_checks.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "evidence"])
        writer.writeheader()
        for name, ok, evidence in checks:
            writer.writerow({"check": name, "status": "PASS" if ok else "FAIL", "evidence": evidence})


def main() -> None:
    root = Path.cwd()
    si_dir = find_stage_si()
    audit_dir = si_dir.parent / "code_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    source_si = si_dir / "Table_S24c_schmoker_baseline_sensitivity.csv"
    protocol_script = find_protocol_script(root)
    canonical_benefit = find_canonical_benefit(root)

    module = import_protocol_module(protocol_script)
    bootstrapper = CanonicalBootstrap()
    module.bootstrap_improvement = bootstrapper
    raw = module.load_raw()
    calculated = module.run_schmoker_sensitivity(raw)
    original = pd.read_csv(source_si, encoding="utf-8-sig")
    calculated = calculated[original.columns]
    recomputed = original.copy()
    for col in ["improvement_ci_low_pct", "improvement_ci_high_pct"]:
        recomputed[col] = calculated[col].values

    recomputed_path = audit_dir / "Table_S24c_schmoker_baseline_sensitivity_recomputed.csv"
    diff_path = audit_dir / "Table_S24c_schmoker_baseline_sensitivity_recomputed.diff"
    recomputed.to_csv(recomputed_path, index=False, encoding="utf-8-sig")
    diff = difflib.unified_diff(
        to_text_csv(original),
        to_text_csv(recomputed),
        fromfile=str(source_si),
        tofile=str(recomputed_path),
        lineterm="",
    )
    diff_path.write_text("".join(diff), encoding="utf-8")
    write_report(
        audit_dir,
        protocol_script,
        canonical_benefit,
        source_si,
        recomputed_path,
        diff_path,
        original,
        recomputed,
        bootstrapper,
    )
    print("Wrote", recomputed_path)
    print("Wrote", diff_path)
    print("Wrote", audit_dir / "S24c_ci_reconcile_report.md")


if __name__ == "__main__":
    main()
