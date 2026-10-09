from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import distance_matrix
from scipy.stats import spearmanr
from risk_tie_rule import percentile_rank, rounded_component


TARGETS = ("TOC", "total_gas")
ZONE_ORDER = ("North", "Central", "South")
ZONE_COLORS = {"North": "#D55E00", "Central": "#0072B2", "South": "#009E73"}
ZONE_MARKERS = {"North": "o", "Central": "s", "South": "^"}
RISK_WEIGHTS = {
    "interval_width": 0.55,
    "domain_knn": 0.25,
    "missing_risk": 0.10,
    "quality_risk": 0.10,
}
MAP_XLIM = (106.70, 107.50)
MAP_YLIM = (37.40, 39.30)
PERMUTATIONS = 999
RANDOM_SEED = 42
MIN_WELLS = 15
MIN_PAIRS_PER_BIN = 10
REQUESTED_BINS = 12
MIN_WIDTH_CV = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Patch A to the well-level reliability-map analysis."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--publish-base",
        type=Path,
        help="Optional manuscript root containing supplementary_tables and code/data_derived.",
    )
    return parser.parse_args()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def robust_distance_versions(logs: pd.DataFrame) -> pd.DataFrame:
    ranked_parts: list[pd.DataFrame] = []
    for target, group in logs.groupby("target", sort=False):
        part = group.copy()
        distances = part["domain_knn"].astype(float)
        p05 = float(distances.quantile(0.05))
        p95 = float(distances.quantile(0.95))
        if not np.isfinite(p05) or not np.isfinite(p95) or p95 <= p05:
            raise RuntimeError(f"Invalid robust distance scale for {target}: p05={p05}, p95={p95}")
        part["domain_p05"] = p05
        part["domain_p95"] = p95
        part["domain_knn_normalized_P"] = ((distances - p05) / (p95 - p05)).clip(0, 1)
        part["domain_knn_normalized_R"] = percentile_rank(distances)

        # q90 is the frozen half-width from the previously passed deployment calibration.
        for version in ("P", "R"):
            dhat = part[f"domain_knn_normalized_{version}"]
            part[f"interval_width_v2_{version}"] = 2.0 * part["q90"] * (1.0 + 0.5 * dhat)
            part[f"interval_width_pct_v2_{version}"] = percentile_rank(part[
                f"interval_width_v2_{version}"
            ])

        part["domain_knn_pct_v2"] = percentile_rank(distances)
        part["missing_risk_pct_v2"] = percentile_rank(part["missing_risk"])
        # quality_risk is constant (0.1); author-approved average ties retain
        # one common percentile and contribute no ordering information.
        part["quality_risk_pct_v2"] = percentile_rank(part["quality_risk"])
        for version in ("P", "R"):
            part[f"composite_risk_v2_{version}"] = (
                RISK_WEIGHTS["interval_width"] * part[f"interval_width_pct_v2_{version}"]
                + RISK_WEIGHTS["domain_knn"] * part["domain_knn_pct_v2"]
                + RISK_WEIGHTS["missing_risk"] * part["missing_risk_pct_v2"]
                + RISK_WEIGHTS["quality_risk"] * part["quality_risk_pct_v2"]
            )
        ranked_parts.append(part)
    return pd.concat(ranked_parts, ignore_index=True)


def build_s28_v2(previous_s28: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = audit[audit["source"].eq("logs_only_deployment")].copy()
    if logs.empty:
        raise RuntimeError("No logs-only deployment rows found in risk_component_audit.csv")
    if not np.allclose(logs["quality_risk"].astype(float), 0.1):
        raise RuntimeError("quality_risk is not the expected constant 0.1")
    ranked = robust_distance_versions(logs)

    per_model = (
        ranked.groupby(["target", "model", "well_id", "zone_en"], as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            domain_p05=("domain_p05", "first"),
            domain_p95=("domain_p95", "first"),
            median_domain_knn=("domain_knn", "median"),
            median_missing_risk=("missing_risk", "median"),
            domain_knn_normalized_P_median=("domain_knn_normalized_P", "median"),
            domain_knn_normalized_R_median=("domain_knn_normalized_R", "median"),
            median_interval_width_v2_P=("interval_width_v2_P", "median"),
            median_interval_width_v2_R=("interval_width_v2_R", "median"),
            composite_risk_v2_P=("composite_risk_v2_P", "median"),
            composite_risk_v2_R=("composite_risk_v2_R", "median"),
            composite_risk_v2_P_iqr=("composite_risk_v2_P", iqr),
            composite_risk_v2_R_iqr=("composite_risk_v2_R", iqr),
        )
    )
    well_level = (
        per_model.groupby(["target", "well_id", "zone_en"], as_index=False)
        .agg(
            n_samples_v2=("n_samples", "median"),
            domain_p05=("domain_p05", "median"),
            domain_p95=("domain_p95", "median"),
            domain_knn_normalized_P_median=("domain_knn_normalized_P_median", "median"),
            domain_knn_normalized_R_median=("domain_knn_normalized_R_median", "median"),
            median_interval_width_v2_P=("median_interval_width_v2_P", "median"),
            median_interval_width_v2_R=("median_interval_width_v2_R", "median"),
            composite_risk_v2_P=("composite_risk_v2_P", "median"),
            composite_risk_v2_R=("composite_risk_v2_R", "median"),
            composite_risk_v2_P_iqr=("composite_risk_v2_P_iqr", "median"),
            composite_risk_v2_R_iqr=("composite_risk_v2_R_iqr", "median"),
        )
    )
    well_level["n_samples_v2"] = well_level["n_samples_v2"].astype(int)
    well_level["composite_risk_v2_consensus"] = well_level[
        ["composite_risk_v2_P", "composite_risk_v2_R"]
    ].median(axis=1)

    result = previous_s28.merge(
        well_level,
        on=["target", "well_id", "zone_en"],
        how="left",
        validate="one_to_one",
    )
    result["used_in_spatial_test"] = result["is_calibration"].eq(0)
    missing_logs = result.loc[
        result["used_in_spatial_test"] & result["composite_risk_v2_P"].isna(),
        ["target", "well_id"],
    ]
    if not missing_logs.empty:
        raise RuntimeError(f"Missing v2 risk for logs-only wells: {missing_logs.to_dict('records')}")
    return result.sort_values(["target", "zone_en", "well_id"]), ranked


def width_summary(s28: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    logs = s28[s28["used_in_spatial_test"]].copy()
    for target in TARGETS:
        target_rows = logs[logs["target"].eq(target)]
        for version in ("P", "R"):
            values = target_rows[f"median_interval_width_v2_{version}"].astype(float)
            cv = float(values.std(ddof=1) / values.mean())
            rows.append(
                {
                    "target": target,
                    "normalization": version,
                    "n_logs_only_wells": len(values),
                    "width_min": float(values.min()),
                    "width_max": float(values.max()),
                    "width_mean": float(values.mean()),
                    "width_sd": float(values.std(ddof=1)),
                    "width_cv": cv,
                    "cv_gate_passed": bool(cv >= MIN_WIDTH_CV),
                }
            )
    return pd.DataFrame(rows)


def lonlat_xy_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    mean_lat = np.deg2rad(np.mean(lat))
    x = lon * 111.320 * np.cos(mean_lat)
    y = lat * 110.574
    return np.column_stack([x, y])


def moran_knn(
    values: np.ndarray,
    xy: np.ndarray,
    k: int,
    permutations: int = PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    n = len(values)
    k_eff = min(k, n - 1)
    distances = distance_matrix(xy, xy)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :k_eff]
    weights = np.zeros((n, n), dtype=float)
    for row, columns in enumerate(neighbors):
        weights[row, columns] = 1.0 / k_eff
    centered = values - values.mean()
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return np.nan, np.nan
    observed = float((n / weights.sum()) * np.sum(weights * np.outer(centered, centered)) / denominator)
    rng = np.random.default_rng(seed)
    permuted = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = rng.permutation(values)
        z = shuffled - shuffled.mean()
        permuted[index] = (n / weights.sum()) * np.sum(weights * np.outer(z, z)) / np.sum(z**2)
    p_positive = float((1 + np.sum(permuted >= observed)) / (permutations + 1))
    return observed, p_positive


def empirical_variogram(values: np.ndarray, xy: np.ndarray) -> tuple[pd.DataFrame, int, int]:
    pairs: list[tuple[float, float]] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            distance = float(np.linalg.norm(xy[i] - xy[j]))
            semivariance = float(0.5 * (values[i] - values[j]) ** 2)
            pairs.append((distance, semivariance))
    pairs.sort(key=lambda item: item[0])
    pair_frame = pd.DataFrame(pairs, columns=["distance_km", "semivariance"])
    max_bins = len(pair_frame) // MIN_PAIRS_PER_BIN
    bins = min(REQUESTED_BINS, max_bins)
    if bins < 1:
        return pd.DataFrame(columns=["bin", "distance_km", "semivariance", "n_pairs"]), max_bins, len(pair_frame)
    chunks = [pair_frame.iloc[index] for index in np.array_split(np.arange(len(pair_frame)), bins)]
    rows = [
        {
            "bin": index,
            "distance_km": float(chunk["distance_km"].mean()),
            "semivariance": float(chunk["semivariance"].mean()),
            "n_pairs": int(len(chunk)),
        }
        for index, chunk in enumerate(chunks, 1)
    ]
    return pd.DataFrame(rows), max_bins, len(pair_frame)


def variogram_diagnostics(variogram: pd.DataFrame) -> dict[str, object]:
    if variogram.empty:
        return {
            "rho": np.nan,
            "rho_p": np.nan,
            "early_slope": np.nan,
            "late_slope": np.nan,
            "tail_head_ratio": np.nan,
            "nugget": np.nan,
            "sill": np.nan,
            "nugget_to_sill_ratio": np.nan,
            "nugget_dominated": True,
            "rise_to_sill": False,
            "usable": False,
        }
    y = variogram["semivariance"].to_numpy(float)
    nugget = float(y[0])
    sill = float(np.mean(y[-min(3, len(y)) :]))
    nugget_to_sill = nugget / sill if sill > 0 else np.inf
    if len(variogram) < 4:
        return {
            "rho": np.nan,
            "rho_p": np.nan,
            "early_slope": np.nan,
            "late_slope": np.nan,
            "tail_head_ratio": np.nan,
            "nugget": nugget,
            "sill": sill,
            "nugget_to_sill_ratio": nugget_to_sill,
            "nugget_dominated": bool(nugget_to_sill > 0.75),
            "rise_to_sill": False,
            "usable": False,
        }
    x = variogram["distance_km"].to_numpy(float)
    rho, rho_p = spearmanr(x, y)
    midpoint = len(variogram) // 2
    early_slope = float(np.polyfit(x[: midpoint + 1], y[: midpoint + 1], 1)[0])
    late_slope = float(np.polyfit(x[midpoint:], y[midpoint:], 1)[0])
    head = float(np.mean(y[:2]))
    tail = float(np.mean(y[-2:]))
    tail_head_ratio = tail / head if head > 0 else np.inf
    rise_to_sill = bool(
        rho > 0.5
        and early_slope > 0
        and tail_head_ratio >= 1.10
        and abs(late_slope) <= 0.5 * abs(early_slope)
    )
    nugget_dominated = bool(nugget_to_sill > 0.75)
    return {
        "rho": float(rho),
        "rho_p": float(rho_p),
        "early_slope": early_slope,
        "late_slope": late_slope,
        "tail_head_ratio": tail_head_ratio,
        "nugget": nugget,
        "sill": sill,
        "nugget_to_sill_ratio": nugget_to_sill,
        "nugget_dominated": nugget_dominated,
        "rise_to_sill": rise_to_sill,
        "usable": bool(rise_to_sill and not nugget_dominated),
    }


def nearest_neighbor_stats(xy: np.ndarray) -> tuple[float, float, float]:
    distances = distance_matrix(xy, xy)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return tuple(float(value) for value in np.quantile(nearest, [0.25, 0.50, 0.75]))


def plot_variogram(
    variogram: pd.DataFrame,
    target: str,
    zone: str,
    version: str,
    diagnostics: dict[str, object],
    preliminary_pass: bool,
    output: Path,
) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(3.54, 2.60), constrained_layout=True)
    color = ZONE_COLORS[zone]
    marker = ZONE_MARKERS[zone]
    line_style = "-" if version == "P" else "--"
    face = color if version == "P" else "white"
    if not variogram.empty:
        ax.plot(
            variogram["distance_km"],
            variogram["semivariance"],
            color=color,
            linestyle=line_style,
            marker=marker,
            markerfacecolor=face,
            markeredgecolor="black",
            markeredgewidth=0.35,
            markersize=4.2,
            linewidth=1.1,
        )
        sill = float(diagnostics["sill"])
        if np.isfinite(sill):
            ax.axhline(sill, color="#666666", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Pair distance (km)")
    ax.set_ylabel("Semivariance")
    ratio = diagnostics["nugget_to_sill_ratio"]
    ratio_text = "NA" if not np.isfinite(float(ratio)) else f"{float(ratio):.2f}"
    ax.text(
        0.0,
        1.02,
        f"{target.replace('_', ' ')} | {zone} | {version}\n"
        f"bins={len(variogram)}; nugget/sill={ratio_text}; "
        f"criterion={'pass' if preliminary_pass else 'fail'}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        clip_on=False,
    )
    ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output.with_suffix(".svg"), transparent=False)
    fig.savefig(output.with_suffix(".pdf"), transparent=False)
    fig.savefig(output.with_suffix(".png"), dpi=600, transparent=False)
    plt.close(fig)


def build_spatial_tests(
    s28: pd.DataFrame,
    previous_s29: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    logs = s28[s28["used_in_spatial_test"]].copy()
    rows: list[dict[str, object]] = []
    variograms: dict[tuple[str, str, str], pd.DataFrame] = {}
    for target in TARGETS:
        for zone in ZONE_ORDER:
            group = logs[logs["target"].eq(target) & logs["zone_en"].eq(zone)].copy()
            if group.empty:
                continue
            xy = lonlat_xy_km(group["lon"].to_numpy(float), group["lat"].to_numpy(float))
            q1, median, q3 = nearest_neighbor_stats(xy)
            for version in ("P", "R"):
                values = group[f"composite_risk_v2_{version}"].to_numpy(float)
                i4, p4 = moran_knn(values, xy, 4)
                i6, p6 = moran_knn(values, xy, 6)
                variogram, max_bins, n_pairs = empirical_variogram(values, xy)
                diagnostics = variogram_diagnostics(variogram)
                n_pass = len(group) >= MIN_WELLS
                moran4_pass = bool(i4 > 0 and p4 < 0.05)
                moran6_pass = bool(i6 > 0 and p6 < 0.05)
                preliminary_pass = bool(n_pass and moran4_pass and moran6_pass and diagnostics["usable"])
                variograms[(target, zone, version)] = variogram
                rows.append(
                    {
                        "target": target,
                        "zone_en": zone,
                        "normalization": version,
                        "n_logs_only_wells": len(group),
                        "n_wells_gate_passed": n_pass,
                        "moran_I_k4": i4,
                        "moran_p_positive_k4": p4,
                        "moran_k4_passed": moran4_pass,
                        "moran_I_k6": i6,
                        "moran_p_positive_k6": p6,
                        "moran_k6_passed": moran6_pass,
                        "variogram_pairs": n_pairs,
                        "variogram_bins_requested": REQUESTED_BINS,
                        "variogram_bins_max_by_pair_count": max_bins,
                        "variogram_bins_actual": len(variogram),
                        "variogram_min_pairs_per_bin": int(variogram["n_pairs"].min()) if not variogram.empty else 0,
                        "variogram_rho": diagnostics["rho"],
                        "variogram_rho_p": diagnostics["rho_p"],
                        "variogram_early_slope": diagnostics["early_slope"],
                        "variogram_late_slope": diagnostics["late_slope"],
                        "variogram_tail_head_ratio": diagnostics["tail_head_ratio"],
                        "variogram_nugget": diagnostics["nugget"],
                        "variogram_sill": diagnostics["sill"],
                        "nugget_to_sill_ratio": diagnostics["nugget_to_sill_ratio"],
                        "nugget_dominated": diagnostics["nugget_dominated"],
                        "variogram_rise_to_sill": diagnostics["rise_to_sill"],
                        "variogram_usable": diagnostics["usable"],
                        "nn_distance_q1_km": q1,
                        "nn_distance_median_km": median,
                        "nn_distance_q3_km": q3,
                        "pass_before_robustness": preliminary_pass,
                    }
                )
                plot_variogram(
                    variogram,
                    target,
                    zone,
                    version,
                    diagnostics,
                    preliminary_pass,
                    output / f"variogram_v2_{target}_{zone}_{version}",
                )

    result = pd.DataFrame(rows)
    statuses: list[dict[str, object]] = []
    for (target, zone), group in result.groupby(["target", "zone_en"], sort=False):
        decisions = dict(zip(group["normalization"], group["pass_before_robustness"]))
        consistent = bool(decisions.get("P") == decisions.get("R"))
        both_pass = bool(decisions.get("P", False) and decisions.get("R", False))
        status = "both_pass" if both_pass else ("both_fail" if consistent else "inconsistent")
        statuses.append(
            {
                "target": target,
                "zone_en": zone,
                "normalization_consistent": consistent,
                "robustness_status": status,
                "interpolation_allowed": both_pass,
            }
        )
    result = result.merge(pd.DataFrame(statuses), on=["target", "zone_en"], validate="many_to_one")
    prior = previous_s29[["target", "zone_en", "interpolation_allowed"]].rename(
        columns={"interpolation_allowed": "previous_interpolation_allowed"}
    )
    prior["previous_interpolation_allowed"] = prior["previous_interpolation_allowed"].map(
        lambda value: str(value).strip().lower() == "true" if not isinstance(value, bool) else value
    )
    result = result.merge(prior, on=["target", "zone_en"], how="left", validate="many_to_one")
    result["conclusion_changed_from_previous"] = (
        result["interpolation_allowed"] != result["previous_interpolation_allowed"]
    )
    result["decision_reason"] = result.apply(decision_reason, axis=1)
    return result.sort_values(["target", "zone_en", "normalization"]), variograms


def decision_reason(row: pd.Series) -> str:
    if row["robustness_status"] == "inconsistent":
        return "P/R conclusions inconsistent; interpolation denied"
    if row["interpolation_allowed"]:
        return "all pre-specified criteria passed under P and R"
    failures: list[str] = []
    if not row["n_wells_gate_passed"]:
        failures.append("n<15")
    if not row["moran_k4_passed"]:
        failures.append("Moran k=4")
    if not row["moran_k6_passed"]:
        failures.append("Moran k=6")
    if not row["variogram_rise_to_sill"]:
        failures.append("variogram rise-to-sill")
    if row["nugget_dominated"]:
        failures.append("nugget/sill>0.75")
    return "failed: " + ", ".join(failures)


def resolve_map_extent(s28: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    lon_min, lon_max = float(s28["lon"].min()), float(s28["lon"].max())
    lat_min, lat_max = float(s28["lat"].min()), float(s28["lat"].max())
    xlim = MAP_XLIM
    ylim = MAP_YLIM
    if lon_min < xlim[0] or lon_max > xlim[1]:
        xlim = (math.floor((lon_min - 0.02) * 10) / 10, math.ceil((lon_max + 0.02) * 10) / 10)
    if lat_min < ylim[0] or lat_max > ylim[1]:
        ylim = (math.floor((lat_min - 0.02) * 10) / 10, math.ceil((lat_max + 0.02) * 10) / 10)
    return xlim, ylim


def map_figure(xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    mean_lat = float(np.mean(ylim))
    width = 4.0
    height = width * (ylim[1] - ylim[0]) / ((xlim[1] - xlim[0]) * math.cos(math.radians(mean_lat)))
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1 / math.cos(math.radians(mean_lat)), adjustable="box")
    ax.axis("off")
    return fig, ax


def save_figure(fig: plt.Figure, path: Path, transparent: bool = False) -> None:
    fig.savefig(path.with_suffix(".svg"), transparent=transparent)
    fig.savefig(path.with_suffix(".pdf"), transparent=transparent)
    fig.savefig(path.with_suffix(".png"), dpi=600, transparent=transparent)
    plt.close(fig)


def plot_overlays(s28: pd.DataFrame, output: Path) -> None:
    configure_style()
    logs = s28[s28["used_in_spatial_test"]].copy()
    xlim, ylim = resolve_map_extent(logs)
    cmap = plt.get_cmap("cividis")
    norm = Normalize(0, 1)
    for target in TARGETS:
        risk = logs[logs["target"].eq(target)]
        fig, ax = map_figure(xlim, ylim)
        ax.scatter(
            risk["lon"],
            risk["lat"],
            s=24,
            marker="o",
            c=risk["composite_risk_v2_consensus"],
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.55,
            clip_on=True,
        )
        save_figure(fig, output / f"overlay_risk_points_v2_{target}", transparent=True)


def plot_colorbar(output: Path) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(3.54, 0.55))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.58, top=0.92)
    cmap = plt.get_cmap("cividis")
    steps = 128
    for index in range(steps):
        left = index / steps
        ax.add_patch(
            Rectangle(
                (left, 0),
                1 / steps,
                1,
                facecolor=cmap((index + 0.5) / steps),
                edgecolor="none",
            )
        )
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Consensus composite risk, median(P, R)", fontsize=8, labelpad=1)
    ax.tick_params(axis="x", labelsize=7, width=0.6, length=2, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure(fig, output / "legend_risk_colorbar_v2", transparent=True)


def image_on_white(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB")


def make_contact_sheet(paths: list[Path], columns: int, thumb_size: tuple[int, int]) -> Image.Image:
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "white")
    for index, path in enumerate(paths):
        image = image_on_white(path)
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        left = (index % columns) * thumb_size[0] + (thumb_size[0] - image.width) // 2
        top = (index // columns) * thumb_size[1] + (thumb_size[1] - image.height) // 2
        sheet.paste(image, (left, top))
    return sheet


def write_qa_previews(output: Path) -> None:
    qa = output / "qa_patch_A_v2"
    qa.mkdir(exist_ok=True)
    variograms = sorted(output.glob("variogram_v2_*.png"))
    overlays = sorted(output.glob("overlay_risk_points_v2_*.png"))
    var_sheet = make_contact_sheet(variograms, columns=3, thumb_size=(720, 540))
    var_sheet.save(qa / "variograms_v2_color.png", dpi=(150, 150))
    var_sheet.convert("L").convert("RGB").save(qa / "variograms_v2_grayscale.png", dpi=(150, 150))
    overlay_sheet = make_contact_sheet(overlays, columns=2, thumb_size=(700, 1000))
    overlay_sheet.save(qa / "overlays_v2_color.png", dpi=(150, 150))
    overlay_sheet.convert("L").convert("RGB").save(qa / "overlays_v2_grayscale.png", dpi=(150, 150))


def bool_word(value: object) -> str:
    return "pass" if bool(value) else "fail"


def decision_word(value: object) -> str:
    return "allowed" if bool(value) else "not allowed"


def yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"


def report_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    output: Path,
    project: Path,
    source_hashes: dict[str, str],
    width_stats: pd.DataFrame,
    s28: pd.DataFrame,
    s29: pd.DataFrame | None,
    stopped_at_width_gate: bool,
) -> None:
    logs = s28[s28["used_in_spatial_test"]]
    counts = (
        logs.groupby(["target", "zone_en"], as_index=False)
        .size()
        .rename(columns={"size": "n_logs_only_wells"})
        .sort_values(["target", "zone_en"])
    )
    count_report = counts.copy()
    count_report["n>=15"] = count_report["n_logs_only_wells"].ge(MIN_WELLS).map(bool_word)
    width_report = width_stats.copy()
    for column in ["width_min", "width_max", "width_mean", "width_sd", "width_cv"]:
        width_report[column] = width_report[column].map(lambda value: f"{value:.6f}")
    width_report["cv_gate_passed"] = width_report["cv_gate_passed"].map(bool_word)

    lines = [
        "# Reliability-map Patch A report",
        "",
        "## Scope and frozen inputs",
        "",
        "This patch corrects the risk variable and repeats the pre-specified spatial tests. "
        "Calibration wells are retained only as reference rows in Table S28 v2; all risk mapping and tests use logs-only wells because these are the deployment population and share one measurement basis.",
        "",
        f"- Project root: `{project}`",
        f"- Previous Table S28 MD5: `{source_hashes['s28']}`",
        f"- Previous Table S29 MD5: `{source_hashes['s29']}`",
        f"- Sample-level risk audit MD5: `{source_hashes['audit']}`",
        f"- Permutations: {PERMUTATIONS}; random seed: {RANDOM_SEED}; Moran neighborhoods: k=4 and k=6.",
        "- No model was retrained and no threshold, kernel, neighborhood, or acceptance criterion was tuned.",
        "",
        "## Logs-only well counts",
        "",
        report_table(count_report),
        "",
        "North remains below n=15 for both targets; calibration wells were not added back to increase sample size.",
        "",
        "## Corrected interval-width variability",
        "",
        "P uses p05-p95 clipped scaling and R uses percentile rank, each computed within all logs-only samples of a target. "
        "The reported interval width is the full conformal width, `2*q90*(1+0.5*dhat)`, using the frozen deployment q90 values from the passed prior reproduction gate.",
        "",
        report_table(width_report),
        "",
        "The quality-risk input is exactly 0.1 for every sample. Its average percentile rank is (n+1)/(2*n) within each target, so the retained 0.10 weight adds a constant offset and contributes nothing to ordering.",
    ]
    if stopped_at_width_gate:
        lines.extend(
            [
                "",
                "## Stop decision",
                "",
                "At least one target-normalization combination has width CV < 0.02. Per Patch A, spatial testing stopped and no interpolation conclusion was attempted.",
            ]
        )
        (output / "report_patch_A.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    assert s29 is not None
    compact = s29[
        [
            "target",
            "zone_en",
            "normalization",
            "n_logs_only_wells",
            "moran_I_k4",
            "moran_p_positive_k4",
            "moran_I_k6",
            "moran_p_positive_k6",
            "variogram_bins_actual",
            "variogram_rho",
            "nugget_to_sill_ratio",
            "pass_before_robustness",
            "robustness_status",
            "interpolation_allowed",
        ]
    ].copy()
    for column in [
        "moran_I_k4",
        "moran_p_positive_k4",
        "moran_I_k6",
        "moran_p_positive_k6",
        "variogram_rho",
        "nugget_to_sill_ratio",
    ]:
        compact[column] = compact[column].map(
            lambda value: "NA" if pd.isna(value) or not np.isfinite(float(value)) else f"{float(value):.4f}"
        )
    for column in ["pass_before_robustness", "interpolation_allowed"]:
        compact[column] = compact[column].map(bool_word)

    comparison = (
        s29.groupby(["target", "zone_en"], as_index=False)
        .agg(
            previous_interpolation_allowed=("previous_interpolation_allowed", "first"),
            corrected_interpolation_allowed=("interpolation_allowed", "first"),
            robustness_status=("robustness_status", "first"),
            conclusion_changed=("conclusion_changed_from_previous", "first"),
        )
        .sort_values(["target", "zone_en"])
    )
    for column in ["previous_interpolation_allowed", "corrected_interpolation_allowed"]:
        comparison[column] = comparison[column].map(decision_word)
    comparison["conclusion_changed"] = comparison["conclusion_changed"].map(yes_no)

    lines.extend(
        [
            "",
            "## Corrected spatial tests",
            "",
            "A row passes before robustness only when n>=15, both Moran tests are positive with one-sided permutation p<0.05, and the variogram satisfies the unchanged rise-to-sill rule while nugget/sill <=0.75. "
            "Final interpolation is allowed only when both P and R pass; disagreement is marked inconsistent and denied.",
            "",
            report_table(compact),
            "",
            "Distance bins were fixed at 12 when the number of pairs allowed at least 10 pairs per bin. Smaller groups report the maximum feasible bin count rather than relaxing the per-bin minimum.",
            "",
            "## Comparison with previous Table S29",
            "",
            report_table(comparison),
            "",
            "The previous six tests mixed calibration and logs-only wells and used a width variable compressed by min-max distance scaling. Their numerical outcomes are retained only for provenance; the corrected decisions above replace them.",
            "",
            "## Final interpolation conclusion",
            "",
        ]
    )
    group_decisions = s29.groupby(["target", "zone_en"], as_index=False)["interpolation_allowed"].first()
    allowed = group_decisions[group_decisions["interpolation_allowed"]]
    denied = group_decisions[~group_decisions["interpolation_allowed"]]
    if allowed.empty:
        lines.append(
            "No target-zone group is eligible for a continuous risk surface after correcting the deployment variable. "
            "This negative conclusion is based exclusively on logs-only wells and is supported by the n, Moran p-values, variogram shape, and nugget/sill ratios reported above."
        )
    else:
        lines.append(
            "Interpolation is eligible only for: "
            + ", ".join(f"{row.target}/{row.zone_en}" for _, row in allowed.iterrows())
            + "."
        )
    if not denied.empty:
        lines.append(
            "Interpolation is not eligible for: "
            + ", ".join(f"{row.target}/{row.zone_en}" for _, row in denied.iterrows())
            + "."
        )
    inconsistent = s29[s29["robustness_status"].eq("inconsistent")][["target", "zone_en"]].drop_duplicates()
    if not inconsistent.empty:
        lines.append(
            "P/R sensitivity caused an inconsistent decision for: "
            + ", ".join(f"{row.target}/{row.zone_en}" for _, row in inconsistent.iterrows())
            + "; these groups are treated as failures by design."
        )
    lines.extend(
        [
            "",
            "## Figure semantics",
            "",
            "The v2 point overlays color only logs-only wells. Their color is the per-well median of the P and R composite risks and is shown only as a point diagnostic; it is not an interpolated surface. Variograms use zone color plus zone marker, with P shown by a solid filled trace and R by a dashed hollow trace for redundant encoding.",
        ]
    )
    (output / "report_patch_A.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_tables(output: Path, publish_base: Path) -> None:
    destinations = [publish_base / "supplementary_tables", publish_base / "code" / "data_derived"]
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for name in [
            "Table_S28_v2_well_level_conformal_risk.csv",
            "Table_S29_v2_spatial_autocorrelation_test.csv",
        ]:
            source = output / name
            if source.exists():
                shutil.copy2(source, destination / name)


def main() -> None:
    args = parse_args()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    prior_s28_path = output / "Table_S28_well_level_conformal_risk.csv"
    prior_s29_path = output / "Table_S29_spatial_autocorrelation_test.csv"
    audit_path = output / "risk_component_audit.csv"
    previous_s28 = pd.read_csv(prior_s28_path, encoding="utf-8-sig")
    previous_s29 = pd.read_csv(prior_s29_path, encoding="utf-8-sig")
    audit = pd.read_csv(audit_path, encoding="utf-8-sig")
    source_hashes = {"s28": md5(prior_s28_path), "s29": md5(prior_s29_path), "audit": md5(audit_path)}

    s28, ranked = build_s28_v2(previous_s28, audit)
    s28_path = output / "Table_S28_v2_well_level_conformal_risk.csv"
    s28.to_csv(s28_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    ranked.to_csv(output / "risk_component_audit_v2.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    widths = width_summary(s28)
    widths.to_csv(output / "width_variability_v2.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    stopped = not bool(widths["cv_gate_passed"].all())
    if stopped:
        write_report(output, args.project_root, source_hashes, widths, s28, None, True)
        if args.publish_base:
            publish_tables(output, args.publish_base)
        raise RuntimeError("Patch A width CV gate failed; spatial tests were not run")

    s29, _ = build_spatial_tests(s28, previous_s29, output)
    s29_path = output / "Table_S29_v2_spatial_autocorrelation_test.csv"
    s29.to_csv(s29_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    plot_overlays(s28, output)
    plot_colorbar(output)
    write_qa_previews(output)
    write_report(output, args.project_root, source_hashes, widths, s28, s29, False)
    if args.publish_base:
        publish_tables(output, args.publish_base)
    print(widths.to_string(index=False))
    print(s29.to_string(index=False))
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
