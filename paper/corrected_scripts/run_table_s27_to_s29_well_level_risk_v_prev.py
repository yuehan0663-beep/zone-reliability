from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from catboost import CatBoostRegressor
from scipy.spatial import ConvexHull, distance_matrix
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


LOG_COLS = ["GR", "AC", "DEN", "CNL", "RT"]
NUM_COLS = [
    "GR",
    "AC",
    "DEN",
    "CNL",
    "RT",
    "pressure_coeff",
    "formation_top_md",
    "formation_bottom_md",
    "max_burial_top",
]
CAT_COLS = ["layer", "lithology", "diagenesis_category"]
TARGETS = {
    "TOC": {
        "dataset": "TOC_high_medium",
        "column": "toc_pct",
        "anchor": 0.926,
    },
    "total_gas": {
        "dataset": "total_gas_high_medium",
        "column": "total_gas_m3_t",
        "anchor": 0.949,
    },
}
ZONE_EN = {"北": "North", "中": "Central", "南": "South"}
ZONE_COLORS = {"North": "#D55E00", "Central": "#0072B2", "South": "#009E73"}
ZONE_MARKERS = {"North": "o", "Central": "s", "South": "^"}
PUBLIC_ID_ALIASES = {"L1": "Le1"}
MAP_XLIM = (106.70, 107.50)
MAP_YLIM = (37.40, 39.30)
RISK_WEIGHTS = {
    "interval_width": 0.55,
    "domain_knn": 0.25,
    "missing_risk": 0.10,
    "quality_risk": 0.10,
}


@dataclass
class GateResult:
    target: str
    models: list[str]
    fold_mean_coverage: float
    fold_mean_width: float
    pooled_coverage: float
    pooled_width: float
    anchor: float
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def model_pool() -> dict[str, object]:
    return {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=180,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "RF": RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGBDT": HistGradientBoostingRegressor(
            random_state=42,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=0.1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
            verbosity=0,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=180,
            depth=3,
            learning_rate=0.04,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
    )


def make_pipe(model: object, num: list[str], cat: list[str]) -> Pipeline:
    return Pipeline([("pre", preprocessor(num, cat)), ("model", clone(model))])


def choose_models(summary: pd.DataFrame, dataset: str) -> list[str]:
    subset = summary[
        summary.cv_scheme.isin(["LOWO", "LOZO"])
        & (summary.dataset == dataset)
        & ~summary.model.isin(["train_mean", "zone_mean"])
    ]
    ranked = (
        subset.groupby("model", as_index=False)
        .agg(MAE=("MAE_w", "mean"), S=("Spearman_mean", "mean"))
        .sort_values(["MAE", "S"], ascending=[True, False])
    )
    return ranked.head(3).model.tolist()


def build_public_id_map(raw: pd.DataFrame, map_table: pd.DataFrame, s25: pd.DataFrame) -> dict[str, str]:
    real_col, code_col = map_table.columns[0], map_table.columns[2]
    mapping = dict(zip(map_table[real_col].astype(str), map_table[code_col].astype(str)))
    s25_ids = set(s25.well_id.astype(str))
    for real_name, public_id in list(mapping.items()):
        if public_id not in s25_ids and public_id in PUBLIC_ID_ALIASES:
            alias = PUBLIC_ID_ALIASES[public_id]
            if alias in s25_ids:
                mapping[real_name] = alias
    unresolved = sorted(set(raw.well.astype(str)) - set(mapping))
    if unresolved:
        raise ValueError(f"Unmapped workbook wells: {unresolved}")
    mapped_ids = set(raw.well.astype(str).map(mapping))
    if mapped_ids != s25_ids:
        raise ValueError(
            f"Public ID reconciliation failed: workbook-only={sorted(mapped_ids - s25_ids)}, "
            f"S25-only={sorted(s25_ids - mapped_ids)}"
        )
    return mapping


def prepare_raw(raw: pd.DataFrame, public_map: dict[str, str]) -> pd.DataFrame:
    result = raw.copy()
    result["well"] = result.well.astype(str)
    result["well_id"] = result.well.map(public_map)
    result["zone_en"] = result.zone.astype(str).map(ZONE_EN)
    if result.zone_en.isna().any():
        raise ValueError("At least one workbook zone could not be translated to North/Central/South.")
    result["model_ready"] = result[LOG_COLS].notna().sum(axis=1) >= 3
    return result


def build_s27(s25: pd.DataFrame) -> pd.DataFrame:
    table = s25.rename(columns={"zone": "zone_en"})[
        [
            "target",
            "well_id",
            "zone_en",
            "lon",
            "lat",
            "is_calibration",
            "n_core_points",
            "feature_dist",
        ]
    ].copy()
    table["feature_dist_pct"] = np.nan
    mask = table.is_calibration.eq(0)
    table.loc[mask, "feature_dist_pct"] = table.loc[mask].groupby("target")[
        "feature_dist"
    ].rank(method="average", pct=True)
    if table.loc[table.is_calibration.eq(1), "feature_dist"].notna().any():
        raise ValueError("Calibration-well feature_dist must remain missing.")
    if table.loc[mask, "feature_dist"].isna().any():
        raise ValueError("At least one logs-only well lacks feature_dist in S25.")
    return table.sort_values(["target", "zone_en", "is_calibration", "well_id"], ascending=[True, True, False, True])


def model_ready_target(raw: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return raw[raw.model_ready & raw[target_col].notna()].copy().reset_index(drop=True)


def normalized_distance(reference: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = (
        NearestNeighbors(n_neighbors=min(5, len(reference)))
        .fit(reference)
        .kneighbors(query, return_distance=True)[0]
        .mean(axis=1)
    )
    normalized = (distances - distances.min()) / (np.ptp(distances) + 1e-9)
    return distances, normalized


def lowo_gate_rows(
    frame: pd.DataFrame,
    target_name: str,
    target_col: str,
    model_names: list[str],
    pool: dict[str, object],
) -> tuple[pd.DataFrame, GateResult]:
    num = [column for column in NUM_COLS if column in frame.columns]
    cat = [column for column in CAT_COLS if column in frame.columns]
    x = frame[num + cat]
    y = frame[target_col].to_numpy(float)
    rows: list[pd.DataFrame] = []
    fold_coverages: list[float] = []
    fold_widths: list[float] = []
    for model_name in model_names:
        for held_out_well in sorted(frame.well.unique()):
            test = np.where(frame.well.to_numpy() == held_out_well)[0]
            outer_train = np.where(frame.well.to_numpy() != held_out_well)[0]
            training_wells = np.array(sorted(frame.iloc[outer_train].well.unique()))
            rng = np.random.default_rng(42)
            calibration_wells = set(
                rng.choice(
                    training_wells,
                    size=max(1, int(len(training_wells) * 0.25)),
                    replace=False,
                )
            )
            calibration = np.array(
                [index for index in outer_train if frame.iloc[index].well in calibration_wells]
            )
            train = np.array(
                [index for index in outer_train if frame.iloc[index].well not in calibration_wells]
            )
            pipe = make_pipe(pool[model_name], num, cat)
            pipe.fit(x.iloc[train], y[train])
            cal_prediction = np.asarray(pipe.predict(x.iloc[calibration]), float)
            q90 = float(np.quantile(np.abs(y[calibration] - cal_prediction), 0.90))
            prediction = np.asarray(pipe.predict(x.iloc[test]), float)
            train_transformed = pipe.named_steps["pre"].transform(x.iloc[train])
            test_transformed = pipe.named_steps["pre"].transform(x.iloc[test])
            distance, distance_norm = normalized_distance(train_transformed, test_transformed)
            half_width = q90 * (1 + 0.5 * distance_norm)
            covered = (y[test] >= prediction - half_width) & (y[test] <= prediction + half_width)
            fold_coverages.append(float(covered.mean()))
            fold_widths.append(float(np.mean(2 * half_width)))
            part = frame.iloc[test][["sample_id", "well_id", "zone_en"]].copy()
            part["target"] = target_name
            part["model"] = model_name
            part["source"] = "calibration_LOWO"
            part["interval_width"] = 2 * half_width
            part["domain_knn"] = distance
            part["missing_risk"] = frame.iloc[test][LOG_COLS].isna().sum(axis=1).to_numpy() / len(LOG_COLS)
            part["quality_risk"] = 0.1
            part["covered"] = covered
            part["fold"] = str(frame.iloc[test].well_id.iloc[0])
            part["q90"] = q90
            rows.append(part)
    samples = pd.concat(rows, ignore_index=True)
    anchor = TARGETS[target_name]["anchor"]
    fold_mean_coverage = float(np.mean(fold_coverages))
    result = GateResult(
        target=target_name,
        models=model_names,
        fold_mean_coverage=fold_mean_coverage,
        fold_mean_width=float(np.mean(fold_widths)),
        pooled_coverage=float(samples.covered.mean()),
        pooled_width=float(samples.interval_width.mean()),
        anchor=anchor,
        passed=abs(fold_mean_coverage - anchor) <= 0.005,
    )
    return samples, result


def deployment_rows(
    raw: pd.DataFrame,
    labeled: pd.DataFrame,
    s25_target: pd.DataFrame,
    target_name: str,
    target_col: str,
    model_names: list[str],
    pool: dict[str, object],
) -> pd.DataFrame:
    logs_only_ids = set(s25_target.loc[s25_target.is_calibration.eq(0), "well_id"].astype(str))
    query = raw[raw.model_ready & raw.well_id.isin(logs_only_ids)].copy().reset_index(drop=True)
    missing_wells = sorted(logs_only_ids - set(query.well_id))
    if missing_wells:
        raise ValueError(f"Logs-only wells lack model-ready feature rows: {missing_wells}")
    num = [column for column in NUM_COLS if column in labeled.columns]
    cat = [column for column in CAT_COLS if column in labeled.columns]
    x_labeled = labeled[num + cat]
    y = labeled[target_col].to_numpy(float)
    x_query = query[num + cat]
    parts: list[pd.DataFrame] = []
    for model_name in model_names:
        rng = np.random.default_rng(42)
        wells = np.array(sorted(labeled.well.unique()))
        calibration_wells = set(
            rng.choice(wells, size=max(1, int(len(wells) * 0.25)), replace=False)
        )
        cal = np.where(labeled.well.isin(calibration_wells).to_numpy())[0]
        train = np.where(~labeled.well.isin(calibration_wells).to_numpy())[0]
        calibration_pipe = make_pipe(pool[model_name], num, cat)
        calibration_pipe.fit(x_labeled.iloc[train], y[train])
        q90 = float(
            np.quantile(
                np.abs(y[cal] - calibration_pipe.predict(x_labeled.iloc[cal])),
                0.90,
            )
        )
        deployment_pipe = make_pipe(pool[model_name], num, cat)
        deployment_pipe.fit(x_labeled, y)
        deployment_prediction = np.asarray(deployment_pipe.predict(x_query), float)
        reference = deployment_pipe.named_steps["pre"].transform(x_labeled)
        transformed_query = deployment_pipe.named_steps["pre"].transform(x_query)
        distance, distance_norm = normalized_distance(reference, transformed_query)
        half_width = q90 * (1 + 0.5 * distance_norm)
        part = query[["sample_id", "well_id", "zone_en"]].copy()
        part["target"] = target_name
        part["model"] = model_name
        part["source"] = "logs_only_deployment"
        part["interval_width"] = 2 * half_width
        part["domain_knn"] = distance
        part["missing_risk"] = query[LOG_COLS].isna().sum(axis=1).to_numpy() / len(LOG_COLS)
        part["quality_risk"] = 0.1
        part["covered"] = np.nan
        part["fold"] = "deployment"
        part["q90"] = q90
        part["prediction"] = deployment_prediction
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def build_s28(
    s25: pd.DataFrame,
    risk_samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked_parts: list[pd.DataFrame] = []
    for (_, _), group in risk_samples.groupby(["target", "model"], sort=False):
        part = group.copy()
        for component, weight in RISK_WEIGHTS.items():
            part[f"{component}_pct_global"] = part[component].rank(method="average", pct=True)
        part["composite_risk_global"] = sum(
            part[f"{component}_pct_global"] * weight
            for component, weight in RISK_WEIGHTS.items()
        )
        part["composite_risk_fold_local"] = part.groupby("fold")[
            "interval_width"
        ].rank(method="average", pct=True) * 0.55
        part["composite_risk_fold_local"] += part.groupby("fold")["domain_knn"].rank(
            method="average", pct=True
        ) * 0.25
        part["composite_risk_fold_local"] += part.groupby("fold")["missing_risk"].rank(
            method="average", pct=True
        ) * 0.10
        part["composite_risk_fold_local"] += part.groupby("fold")["quality_risk"].rank(
            method="average", pct=True
        ) * 0.10
        ranked_parts.append(part)
    ranked = pd.concat(ranked_parts, ignore_index=True)
    per_model = (
        ranked.groupby(["target", "model", "well_id", "zone_en"], as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            median_interval_width=("interval_width", "median"),
            median_domain_knn=("domain_knn", "median"),
            median_missing_risk=("missing_risk", "median"),
            composite_risk_global_median=("composite_risk_global", "median"),
            composite_risk_global_iqr=("composite_risk_global", iqr),
        )
    )
    well_level = (
        per_model.groupby(["target", "well_id", "zone_en"], as_index=False)
        .agg(
            n_samples=("n_samples", "median"),
            median_interval_width=("median_interval_width", "median"),
            median_domain_knn=("median_domain_knn", "median"),
            median_missing_risk=("median_missing_risk", "median"),
            composite_risk_global_median=("composite_risk_global_median", "median"),
            composite_risk_global_iqr=("composite_risk_global_iqr", "median"),
        )
    )
    lookup = s25.rename(columns={"zone": "zone_en"})[
        ["target", "well_id", "zone_en", "lon", "lat", "is_calibration"]
    ]
    result = lookup.merge(well_level, on=["target", "well_id", "zone_en"], how="left", validate="one_to_one")
    if result.composite_risk_global_median.isna().any():
        missing = result.loc[result.composite_risk_global_median.isna(), ["target", "well_id"]]
        raise ValueError(f"Missing well-level risk rows: {missing.to_dict('records')}")
    result["n_samples"] = result.n_samples.astype(int)
    result["rejected_at_20pct"] = result.groupby("target")[
        "composite_risk_global_median"
    ].transform(lambda values: values.rank(method="average", pct=True) >= 0.80)
    result = result[
        [
            "target",
            "well_id",
            "zone_en",
            "lon",
            "lat",
            "is_calibration",
            "n_samples",
            "median_interval_width",
            "median_domain_knn",
            "median_missing_risk",
            "composite_risk_global_median",
            "composite_risk_global_iqr",
            "rejected_at_20pct",
        ]
    ].sort_values(["target", "zone_en", "well_id"])
    return result, ranked


def lonlat_xy_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    mean_lat = np.deg2rad(np.mean(lat))
    x = lon * 111.320 * np.cos(mean_lat)
    y = lat * 110.574
    return np.column_stack([x, y])


def moran_knn(
    values: np.ndarray,
    xy: np.ndarray,
    k: int,
    permutations: int = 999,
    seed: int = 42,
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
    observed = float((n / weights.sum()) * np.sum(weights * np.outer(centered, centered)) / denominator)
    rng = np.random.default_rng(seed)
    permuted = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = rng.permutation(values)
        z = shuffled - shuffled.mean()
        permuted[index] = (n / weights.sum()) * np.sum(weights * np.outer(z, z)) / np.sum(z**2)
    p_positive = float((1 + np.sum(permuted >= observed)) / (permutations + 1))
    return observed, p_positive


def empirical_variogram(values: np.ndarray, xy: np.ndarray) -> pd.DataFrame:
    pairs: list[tuple[float, float]] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            distance = float(np.linalg.norm(xy[i] - xy[j]))
            semivariance = float(0.5 * (values[i] - values[j]) ** 2)
            pairs.append((distance, semivariance))
    pairs.sort(key=lambda item: item[0])
    pair_frame = pd.DataFrame(pairs, columns=["distance_km", "semivariance"])
    bins = min(8, len(pair_frame) // 10)
    if bins < 3:
        return pd.DataFrame(columns=["bin", "distance_km", "semivariance", "n_pairs"])
    chunks = [pair_frame.iloc[index] for index in np.array_split(np.arange(len(pair_frame)), bins)]
    rows = []
    for index, chunk in enumerate(chunks, 1):
        rows.append(
            {
                "bin": index,
                "distance_km": float(chunk.distance_km.mean()),
                "semivariance": float(chunk.semivariance.mean()),
                "n_pairs": int(len(chunk)),
            }
        )
    return pd.DataFrame(rows)


def variogram_shape(variogram: pd.DataFrame) -> tuple[bool, dict[str, float]]:
    if len(variogram) < 4:
        return False, {
            "rho": np.nan,
            "rho_p": np.nan,
            "early_slope": np.nan,
            "late_slope": np.nan,
            "tail_head_ratio": np.nan,
        }
    x = variogram.distance_km.to_numpy(float)
    y = variogram.semivariance.to_numpy(float)
    rho, rho_p = spearmanr(x, y)
    midpoint = len(variogram) // 2
    early_slope = float(np.polyfit(x[: midpoint + 1], y[: midpoint + 1], 1)[0])
    late_slope = float(np.polyfit(x[midpoint:], y[midpoint:], 1)[0])
    head = float(np.mean(y[:2]))
    tail = float(np.mean(y[-2:]))
    ratio = tail / head if head > 0 else np.inf
    passed = bool(
        rho > 0.5
        and early_slope > 0
        and ratio >= 1.10
        and abs(late_slope) <= 0.5 * abs(early_slope)
    )
    return passed, {
        "rho": float(rho),
        "rho_p": float(rho_p),
        "early_slope": early_slope,
        "late_slope": late_slope,
        "tail_head_ratio": ratio,
    }


def nearest_neighbor_stats(xy: np.ndarray) -> tuple[float, float, float]:
    distances = distance_matrix(xy, xy)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return tuple(float(value) for value in np.quantile(nearest, [0.25, 0.50, 0.75]))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def plot_variogram(
    variogram: pd.DataFrame,
    target: str,
    zone: str,
    passed: bool,
    output: Path,
) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(3.54, 2.60), constrained_layout=True)
    color = ZONE_COLORS[zone]
    marker = ZONE_MARKERS[zone]
    ax.plot(
        variogram.distance_km,
        variogram.semivariance,
        color=color,
        marker=marker,
        markersize=4,
        linewidth=1.1,
        markeredgecolor="black",
        markeredgewidth=0.35,
    )
    ax.set_xlabel("Pair distance (km)")
    ax.set_ylabel("Semivariance")
    ax.text(
        0.02,
        0.98,
        f"{target.replace('_', ' ')} | {zone}\ncriterion: {'pass' if passed else 'fail'}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output.with_suffix(".svg"), transparent=False)
    fig.savefig(output.with_suffix(".png"), dpi=600, transparent=False)
    plt.close(fig)


def build_spatial_tests(s28: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    rows = []
    variograms: dict[tuple[str, str], pd.DataFrame] = {}
    for (target, zone), group in s28.groupby(["target", "zone_en"], sort=False):
        values = group.composite_risk_global_median.to_numpy(float)
        xy = lonlat_xy_km(group.lon.to_numpy(float), group.lat.to_numpy(float))
        i4, p4 = moran_knn(values, xy, 4)
        i6, p6 = moran_knn(values, xy, 6)
        variogram = empirical_variogram(values, xy)
        shape_pass, shape = variogram_shape(variogram)
        q1, median, q3 = nearest_neighbor_stats(xy)
        passed = bool(len(group) >= 15 and i4 > 0 and i6 > 0 and p4 < 0.05 and p6 < 0.05 and shape_pass)
        variograms[(target, zone)] = variogram
        rows.append(
            {
                "target": target,
                "zone_en": zone,
                "n_wells": len(group),
                "moran_I_k4": i4,
                "moran_p_positive_k4": p4,
                "moran_I_k6": i6,
                "moran_p_positive_k6": p6,
                "variogram_bins": len(variogram),
                "variogram_rho": shape["rho"],
                "variogram_rho_p": shape["rho_p"],
                "variogram_early_slope": shape["early_slope"],
                "variogram_late_slope": shape["late_slope"],
                "variogram_tail_head_ratio": shape["tail_head_ratio"],
                "variogram_rise_to_sill": shape_pass,
                "nn_distance_q1_km": q1,
                "nn_distance_median_km": median,
                "nn_distance_q3_km": q3,
                "interpolation_allowed": passed,
                "decision_reason": "all criteria passed" if passed else "one or more pre-specified criteria failed",
            }
        )
        plot_variogram(
            variogram,
            target,
            zone,
            passed,
            output / f"variogram_{target}_{zone}",
        )
    return pd.DataFrame(rows).sort_values(["target", "zone_en"]), variograms


def resolve_map_extent(s28: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    lon_min, lon_max = float(s28.lon.min()), float(s28.lon.max())
    lat_min, lat_max = float(s28.lat.min()), float(s28.lat.max())
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


def save_overlay(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
    fig.savefig(path.with_suffix(".svg"), transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def core_size(points: pd.Series) -> np.ndarray:
    positive = points[points > 0]
    if positive.empty:
        return np.full(len(points), 20.0)
    q1, q2 = positive.quantile([1 / 3, 2 / 3])
    return np.select([points <= q1, points <= q2], [26.0, 48.0], default=76.0)


def plot_overlays(s25: pd.DataFrame, s28: pd.DataFrame, output: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    xlim, ylim = resolve_map_extent(s28)
    norm = Normalize(0, 1)
    cmap = plt.get_cmap("cividis")
    for target in TARGETS:
        geometry = s25[s25.target == target].copy()
        fig, ax = map_figure(xlim, ylim)
        calibration = geometry[geometry.is_calibration.eq(1)]
        logs = geometry[geometry.is_calibration.eq(0)]
        ax.scatter(
            calibration.lon,
            calibration.lat,
            s=core_size(calibration.n_core_points),
            marker="o",
            facecolor="black",
            edgecolor="black",
            linewidth=0.5,
            clip_on=True,
        )
        ax.scatter(
            logs.lon,
            logs.lat,
            s=18,
            marker="o",
            facecolor="none",
            edgecolor="black",
            linewidth=0.7,
            clip_on=True,
        )
        save_overlay(fig, output / f"overlay_wells_{target}")

        risk = s28[s28.target == target].copy()
        fig, ax = map_figure(xlim, ylim)
        calibration = risk[risk.is_calibration.eq(1)]
        logs = risk[risk.is_calibration.eq(0)]
        ax.scatter(
            calibration.lon,
            calibration.lat,
            s=core_size(
                geometry.set_index("well_id").loc[calibration.well_id, "n_core_points"].reset_index(drop=True)
            ),
            marker="o",
            c=calibration.composite_risk_global_median,
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.55,
            clip_on=True,
        )
        ax.scatter(
            logs.lon,
            logs.lat,
            s=22,
            marker="o",
            facecolor="none",
            edgecolor=cmap(norm(logs.composite_risk_global_median.to_numpy())),
            linewidth=1.1,
            clip_on=True,
        )
        save_overlay(fig, output / f"overlay_risk_points_{target}")
    return xlim, ylim


def plot_registration_check(
    s25: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    output: Path,
) -> None:
    configure_style()
    fig, ax = map_figure(xlim, ylim)
    corners = [
        (xlim[0], ylim[0]),
        (xlim[0], ylim[1]),
        (xlim[1], ylim[0]),
        (xlim[1], ylim[1]),
    ]
    dx = (xlim[1] - xlim[0]) * 0.018
    dy = (ylim[1] - ylim[0]) * 0.018
    for lon, lat in corners:
        x_inner = lon + dx if lon == xlim[0] else lon - dx
        y_inner = lat + dy if lat == ylim[0] else lat - dy
        ax.plot([lon, x_inner], [lat, lat], color="black", linewidth=1.0, clip_on=True)
        ax.plot([lon, lon], [lat, y_inner], color="black", linewidth=1.0, clip_on=True)
        ax.annotate(
            f"{lon:.2f}, {lat:.2f}",
            (lon, lat),
            xytext=(10 if lon == xlim[0] else -10, 10 if lat == ylim[0] else -10),
            textcoords="offset points",
            ha="left" if lon == xlim[0] else "right",
            va="bottom" if lat == ylim[0] else "top",
            fontsize=7,
        )
    wells = s25.drop_duplicates("well_id").set_index("well_id")
    for well_id in ["QT9", "R14", "L53"]:
        if well_id not in wells.index:
            raise ValueError(f"Registration-check well not found in S25: {well_id}")
        row = wells.loc[well_id]
        ax.scatter(row.lon, row.lat, s=30, facecolor="white", edgecolor="black", linewidth=0.8)
        ax.annotate(well_id, (row.lon, row.lat), xytext=(4, 4), textcoords="offset points", fontsize=7)
    fig.savefig(output / "overlay_registration_check.png", dpi=600, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


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
    ax.set_xlabel("Global composite risk", fontsize=8, labelpad=1)
    ax.tick_params(axis="x", labelsize=7, width=0.6, length=2, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output / "legend_risk_colorbar.svg", transparent=True)
    plt.close(fig)


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
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    variograms = sorted(output.glob("variogram_*.png"))
    variogram_sheet = make_contact_sheet(variograms, columns=3, thumb_size=(560, 430))
    variogram_sheet.save(qa / "variograms_contact_sheet.png", dpi=(300, 300))
    variogram_sheet.convert("L").save(qa / "variograms_contact_sheet_grayscale.png", dpi=(300, 300))
    risks = sorted(output.glob("overlay_risk_points_*.png"))
    risk_sheet = make_contact_sheet(risks, columns=2, thumb_size=(420, 1060))
    risk_sheet.save(qa / "risk_points_contact_sheet.png", dpi=(300, 300))
    risk_sheet.convert("L").save(qa / "risk_points_contact_sheet_grayscale.png", dpi=(300, 300))


def build_well_anon_map(public_map: dict[str, str], anon: pd.DataFrame) -> pd.DataFrame:
    public = pd.DataFrame({"well_real": list(public_map), "well_id": list(public_map.values())})
    result = public.merge(anon, on="well_real", how="inner")[["well_id", "well_anon"]]
    return result.drop_duplicates().sort_values("well_anon")


def write_report(
    output: Path,
    workbook: Path,
    workbook_md5: str,
    sheet_details: list[tuple[str, tuple[int, int], list[str]]],
    gates: list[GateResult],
    s27: pd.DataFrame,
    s28: pd.DataFrame,
    s29: pd.DataFrame,
    ranked: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    anon_map: pd.DataFrame,
) -> None:
    lines = [
        "# High-resolution reliability map-input calculations",
        "",
        "## 1. Data locations",
        "",
        f"- Model-ready workbook: `{workbook}`",
        f"- MD5：`{workbook_md5}`",
    ]
    for name, shape, columns in sheet_details:
        lines.append(f"- Sheet `{name}`：{shape[0]} rows x {shape[1]} columns; fields: {', '.join(columns)}")
    lines += [
        "- Approved mapping converts wells to public romanized IDs. The L1/Le1 alias in S25 is documented explicitly; source files are unchanged.",
        f"- Layer 1 output: {len(s27)} rows, each target has {len(s27) // 2} wells. Calibration-well feature_dist remains blank.",
        "",
        "## 2. Layer 2 reproduction gates",
        "",
        "Coverage anchors use the simple mean over model/outer-well-fold coverages. Pooled sample coverage is reported separately; these conventions must not be interchanged.",
        "",
        "| target | models | fold-mean coverage | anchor | delta | pooled-sample coverage | fold-mean width | gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for gate in gates:
        lines.append(
            f"| {gate.target} | {', '.join(gate.models)} | {gate.fold_mean_coverage:.6f} | "
            f"{gate.anchor:.3f} | {gate.fold_mean_coverage - gate.anchor:+.6f} | "
            f"{gate.pooled_coverage:.6f} | {gate.fold_mean_width:.6f} | {'PASS' if gate.passed else 'FAIL'} |"
        )
    lines += [
        "",
        "Layer 2 logs-only models refit all labelled samples. q90 uses a 25% calibration-well split with seed 42. Strict LOWO held-out predictions prevent underestimated calibration-well risk. Compute each of the three canonical models, then take the well-level median.",
        "",
        "## 3. Global risk scale",
        "",
        "For each target/model, rank all calibration-well held-out and logs-only samples together. Fixed weights are width .55, domain kNN .25, missing risk .10 and quality risk .10. Constant quality risk .1 preserves the protocol but carries no ordering information.",
    ]
    audit = ranked.copy()
    audit["rank_delta"] = audit.composite_risk_global - audit.composite_risk_fold_local
    example = audit.loc[audit.rank_delta.abs().idxmax()]
    lines.append(
        f"- Example difference: {example.target}/{example.model}/{example.well_id} fold-local composite risk = "
        f"{example.composite_risk_fold_local:.4f}, global scale = {example.composite_risk_global:.4f}, "
        f"difference = {example.rank_delta:+.4f}."
    )
    lines += [
        f"- S28 rows: {len(s28)}, per target {s28.groupby('target').size().to_dict()}.",
        "",
        "## 4. Layer 3 spatial autocorrelation",
        "",
        "Moran's I uses row-normalized directed kNN weights for k=4 and 6, with 999 one-sided positive-correlation permutations (seed 42). Equal-frequency distance bins contain at least ten pairs. Predefined rise-to-sill criteria: Spearman rho>.5, positive initial slope, final-two/initial-two semivariance ratio >=1.10, and absolute late slope <=.5 times the initial slope.",
        "",
        "| target | zone | n | I4 | p4 | I6 | p6 | variogram | NN median km | interpolate |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for _, row in s29.iterrows():
        lines.append(
            f"| {row.target} | {row.zone_en} | {int(row.n_wells)} | {row.moran_I_k4:.4f} | "
            f"{row.moran_p_positive_k4:.3f} | {row.moran_I_k6:.4f} | "
            f"{row.moran_p_positive_k6:.3f} | {'rise-to-sill' if row.variogram_rise_to_sill else 'not supported'} | "
            f"{row.nn_distance_median_km:.2f} | {'YES' if row.interpolation_allowed else 'NO'} |"
        )
    allowed = s29[s29.interpolation_allowed]
    denied = s29[~s29.interpolation_allowed]
    lines += ["", "## 5. Conclusions", ""]
    if allowed.empty:
        lines.append(
            "None of the six target/zone combinations passes all predefined admission criteria, so no continuous risk surfaces are generated. k, p thresholds, bins and variogram criteria are unchanged. Reliability depends on feature-space position and cannot be inferred from geographic proximity alone."
        )
    else:
        lines.append(
            "Combinations admitted for interpolation: "
            + ", ".join(f"{row.target}/{row.zone_en}" for _, row in allowed.iterrows())
            + ". Other combinations are not interpolated."
        )
    if not denied.empty:
        reasons = []
        for _, row in denied.iterrows():
            failed = []
            if not (row.moran_I_k4 > 0 and row.moran_p_positive_k4 < 0.05):
                failed.append("Moran k=4")
            if not (row.moran_I_k6 > 0 and row.moran_p_positive_k6 < 0.05):
                failed.append("Moran k=6")
            if not row.variogram_rise_to_sill:
                failed.append("variogram")
            if row.n_wells < 15:
                failed.append("n<15")
            reasons.append(f"{row.target}/{row.zone_en}: {', '.join(failed)}")
        lines.append("- Failed criteria: " + "; ".join(reasons) + ".")
    lines += [
        "",
        "## 6. Overlay layers",
        "",
        f"- Coordinate range: longitude [{xlim[0]:.2f}, {xlim[1]:.2f}], lat [{ylim[0]:.2f}, {ylim[1]:.2f}].",
        "- Overlays use actual longitude/latitude. Register them against the author-prepared Fig. 11 coordinate axes.",
        "- Well layer: calibration wells are filled black circles in three core-count sizes; logs-only wells are open black circles.",
        "- Risk-point layer: cividis-filled calibration wells; open logs-only markers with cividis outlines encode risk.",
        "- Do not manually add risk surfaces in artwork for combinations lacking an admitted surface.",
        "",
        "## 7. Well-ID mapping",
        "",
        "| public well_id | anonymous ID |",
        "|---|---|",
    ]
    for _, row in anon_map.iterrows():
        lines.append(f"| {row.well_id} | {row.well_anon} |")
    lines += [
        "",
        "## 8. File list",
        "",
        "- `Table_S27_well_level_map_inputs.csv`",
        "- `Table_S28_well_level_conformal_risk.csv`",
        "- `Table_S29_spatial_autocorrelation_test.csv`",
        "- `variogram_<target>_<zone>.svg/png` (6 groups)",
        "- `overlay_wells_<target>.svg/png`",
        "- `overlay_risk_points_<target>.svg/png`",
        "- `overlay_registration_check.png`",
        "- `legend_risk_colorbar.svg`",
        "- qa/variograms_contact_sheet.png and grayscale version",
        "- qa/risk_points_contact_sheet.png and grayscale version",
        "",
        "Outputs are additional files; existing CSVs, scripts, author Fig. 11 artwork and manuscript TeX are unchanged.",
    ]
    (output / "report_reliability_map_upgrade.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_tables(output: Path, publish_base: Path) -> None:
    targets = [
        publish_base / "supplementary_tables",
        publish_base / "code" / "data_derived",
    ]
    for directory in targets:
        directory.mkdir(parents=True, exist_ok=True)
        for name in [
            "Table_S27_well_level_map_inputs.csv",
            "Table_S28_well_level_conformal_risk.csv",
            "Table_S29_spatial_autocorrelation_test.csv",
        ]:
            shutil.copy2(output / name, directory / name)


def main() -> None:
    args = parse_args()
    project = args.project_root
    manuscript = project / "manscript"
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    workbook = manuscript / "数据表9.xlsx"
    s25_path = manuscript / "Online Resource 1" / "supplementary_tables" / "Table_S25_coring_priority.csv"
    s9_path = manuscript / "Online Resource 1" / "supplementary_tables" / "Table_S9_well_inventory.csv"
    model_rank_path = manuscript / "code" / "data_derived" / "Table_S1_full_model_rank.csv"
    map_path = manuscript / "旧文件归档" / "code_audit" / "well_name_map_final_approved.csv"
    anon_path = (
        Path(__file__).resolve().parents[1]
        / "EVO输出"
        / "审稿补充_round1_EVO独立版"
        / "_private"
        / "well_id_anon_map.csv"
    )
    if not anon_path.exists():
        anon_path = Path("D:/论文/毕业论文资料/小论文_测井跨区泛化/EVO输出/审稿补充_round1_EVO独立版/_private/well_id_anon_map.csv")

    excel = pd.ExcelFile(workbook)
    sheet_details = []
    for sheet in excel.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet)
        sheet_details.append((sheet, frame.shape, [str(column) for column in frame.columns]))
    raw = pd.read_excel(workbook, sheet_name=excel.sheet_names[0])
    s25 = pd.read_csv(s25_path, encoding="utf-8-sig")
    _ = pd.read_csv(s9_path, encoding="utf-8-sig")
    map_table = pd.read_csv(map_path, encoding="utf-8-sig")
    summary = pd.read_csv(model_rank_path, encoding="utf-8-sig")
    anon = pd.read_csv(anon_path, encoding="utf-8-sig")
    public_map = build_public_id_map(raw, map_table, s25)
    raw = prepare_raw(raw, public_map)

    s27 = build_s27(s25)
    s27.to_csv(output / "Table_S27_well_level_map_inputs.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    pool = model_pool()
    gate_rows = []
    gates = []
    deployment = []
    for target_name, spec in TARGETS.items():
        labeled = model_ready_target(raw, spec["column"])
        model_names = choose_models(summary, spec["dataset"])
        rows, gate = lowo_gate_rows(labeled, target_name, spec["column"], model_names, pool)
        gate_rows.append(rows)
        gates.append(gate)
        if not gate.passed:
            pd.DataFrame([result.__dict__ for result in gates]).to_csv(
                output / "layer2_reproduction_gate.csv", index=False, encoding="utf-8-sig"
            )
            raise RuntimeError(
                f"Layer 2 gate failed for {target_name}: {gate.fold_mean_coverage:.6f} vs {gate.anchor:.3f}"
            )
        deployment.append(
            deployment_rows(
                raw,
                labeled,
                s25[s25.target == target_name],
                target_name,
                spec["column"],
                model_names,
                pool,
            )
        )
    gate_table = pd.DataFrame([result.__dict__ for result in gates])
    gate_table.to_csv(output / "layer2_reproduction_gate.csv", index=False, encoding="utf-8-sig")
    risk_samples = pd.concat(gate_rows + deployment, ignore_index=True)
    s28, ranked = build_s28(s25, risk_samples)
    s28.to_csv(output / "Table_S28_well_level_conformal_risk.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    ranked.to_csv(output / "risk_component_audit.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    s29, _ = build_spatial_tests(s28, output)
    s29.to_csv(output / "Table_S29_spatial_autocorrelation_test.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    xlim, ylim = plot_overlays(s25, s28, output)
    plot_registration_check(s25, xlim, ylim, output)
    plot_colorbar(output)
    write_qa_previews(output)
    anon_map = build_well_anon_map(public_map, anon)
    anon_map.to_csv(output / "well_id_to_anonymous_id.csv", index=False, encoding="utf-8-sig")
    write_report(
        output,
        workbook,
        md5(workbook),
        sheet_details,
        gates,
        s27,
        s28,
        s29,
        ranked,
        xlim,
        ylim,
        anon_map,
    )
    if args.publish_base:
        publish_tables(output, args.publish_base)
    print(f"WROTE {output}")
    print(gate_table.to_string(index=False))
    print(s29.to_string(index=False))


if __name__ == "__main__":
    main()
