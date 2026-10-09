# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

SEED = 42
BOOTSTRAP_SEED = 20260613
BOOTSTRAP_N = 500
ALPHA = 0.90

SCRIPT = Path(__file__).resolve()
OUT_DIR = SCRIPT.parent
PROJECT_ROOT = SCRIPT.parents[3]
CANON_DIR = PROJECT_ROOT / "EVO\u8f93\u51fa" / "full_condition_analysis_derived_dataset"
RECAL_DIR = PROJECT_ROOT / "EVO\u8f93\u51fa" / "within_zone_recalibration_analysis_derived_dataset"
SHEET = "\u6570\u636e"

INPUT_CANDIDATES = [
    PROJECT_ROOT / "Input Program" / "derived_dataset.xlsx",
    PROJECT_ROOT / "EVO\u8f93\u51fa" / "Input Program" / "derived_dataset.xlsx",
    Path(r"PROJECT_ROOT_PLACEHOLDER/evo_workspace\Input Program\derived_dataset.xlsx"),
]
for candidate in INPUT_CANDIDATES:
    if candidate.exists():
        INFILE = candidate
        break
else:
    raise FileNotFoundError(
        "Cannot locate derived_dataset.xlsx in: "
        + "; ".join(str(path) for path in INPUT_CANDIDATES)
    )

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
EXCLUDED = [
    "organic_level",
    "gas_interpretation",
    "toc_pct",
    "total_gas_m3_t",
    "desorbed_gas_m3_t",
    "residual_free_gas_m3_t",
    "desorption_gas_content_m3_t",
    "lost_gas_m3_t",
]
ZONE_ORDER = ["\u5317", "\u4e2d", "\u5357"]
ZONE_EN = {"\u5317": "North", "\u4e2d": "Central", "\u5357": "South"}
TARGET_LABEL = {"toc_pct": "TOC", "total_gas_m3_t": "total_gas"}
DATASET_TARGET = {
    "TOC_high_medium": "toc_pct",
    "total_gas_high_medium": "total_gas_m3_t",
}
MODEL_ORDER = [
    "train_mean",
    "zone_mean",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "Huber",
    "SVR_RBF",
    "KNN",
    "RF",
    "ExtraTrees",
    "GBDT",
    "HistGBDT",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

ANCHOR_A = {
    ("TOC", "LOWO", "global"): (0.871, 1.134),
    ("TOC", "LOWO", "risk_inflated"): (0.926, 1.377),
    ("TOC", "LOZO", "global"): (0.720, None),
    ("TOC", "LOZO", "risk_inflated"): (0.805, None),
    ("total_gas", "LOWO", "global"): (0.914, 1.609),
    ("total_gas", "LOWO", "risk_inflated"): (0.949, 1.950),
    ("total_gas", "LOZO", "global"): (0.788, None),
    ("total_gas", "LOZO", "risk_inflated"): (0.850, None),
}


try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None


class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, group_col: str = "zone"):
        self.group_col = group_col

    def fit(self, x, y):
        frame = pd.DataFrame(x).copy()
        frame["__y__"] = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.nanmean(frame["__y__"]))
        self.group_means_ = (
            frame.groupby(self.group_col)["__y__"].mean().to_dict()
            if self.group_col in frame.columns
            else {}
        )
        return self

    def predict(self, x):
        frame = pd.DataFrame(x)
        if self.group_col not in frame.columns:
            return np.full(len(frame), self.global_mean_)
        return frame[self.group_col].map(self.group_means_).fillna(self.global_mean_).to_numpy(float)


def preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            )
        )
    if cat:
        transformers.append(
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
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def regressor_pool() -> dict[str, object]:
    models: dict[str, object] = {
        "train_mean": DummyRegressor(strategy="mean"),
        "zone_mean": GroupMeanRegressor("zone"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=20000, random_state=SEED),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=SEED),
        "Huber": HuberRegressor(max_iter=2000),
        "SVR_RBF": SVR(C=3.0, gamma="scale", epsilon=0.05),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "RF": RandomForestRegressor(
            n_estimators=180, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ),
        "GBDT": GradientBoostingRegressor(
            random_state=SEED, n_estimators=180, learning_rate=0.04, max_depth=2
        ),
        "HistGBDT": HistGradientBoostingRegressor(
            random_state=SEED, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=2,
            verbosity=0,
        )
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=180,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=SEED,
            n_jobs=2,
            verbose=-1,
        )
    if CatBoostRegressor is not None:
        models["CatBoost"] = CatBoostRegressor(
            iterations=180,
            depth=3,
            learning_rate=0.04,
            loss_function="RMSE",
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
        )
    return models


def make_pipe(name: str, model, num: list[str], cat: list[str]):
    if name == "zone_mean":
        return clone(model)
    return Pipeline([("pre", preprocessor(num, cat)), ("model", clone(model))])


def transformed(pipe, x: pd.DataFrame) -> np.ndarray:
    if isinstance(pipe, Pipeline):
        return pipe.named_steps["pre"].transform(x)
    return x.select_dtypes(include=[np.number]).fillna(0).to_numpy()


def split_iter(df: pd.DataFrame, scheme: str):
    if scheme == "random":
        for i, (tr, te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=SEED).split(df), 1):
            yield f"random_{i}", np.asarray(tr), np.asarray(te)
    elif scheme == "LOWO":
        for well in sorted(df["well"].dropna().unique()):
            te = np.where(df["well"].to_numpy() == well)[0]
            tr = np.where(df["well"].to_numpy() != well)[0]
            if len(te) and len(tr) >= 10:
                yield str(well), tr, te
    elif scheme == "LOZO":
        for zone in ZONE_ORDER:
            te = np.where(df["zone"].to_numpy() == zone)[0]
            tr = np.where(df["zone"].to_numpy() != zone)[0]
            if len(te) and len(tr) >= 10:
                yield str(zone), tr, te
    else:
        raise ValueError(scheme)


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(INFILE, sheet_name=SHEET)
    for col in ["well", "zone", "layer", "lithology", "diagenesis_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": np.nan})
    return df


def load_datasets(raw: pd.DataFrame):
    datasets = []
    for name, target in [("TOC_high_medium", "toc_pct"), ("total_gas_high_medium", "total_gas_m3_t")]:
        d = raw[raw[target].notna()].copy()
        d = d[d[[col for col in LOG_COLS if col in d.columns]].notna().sum(axis=1) >= 3].reset_index(drop=True)
        d["cleaning_confidence"] = "high_medium"
        datasets.append((name, target, d))
    return datasets


def choose_models_from_canonical(dataset: str) -> list[str]:
    summary = pd.read_csv(CANON_DIR / "metrics_pooled_corrected.csv", encoding="utf-8-sig")
    sub = summary[
        summary["dataset"].eq(dataset)
        & summary["cv_scheme"].isin(["LOWO", "LOZO"])
        & ~summary["model"].isin(["train_mean", "zone_mean"])
    ]
    rank = (
        sub.groupby("model")
        .agg(MAE=("MAE_w", "mean"), S=("Spearman_mean", "mean"))
        .reset_index()
        .sort_values(["MAE", "S"], ascending=[True, False])
    )
    return rank.head(3)["model"].tolist()


def quantile_empirical(resid: np.ndarray, alpha: float = ALPHA) -> float:
    return float(np.quantile(np.asarray(resid, dtype=float), alpha))


def quantile_finite(resid: np.ndarray, alpha: float = ALPHA) -> float:
    vals = np.sort(np.asarray(resid, dtype=float))
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan
    k = int(math.ceil((n + 1) * alpha))
    if k > n:
        return float(vals[-1])
    return float(vals[k - 1])


def interval_metrics(y_true: np.ndarray, pred: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo = pred - q
    hi = pred + q
    covered = (y_true >= lo) & (y_true <= hi)
    return covered, float(np.mean(covered)), float(np.mean(hi - lo))


def zone_quantiles(
    resid: np.ndarray,
    cal_zone: np.ndarray,
    test_zone: Iterable[str],
    fallback: float,
    finite: bool,
) -> np.ndarray:
    qfun = quantile_finite if finite else quantile_empirical
    zone_q = {
        zone: qfun(resid[cal_zone == zone])
        for zone in np.unique(cal_zone)
        if int((cal_zone == zone).sum()) >= 3
    }
    return np.array([zone_q.get(str(zone), fallback) for zone in test_zone], dtype=float)


def summarize_rows(rows: list[dict[str, object]], variant: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if variant in {"baseline", "A1_finite_sample_quantile"}:
        out = (
            df.groupby(["target", "cv_scheme", "method", "variant"], dropna=False)
            .agg(coverage=("coverage", "mean"), width=("width", "mean"), n_rows=("coverage", "size"))
            .reset_index()
        )
        out["coverage_min"] = out["coverage"]
        out["coverage_median"] = out["coverage"]
        out["coverage_max"] = out["coverage"]
        out["width_min"] = out["width"]
        out["width_median"] = out["width"]
        out["width_max"] = out["width"]
        return out
    out = (
        df.groupby(["target", "cv_scheme", "method", "variant"], dropna=False)
        .agg(
            coverage=("coverage_median", "mean"),
            width=("width_median", "mean"),
            coverage_min=("coverage_min", "mean"),
            coverage_median=("coverage_median", "mean"),
            coverage_max=("coverage_max", "mean"),
            width_min=("width_min", "mean"),
            width_median=("width_median", "mean"),
            width_max=("width_max", "mean"),
            n_rows=("coverage_median", "size"),
        )
        .reset_index()
    )
    return out


def decision_label(target: str, zone_en: str, coverage: float) -> str:
    if coverage >= 0.80:
        return "near-nominal"
    if target == "TOC" and zone_en == "North":
        return "low-coverage-calibratable"
    return "low-coverage-acquire"


def run_conformal_sensitivity(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = regressor_pool()
    baseline_rows: list[dict[str, object]] = []
    a1_rows: list[dict[str, object]] = []
    a2_rows: list[dict[str, object]] = []
    zone_records: list[dict[str, object]] = []

    for dataset, target_col, df in load_datasets(raw):
        target = TARGET_LABEL[target_col]
        model_names = choose_models_from_canonical(dataset)
        missing = [name for name in model_names if name not in models]
        if missing:
            raise RuntimeError(f"Missing canonical conformal models for {dataset}: {missing}")
        num = [col for col in NUM_COLS if col in df.columns]
        cat = [col for col in CAT_COLS if col in df.columns]
        x = df[num + cat].copy()
        y = df[target_col].astype(float).to_numpy()

        for model_name in model_names:
            for scheme in ["LOWO", "LOZO"]:
                for fold, tr_all, te in split_iter(df, scheme):
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
                    if len(tr) < 10 or len(cal) < 5:
                        continue

                    pipe = make_pipe(model_name, models[model_name], num, cat)
                    pipe.fit(x.iloc[tr], y[tr])
                    cal_pred = np.asarray(pipe.predict(x.iloc[cal]), dtype=float)
                    pred = np.asarray(pipe.predict(x.iloc[te]), dtype=float)
                    y_te = y[te]
                    resid = np.abs(y[cal] - cal_pred)
                    cal_zone = df.iloc[cal]["zone"].astype(str).to_numpy()
                    test_zone = df.iloc[te]["zone"].astype(str).to_numpy()
                    zone_en = [ZONE_EN.get(str(zone), str(zone)) for zone in test_zone]
                    x_tr = transformed(pipe, x.iloc[tr])
                    x_te = transformed(pipe, x.iloc[te])
                    dist = (
                        NearestNeighbors(n_neighbors=min(5, len(tr)))
                        .fit(x_tr)
                        .kneighbors(x_te, return_distance=True)[0]
                        .mean(axis=1)
                    )
                    dn = (dist - dist.min()) / (dist.max() - dist.min() + 1e-9)

                    q_emp = quantile_empirical(resid)
                    q_fin = quantile_finite(resid)
                    qz_emp = zone_quantiles(resid, cal_zone, test_zone, q_emp, finite=False)
                    qz_fin = zone_quantiles(resid, cal_zone, test_zone, q_fin, finite=True)
                    q_map = {
                        "global": np.full(len(te), q_emp),
                        "zone_conditional": qz_emp,
                        "risk_inflated": q_emp * (1 + 0.5 * dn),
                    }
                    q_map_a1 = {
                        "global": np.full(len(te), q_fin),
                        "zone_conditional": qz_fin,
                        "risk_inflated": q_fin * (1 + 0.5 * dn),
                    }
                    for method, q in q_map.items():
                        covered, cov, width = interval_metrics(y_te, pred, q)
                        baseline_rows.append(
                            {
                                "target": target,
                                "dataset": dataset,
                                "model": model_name,
                                "cv_scheme": scheme,
                                "fold": fold,
                                "method": method,
                                "variant": "baseline",
                                "coverage": cov,
                                "width": width,
                            }
                        )
                        if method == "zone_conditional":
                            for z_en, c in zip(zone_en, covered):
                                zone_records.append(
                                    {
                                        "target": target,
                                        "zone_en": z_en,
                                        "variant": "baseline",
                                        "covered": bool(c),
                                    }
                                )
                    for method, q in q_map_a1.items():
                        covered, cov, width = interval_metrics(y_te, pred, q)
                        a1_rows.append(
                            {
                                "target": target,
                                "dataset": dataset,
                                "model": model_name,
                                "cv_scheme": scheme,
                                "fold": fold,
                                "method": method,
                                "variant": "A1_finite_sample_quantile",
                                "coverage": cov,
                                "width": width,
                            }
                        )
                        if method == "zone_conditional":
                            for z_en, c in zip(zone_en, covered):
                                zone_records.append(
                                    {
                                        "target": target,
                                        "zone_en": z_en,
                                        "variant": "A1_finite_sample_quantile",
                                        "covered": bool(c),
                                    }
                                )

                    for method in ["global", "zone_conditional", "risk_inflated"]:
                        cov_values: list[float] = []
                        width_values: list[float] = []
                        scenario_records: list[tuple[str, np.ndarray, float, float]] = []
                        for left_well in sorted(cal_wells):
                            keep = df.iloc[cal]["well"].to_numpy() != left_well
                            if int(keep.sum()) < 1:
                                continue
                            r2 = resid[keep]
                            z2 = cal_zone[keep]
                            q2_global = quantile_empirical(r2)
                            if method == "global":
                                q2 = np.full(len(te), q2_global)
                            elif method == "zone_conditional":
                                q2 = zone_quantiles(r2, z2, test_zone, q2_global, finite=False)
                            else:
                                q2 = q2_global * (1 + 0.5 * dn)
                            covered, cov, width = interval_metrics(y_te, pred, q2)
                            cov_values.append(cov)
                            width_values.append(width)
                            scenario_records.append((str(left_well), covered, cov, width))
                        if not cov_values:
                            continue
                        cov_arr = np.asarray(cov_values, dtype=float)
                        wid_arr = np.asarray(width_values, dtype=float)
                        a2_rows.append(
                            {
                                "target": target,
                                "dataset": dataset,
                                "model": model_name,
                                "cv_scheme": scheme,
                                "fold": fold,
                                "method": method,
                                "variant": "A2_leave_one_calibration_well",
                                "coverage_min": float(np.min(cov_arr)),
                                "coverage_median": float(np.median(cov_arr)),
                                "coverage_max": float(np.max(cov_arr)),
                                "width_min": float(np.min(wid_arr)),
                                "width_median": float(np.median(wid_arr)),
                                "width_max": float(np.max(wid_arr)),
                            }
                        )
                        if method == "zone_conditional":
                            min_idx = int(np.argmin(cov_arr))
                            max_idx = int(np.argmax(cov_arr))
                            for variant_name, scenario_idx in [("A2min", min_idx), ("A2max", max_idx)]:
                                _, covered, _, _ = scenario_records[scenario_idx]
                                for z_en, c in zip(zone_en, covered):
                                    zone_records.append(
                                        {
                                            "target": target,
                                            "zone_en": z_en,
                                            "variant": variant_name,
                                            "covered": bool(c),
                                        }
                                    )

    baseline = summarize_rows(baseline_rows, "baseline")
    a1 = summarize_rows(a1_rows, "A1_finite_sample_quantile")
    a2 = summarize_rows(a2_rows, "A2_leave_one_calibration_well")
    out = pd.concat([baseline, a1, a2], ignore_index=True)
    anchors = baseline[baseline["variant"].eq("baseline")][
        ["target", "cv_scheme", "method", "coverage", "width"]
    ].rename(columns={"coverage": "anchor_coverage", "width": "anchor_width"})
    out = out.merge(anchors, on=["target", "cv_scheme", "method"], how="left")
    out["coverage_delta_vs_anchor"] = out["coverage"] - out["anchor_coverage"]
    out["width_delta_vs_anchor"] = out["width"] - out["anchor_width"]

    zr = pd.DataFrame(zone_records)
    zone_cov = (
        zr.groupby(["target", "zone_en", "variant"], dropna=False)
        .agg(coverage=("covered", "mean"), n_rows=("covered", "size"))
        .reset_index()
    )
    pivot = zone_cov.pivot(index=["target", "zone_en"], columns="variant", values="coverage").reset_index()
    for col in ["baseline", "A1_finite_sample_quantile", "A2min", "A2max"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot.rename(
        columns={
            "baseline": "baseline_coverage",
            "A1_finite_sample_quantile": "A1_coverage",
            "A2min": "A2min_coverage",
            "A2max": "A2max_coverage",
        }
    )
    for col in ["baseline", "A1", "A2min", "A2max"]:
        cov_col = f"{col}_coverage"
        pivot[f"{col}_label"] = [
            decision_label(str(t), str(z), float(c))
            for t, z, c in zip(pivot["target"], pivot["zone_en"], pivot[cov_col])
        ]
    pivot["label_flipped_A1"] = pivot["A1_label"] != pivot["baseline_label"]
    pivot["label_flipped_A2min"] = pivot["A2min_label"] != pivot["baseline_label"]
    pivot["label_flipped_A2max"] = pivot["A2max_label"] != pivot["baseline_label"]
    pivot["any_label_flip"] = (
        pivot["label_flipped_A1"] | pivot["label_flipped_A2min"] | pivot["label_flipped_A2max"]
    )
    return out, pivot, pd.DataFrame(zone_records)


def add_schmoker(df: pd.DataFrame, quantile: float = 0.90, coeff: float = 1.378) -> tuple[pd.DataFrame, dict[str, float]]:
    d = df.copy()
    bases = {
        str(zone): float(group["DEN"].quantile(quantile))
        for zone, group in d.groupby("zone")
        if group["DEN"].notna().any()
    }
    d["schmoker_toc_proxy"] = [
        max(0.0, (bases.get(str(zone), np.nan) - float(den)) * coeff)
        if pd.notna(den) and str(zone) in bases
        else np.nan
        for zone, den in zip(d["zone"], d["DEN"])
    ]
    return d, bases


def mae(y, pred) -> float:
    return float(mean_absolute_error(np.asarray(y, dtype=float), np.asarray(pred, dtype=float)))


def bootstrap_improvement(y, pred, baseline, wells, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    unique = sorted(set(wells))
    if len(unique) < 3:
        return np.nan, np.nan
    idx_by_well = {well: [i for i, value in enumerate(wells) if value == well] for well in unique}
    values = []
    for _ in range(n_boot):
        idx: list[int] = []
        for sampled in rng.choice(unique, size=len(unique), replace=True):
            idx.extend(idx_by_well[sampled])
        yy = [y[i] for i in idx]
        pp = [pred[i] for i in idx]
        zz = [baseline[i] for i in idx]
        base = mae(yy, zz)
        corr = mae(yy, pp)
        if base > 0:
            values.append((base - corr) / base * 100.0)
    if len(values) < 20:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_c1_zone(
    raw: pd.DataFrame,
    zone: str,
    variant: str,
    quantile: float = 0.90,
) -> dict[str, object]:
    y_all: list[float] = []
    pred_all: list[float] = []
    base_all: list[float] = []
    wells_all: list[str] = []
    offsets: list[float] = []

    if variant == "B1_strict_fold_density_baseline":
        base_source = raw[raw["zone"].eq(zone)].copy()
        label_source = raw[
            raw["zone"].eq(zone) & raw["toc_pct"].notna() & raw["DEN"].notna()
        ].copy()
        for well in sorted(label_source["well"].dropna().unique()):
            den_train = base_source[base_source["well"].ne(well)]["DEN"].dropna()
            if len(den_train) == 0:
                continue
            rho0 = float(den_train.quantile(0.90))
            local = label_source.copy()
            local["schmoker_toc_proxy"] = np.maximum(0.0, (rho0 - local["DEN"].astype(float)) * 1.378)
            tr = local[local["well"].ne(well)]
            te = local[local["well"].eq(well)]
            if len(tr) < 3 or len(te) == 0:
                continue
            offset = float((tr["toc_pct"] - tr["schmoker_toc_proxy"]).mean())
            y_all.extend(te["toc_pct"].astype(float).tolist())
            pred_all.extend((te["schmoker_toc_proxy"] + offset).astype(float).tolist())
            base_all.extend([float(tr["toc_pct"].mean())] * len(te))
            wells_all.extend([str(well)] * len(te))
            offsets.append(offset)
    else:
        d, _ = add_schmoker(raw, quantile=quantile)
        label_source = d[
            d["zone"].eq(zone) & d["toc_pct"].notna() & d["schmoker_toc_proxy"].notna()
        ].copy()
        for well in sorted(label_source["well"].dropna().unique()):
            tr = label_source[label_source["well"].ne(well)]
            te = label_source[label_source["well"].eq(well)]
            if len(tr) < 3 or len(te) == 0:
                continue
            offset = float((tr["toc_pct"] - tr["schmoker_toc_proxy"]).mean())
            y_all.extend(te["toc_pct"].astype(float).tolist())
            pred_all.extend((te["schmoker_toc_proxy"] + offset).astype(float).tolist())
            base_all.extend([float(tr["toc_pct"].mean())] * len(te))
            wells_all.extend([str(well)] * len(te))
            offsets.append(offset)

    base_mae = mae(y_all, base_all)
    corr_mae = mae(y_all, pred_all)
    improvement = (base_mae - corr_mae) / base_mae * 100.0 if base_mae > 0 else np.nan
    ci_low, ci_high = bootstrap_improvement(y_all, pred_all, base_all, wells_all)
    return {
        "zone": zone,
        "zone_en": ZONE_EN.get(zone, zone),
        "variant": variant,
        "density_quantile": quantile,
        "n_points": len(y_all),
        "n_wells": len(set(wells_all)),
        "baseline_MAE": base_mae,
        "corrected_MAE": corr_mae,
        "relative_improvement_pct": improvement,
        "improvement_ci_low_pct": ci_low,
        "improvement_ci_high_pct": ci_high,
        "mean_offset": float(np.mean(offsets)) if offsets else np.nan,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def run_schmoker_sensitivity(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zone in ZONE_ORDER:
        rows.append(evaluate_c1_zone(raw, zone, "baseline", 0.90))
        rows.append(evaluate_c1_zone(raw, zone, "B1_strict_fold_density_baseline", 0.90))
        rows.append(evaluate_c1_zone(raw, zone, "B2_density_quantile_0.85", 0.85))
        rows.append(evaluate_c1_zone(raw, zone, "B2_density_quantile_0.95", 0.95))
    return pd.DataFrame(rows)


def check_anchors(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    checks = []
    baseline = a[a["variant"].eq("baseline")].copy()
    for key, (coverage_anchor, width_anchor) in ANCHOR_A.items():
        target, scheme, method = key
        row = baseline[
            baseline["target"].eq(target)
            & baseline["cv_scheme"].eq(scheme)
            & baseline["method"].eq(method)
        ]
        if row.empty:
            checks.append(f"FAIL A anchor missing {key}")
            continue
        coverage = float(row["coverage"].iloc[0])
        ok = round(coverage, 3) == round(float(coverage_anchor), 3)
        msg = f"{'PASS' if ok else 'FAIL'} A {target} {scheme} {method} coverage {coverage:.6f} anchor {coverage_anchor:.3f}"
        checks.append(msg)
        if width_anchor is not None:
            width = float(row["width"].iloc[0])
            ok_w = round(width, 3) == round(float(width_anchor), 3)
            checks.append(
                f"{'PASS' if ok_w else 'FAIL'} A {target} {scheme} {method} width {width:.6f} anchor {width_anchor:.3f}"
            )
    north = b[b["zone_en"].eq("North") & b["variant"].eq("baseline")]
    if north.empty:
        checks.append("FAIL B North baseline missing")
    else:
        r = north.iloc[0]
        checks.append(
            f"{'PASS' if round(float(r['baseline_MAE']), 3) == 0.243 else 'FAIL'} B North baseline_MAE {float(r['baseline_MAE']):.6f} anchor 0.243"
        )
        checks.append(
            f"{'PASS' if round(float(r['corrected_MAE']), 3) == 0.225 else 'FAIL'} B North corrected_MAE {float(r['corrected_MAE']):.6f} anchor 0.225"
        )
        checks.append(
            f"{'PASS' if round(float(r['relative_improvement_pct']), 2) == 7.39 else 'FAIL'} B North improvement {float(r['relative_improvement_pct']):.6f} anchor 7.39"
        )
    return checks


def write_report(a: pd.DataFrame, z: pd.DataFrame, b: pd.DataFrame, checks: list[str]) -> None:
    lines: list[str] = []
    lines.append("# Protocol sensitivity protocol sensitivity")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append(
        "Sub-experiment A reuses the canonical split-conformal implementation from "
        "`run_full_condition_evaluation.py`: the same model-ready model-ready dataset data, feature set, "
        "outer LOWO/LOZO folds, top-three canonical model pool selected by `choose_models()`, "
        "training/calibration split, empirical residual intervals, and distance inflation. "
        "Only the protocol convention named in the instruction was changed: A1 replaces the "
        "empirical 0.90 residual quantile with the finite-sample conformal order statistic, "
        "and A2 leaves out one calibration well at a time before recomputing q90."
    )
    lines.append(
        "Sub-experiment B reuses the North/Central/South TOC Schmoker C1 LOWO offset protocol: "
        "the baseline prediction is the training-well zone mean and the correction offset is "
        "estimated inside each held-out-well fold. Only the density-baseline convention was "
        "changed: B1 recomputes rho_b,0 from training wells inside each fold, and B2 changes "
        "the full-zone density quantile from 0.90 to 0.85 or 0.95."
    )
    lines.append("")
    lines.append("## Anchor Checks")
    lines.append("")
    lines.extend(f"- {line}" for line in checks)
    lines.append("")
    lines.append("## A. Conformal Sensitivity")
    lines.append("")
    lines.append("| target | scheme | method | variant | coverage | width | delta coverage | delta width |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for _, r in a.sort_values(["target", "cv_scheme", "method", "variant"]).iterrows():
        lines.append(
            f"| {r['target']} | {r['cv_scheme']} | {r['method']} | {r['variant']} | "
            f"{float(r['coverage']):.6f} | {float(r['width']):.6f} | "
            f"{float(r['coverage_delta_vs_anchor']):.6f} | {float(r['width_delta_vs_anchor']):.6f} |"
        )
    lines.append("")
    lines.append("## zone_coverage. Zone Label Check")
    lines.append("")
    lines.append("| target | zone | baseline | A1 | A2 min | A2 max | baseline label | any flip |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for _, r in z.sort_values(["target", "zone_en"]).iterrows():
        lines.append(
            f"| {r['target']} | {r['zone_en']} | {float(r['baseline_coverage']):.6f} | "
            f"{float(r['A1_coverage']):.6f} | {float(r['A2min_coverage']):.6f} | "
            f"{float(r['A2max_coverage']):.6f} | {r['baseline_label']} | {bool(r['any_label_flip'])} |"
        )
    lines.append("")
    lines.append("Decision label rule used here: coverage >= 0.80 -> near-nominal; otherwise North TOC -> low-coverage-calibratable; all other low-coverage blocks -> low-coverage-acquire.")
    lines.append("")
    lines.append("## B. Schmoker Baseline Sensitivity")
    lines.append("")
    lines.append("| zone | variant | baseline MAE | corrected MAE | improvement % | CI low | CI high | mean offset |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in b.sort_values(["zone_en", "variant"]).iterrows():
        lines.append(
            f"| {r['zone_en']} | {r['variant']} | {float(r['baseline_MAE']):.6f} | "
            f"{float(r['corrected_MAE']):.6f} | {float(r['relative_improvement_pct']):.6f} | "
            f"{float(r['improvement_ci_low_pct']):.6f} | {float(r['improvement_ci_high_pct']):.6f} | "
            f"{float(r['mean_offset']):.6f} |"
        )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append(f"`py -3.11 \"{SCRIPT}\"`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `A_conformal_sensitivity.csv`")
    lines.append("- `A_zone_label_check.csv`")
    lines.append("- `B_schmoker_baseline_sensitivity.csv`")
    lines.append("- `report_protocol_sensitivity.md`")
    lines.append("- `run_protocol_sensitivity.py`")
    (OUT_DIR / "report_protocol_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    a, zone, _ = run_conformal_sensitivity(raw)
    b = run_schmoker_sensitivity(raw)
    a.to_csv(OUT_DIR / "A_conformal_sensitivity.csv", index=False, encoding="utf-8-sig")
    zone.to_csv(OUT_DIR / "A_zone_label_check.csv", index=False, encoding="utf-8-sig")
    b.to_csv(OUT_DIR / "B_schmoker_baseline_sensitivity.csv", index=False, encoding="utf-8-sig")
    checks = check_anchors(a, b)
    write_report(a, zone, b, checks)
    if any(line.startswith("FAIL") for line in checks):
        print("\n".join(checks))
        raise SystemExit(2)
    print("PASS")
    print("\n".join(checks))
    print(a[["target", "cv_scheme", "method", "variant", "coverage", "width"]].to_string(index=False))
    print(b[["zone_en", "variant", "baseline_MAE", "corrected_MAE", "relative_improvement_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
