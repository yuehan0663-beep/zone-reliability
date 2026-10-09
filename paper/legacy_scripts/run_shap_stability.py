# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
SCRIPT = Path(__file__).resolve()
OUT_DIR = SCRIPT.parent
PROJECT_ROOT = SCRIPT.parents[3]
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
        "Could not locate derived_dataset.xlsx in canonical candidate paths: "
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

TARGETS = [
    {
        "target": "TOC",
        "dataset": "TOC_high_medium",
        "target_col": "toc_pct",
        "model_name": "ExtraTreesRegressor",
        "model": ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ),
    },
    {
        "target": "total_gas",
        "dataset": "total_gas_high_medium",
        "target_col": "total_gas_m3_t",
        "model_name": "RandomForestRegressor",
        "model": RandomForestRegressor(
            n_estimators=180, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        ),
    },
]


try:
    import shap  # type: ignore

    METHOD = "shap"
except Exception:
    shap = None
    METHOD = "permutation"


def preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
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
                cat_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def load_model_ready(target_col: str) -> pd.DataFrame:
    df = pd.read_excel(INFILE, sheet_name=SHEET)
    for col in ["well", "zone", "layer", "lithology", "diagenesis_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({"nan": np.nan})
    d = df[df[target_col].notna()].copy()
    log_cols = [col for col in LOG_COLS if col in d.columns]
    d = d[d[log_cols].notna().sum(axis=1) >= 3].reset_index(drop=True)
    return d


def feature_config(df: pd.DataFrame, feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set == "logs-only":
        return [col for col in LOG_COLS if col in df.columns], []
    if feature_set == "full":
        return [col for col in NUM_COLS if col in df.columns], [
            col for col in CAT_COLS if col in df.columns
        ]
    raise ValueError(feature_set)


def split_iter(df: pd.DataFrame, scheme: str):
    if scheme == "LOWO":
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


def anonymizers(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    wells = sorted(str(w) for w in df["well"].dropna().unique())
    well_map = {well: f"well_anon_{i:03d}" for i, well in enumerate(wells, 1)}
    zone_map = {zone: f"zone_en_{ZONE_EN.get(zone, zone)}" for zone in ZONE_ORDER}
    return well_map, zone_map


def aggregate_shap(
    pre: ColumnTransformer,
    shap_trans: np.ndarray,
    x_orig: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=x_orig.index)
    idx = 0
    for col in num_cols:
        out[col] = shap_trans[:, idx]
        idx += 1
    if cat_cols:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        for col, categories in zip(cat_cols, ohe.categories_):
            n = len(categories)
            out[col] = shap_trans[:, idx : idx + n].sum(axis=1)
            idx += n
    return out


def importance_for_fold(
    model,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.Series:
    pipe = Pipeline(
        [("pre", preprocessor(num_cols, cat_cols)), ("model", clone(model))]
    )
    pipe.fit(x_train, y_train)
    if METHOD == "shap":
        x_trans = pipe.named_steps["pre"].transform(x_train)
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        values = explainer.shap_values(x_trans)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values, dtype=float)
        shap_orig = aggregate_shap(
            pipe.named_steps["pre"], values, x_train, num_cols, cat_cols
        )
        return shap_orig.abs().mean().astype(float)

    result = permutation_importance(
        pipe,
        x_train,
        y_train,
        n_repeats=10,
        random_state=SEED,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=list(x_train.columns), dtype=float)


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or len(np.unique(a[ok])) < 2 or len(np.unique(b[ok])) < 2:
        return np.nan
    rho = spearmanr(a[ok], b[ok], nan_policy="omit").correlation
    return float(rho) if np.isfinite(rho) else np.nan


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], float]]:
    perfold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    average_ranks: dict[tuple[str, str, str], pd.Series] = {}

    for target_spec in TARGETS:
        df = load_model_ready(target_spec["target_col"])
        well_map, zone_map = anonymizers(df)
        y = df[target_spec["target_col"]].astype(float).to_numpy()

        for feature_set in ["full", "logs-only"]:
            num_cols, cat_cols = feature_config(df, feature_set)
            x = df[num_cols + cat_cols].copy()
            features = list(x.columns)

            for scheme in ["LOWO", "LOZO"]:
                fold_vectors: list[pd.Series] = []
                fold_rank_frames: list[pd.DataFrame] = []

                for fold_raw, train_idx, _ in split_iter(df, scheme):
                    if scheme == "LOWO":
                        fold_id = well_map[str(fold_raw)]
                    else:
                        fold_id = zone_map[str(fold_raw)]
                    imp = importance_for_fold(
                        target_spec["model"],
                        x.iloc[train_idx],
                        y[train_idx],
                        num_cols,
                        cat_cols,
                    ).reindex(features)
                    ranks = imp.rank(ascending=False, method="min").astype(int)
                    fold_vectors.append(imp)
                    fold_frame = pd.DataFrame(
                        {
                            "feature": features,
                            "importance": imp.to_numpy(float),
                            "rank": ranks.to_numpy(int),
                        }
                    )
                    fold_rank_frames.append(fold_frame)
                    for _, row in fold_frame.iterrows():
                        perfold_rows.append(
                            {
                                "target": target_spec["target"],
                                "feature_set": feature_set,
                                "cv_scheme": scheme,
                                "fold_id": fold_id,
                                "feature": row["feature"],
                                "importance": float(row["importance"]),
                                "rank": int(row["rank"]),
                            }
                        )

                matrix = pd.DataFrame(
                    [vec.reindex(features).to_numpy(float) for vec in fold_vectors],
                    columns=features,
                )
                pairwise = [
                    spearman_safe(matrix.iloc[i].to_numpy(), matrix.iloc[j].to_numpy())
                    for i, j in combinations(range(len(matrix)), 2)
                ]
                pairwise = [v for v in pairwise if np.isfinite(v)]
                ranks_by_fold = pd.DataFrame(
                    [frame.set_index("feature")["rank"] for frame in fold_rank_frames]
                )
                mean_rank = ranks_by_fold.mean(axis=0).sort_values()
                top8 = list(mean_rank.head(8).index)
                top5_counts = (ranks_by_fold <= 5).sum(axis=0).sort_values(
                    ascending=False
                )
                top5_freq = {
                    feature: round(float(count) / float(len(ranks_by_fold)), 6)
                    for feature, count in top5_counts.items()
                    if count > 0
                }
                average_ranks[(target_spec["target"], feature_set, scheme)] = mean_rank
                summary_rows.append(
                    {
                        "target": target_spec["target"],
                        "feature_set": feature_set,
                        "cv_scheme": scheme,
                        "pairwise_spearman_mean": float(np.mean(pairwise))
                        if pairwise
                        else np.nan,
                        "pairwise_spearman_sd": float(np.std(pairwise, ddof=1))
                        if len(pairwise) > 1
                        else np.nan,
                        "n_folds": int(len(matrix)),
                        "top5_membership_json": json.dumps(
                            top5_freq, ensure_ascii=False, sort_keys=False
                        ),
                        "top8_features": ", ".join(top8),
                        "method": METHOD,
                        "seed": SEED,
                        "model": target_spec["model_name"],
                        "n_features": len(features),
                    }
                )

    full_vs_logs: dict[tuple[str, str], float] = {}
    for target_spec in TARGETS:
        for scheme in ["LOWO", "LOZO"]:
            full_rank = average_ranks[(target_spec["target"], "full", scheme)]
            logs_rank = average_ranks[(target_spec["target"], "logs-only", scheme)]
            common = [feature for feature in logs_rank.index if feature in full_rank.index]
            full_vs_logs[(target_spec["target"], scheme)] = spearman_safe(
                full_rank[common].to_numpy(), logs_rank[common].to_numpy()
            )

    summary = pd.DataFrame(summary_rows)
    summary["full_vs_logs_only_common_feature_spearman"] = summary.apply(
        lambda row: full_vs_logs[(row["target"], row["cv_scheme"])], axis=1
    )
    perfold = pd.DataFrame(perfold_rows)
    return perfold, summary, full_vs_logs


def format_float(value: object) -> str:
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.6f}"


def validate_outputs(perfold: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    checks: list[str] = []
    expected = {
        ("TOC", "full", "LOWO"): (32, len(NUM_COLS) + len(CAT_COLS)),
        ("TOC", "logs-only", "LOWO"): (32, len(LOG_COLS)),
        ("total_gas", "full", "LOWO"): (26, len(NUM_COLS) + len(CAT_COLS)),
        ("total_gas", "logs-only", "LOWO"): (26, len(LOG_COLS)),
        ("TOC", "full", "LOZO"): (3, len(NUM_COLS) + len(CAT_COLS)),
        ("TOC", "logs-only", "LOZO"): (3, len(LOG_COLS)),
        ("total_gas", "full", "LOZO"): (3, len(NUM_COLS) + len(CAT_COLS)),
        ("total_gas", "logs-only", "LOZO"): (3, len(LOG_COLS)),
    }
    for key, (folds, n_features) in expected.items():
        target, feature_set, scheme = key
        sub = perfold[
            perfold["target"].eq(target)
            & perfold["feature_set"].eq(feature_set)
            & perfold["cv_scheme"].eq(scheme)
        ]
        got_folds = sub["fold_id"].nunique()
        got_rows = len(sub)
        want_rows = folds * n_features
        status = "PASS" if got_folds == folds and got_rows == want_rows else "FAIL"
        checks.append(
            f"- {status}: {target} {feature_set} {scheme}: "
            f"{got_folds} folds, {got_rows} rows; expected {folds} folds, {want_rows} rows."
        )

    df_toc = load_model_ready("toc_pct")
    df_gas = load_model_ready("total_gas_m3_t")
    real_tokens = set()
    for df in [df_toc, df_gas]:
        for col in ["well"]:
            real_tokens.update(str(x) for x in df[col].dropna().unique())
    text = "\n".join(
        (OUT_DIR / name).read_text(encoding="utf-8-sig", errors="ignore")
        for name in ["S22a_perfold_importance.csv", "S22b_stability_summary.csv"]
    )
    hits = sorted(token for token in real_tokens if token and token in text)
    checks.append(
        "- PASS: anonymization scan found 0 real well names."
        if not hits
        else "- FAIL: anonymization scan found real well names: " + ", ".join(hits[:8])
    )

    full_features = set(
        perfold[perfold["feature_set"].eq("full")]["feature"].dropna().unique()
    )
    log_features = set(
        perfold[perfold["feature_set"].eq("logs-only")]["feature"].dropna().unique()
    )
    checks.append(
        "- PASS: full feature list exactly matches canonical aggregated features."
        if full_features == set(NUM_COLS + CAT_COLS)
        else "- FAIL: full feature list differs from canonical aggregated features."
    )
    checks.append(
        "- PASS: logs-only feature list exactly matches Table S18 logs-only features."
        if log_features == set(LOG_COLS)
        else "- FAIL: logs-only feature list differs from Table S18 logs-only features."
    )
    if summary["method"].nunique() == 1:
        checks.append(f"- PASS: one attribution method used throughout ({METHOD}).")
    else:
        checks.append("- FAIL: mixed attribution methods detected.")
    return checks


def write_report(summary: pd.DataFrame, checks: list[str]) -> None:
    lines: list[str] = []
    lines.append("# S22 SHAP/importance fold-stability check")
    lines.append("")
    lines.append("## Method")
    lines.append(
        "Model-ready data were loaded from the same Table-9 source and filtering used by "
        "`run_full_condition_evaluation.py`: target nonmissing plus at least three available "
        "conventional logs. The attribution models match the local SHAP figure scripts "
        "(`Figure_14_shap_feature_importance_script.py` and `shap_matrix/generate_shap_matrix.py`): "
        "TOC uses ExtraTreesRegressor(n_estimators=180, min_samples_leaf=3, random_state=42), and total "
        "gas uses RandomForestRegressor(n_estimators=180, min_samples_leaf=3, random_state=42). "
        "For every LOWO or LOZO fold, imputation, scaling, one-hot encoding, model fitting, "
        "and SHAP aggregation were fit only on the training partition. Target-derived fields "
        "listed in the canonical exclusion list were not included."
    )
    if METHOD == "shap":
        lines.append(
            "Attribution method: fold-wise TreeExplainer mean absolute SHAP values, aggregated "
            "from transformed one-hot columns back to original canonical feature names."
        )
    else:
        lines.append(
            "Attribution method: fold-wise permutation importance on the training partition "
            "with n_repeats=10 and seed=42, used uniformly because SHAP was unavailable."
        )
    lines.append("")
    lines.append("## Stability Summary")
    lines.append("")
    cols = [
        "target",
        "feature_set",
        "cv_scheme",
        "n_folds",
        "pairwise_spearman_mean",
        "pairwise_spearman_sd",
        "top8_features",
        "full_vs_logs_only_common_feature_spearman",
    ]
    lines.append(
        "| target | feature set | CV | folds | pairwise Spearman mean | pairwise Spearman sd | top-8 mean-rank features | full vs logs-only Spearman |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|---:|")
    for _, row in summary[cols].iterrows():
        lines.append(
            "| {target} | {feature_set} | {cv_scheme} | {n_folds} | {mean} | {sd} | {top8} | {cross} |".format(
                target=row["target"],
                feature_set=row["feature_set"],
                cv_scheme=row["cv_scheme"],
                n_folds=int(row["n_folds"]),
                mean=format_float(row["pairwise_spearman_mean"]),
                sd=format_float(row["pairwise_spearman_sd"]),
                top8=row["top8_features"],
                cross=format_float(row["full_vs_logs_only_common_feature_spearman"]),
            )
        )
    lines.append("")
    lines.append("## Top-5 Membership")
    lines.append("")
    lines.append("| target | feature set | CV | top-5 membership frequency JSON |")
    lines.append("|---|---|---|---|")
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['target']} | {row['feature_set']} | {row['cv_scheme']} | `{row['top5_membership_json']}` |"
        )
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.extend(checks)
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append(
        "Run from any working directory with: "
        f"`py -3.11 \"{SCRIPT}\"`"
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `S22a_perfold_importance.csv`")
    lines.append("- `S22b_stability_summary.csv`")
    lines.append("- `report_shap_stability.md`")
    lines.append("- `run_shap_stability.py`")
    (OUT_DIR / "report_shap_stability.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perfold, summary, _ = run_experiment()
    perfold.to_csv(
        OUT_DIR / "S22a_perfold_importance.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        OUT_DIR / "S22b_stability_summary.csv", index=False, encoding="utf-8-sig"
    )
    checks = validate_outputs(perfold, summary)
    write_report(summary, checks)
    if any(line.startswith("- FAIL") for line in checks):
        print("\n".join(checks))
        raise SystemExit(2)
    print("PASS")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
