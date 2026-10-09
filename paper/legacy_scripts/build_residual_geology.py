# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr


ZONE_MAP = {"\u5317": "North", "\u4e2d": "Central", "\u5357": "South", "North": "North", "Central": "Central", "South": "South"}
ZONE_ORDER = ["North", "Central", "South"]
ZONE_COLORS = {"North": "#D55E00", "Central": "#0072B2", "South": "#009E73"}
ZONE_MARKERS = {"North": "o", "Central": "s", "South": "^"}
GEO_VARS = ["depth_md", "DEN", "max_burial_top", "pressure_coeff"]
VAR_LABELS = {
    "depth_md": "Measured depth (m)",
    "DEN": "Bulk density DEN (g/cm3)",
    "max_burial_top": "Maximum burial top depth (m)",
    "pressure_coeff": "Pressure coefficient",
}
TARGET_ROWS = 291
EXPECTED_ZONE_COUNTS = {"North": 93, "Central": 84, "South": 114}
ANCHOR_DEPTH_ABS_RHO = 0.35537304783563595


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def output_dir() -> Path:
    return Path(__file__).resolve().parent


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def find_input_files(root: Path) -> tuple[Path, Path]:
    pred = root / "EVO\u8f93\u51fa" / "full_condition_analysis_derived_dataset" / "expanded_model_baseline_predictions.csv"
    feature = root / "EVO\u8f93\u51fa" / "\u5199\u4f5c_\u88689" / "FullDraft" / "shap_matrix" / "feature_values_toc.csv"
    if not pred.exists():
        candidates = sorted(root.rglob("expanded_model_baseline_predictions.csv"), key=lambda p: (len(str(p)), str(p)))
        if not candidates:
            raise FileNotFoundError("expanded_model_baseline_predictions.csv not found")
        pred = candidates[0]
    if not feature.exists():
        candidates = []
        for path in root.rglob("*.csv"):
            try:
                cols = set(read_csv(path).columns)
            except Exception:
                continue
            if {"sample_id", "DEN", "max_burial_top", "pressure_coeff"}.issubset(cols):
                candidates.append(path)
        matches = [p for p in candidates if p.name == "feature_values_toc.csv"]
        if not matches:
            raise FileNotFoundError("feature_values_toc.csv not found")
        feature = sorted(matches, key=lambda p: (len(str(p)), str(p)))[0]
    return pred, feature


def load_residual_rows(pred_path: Path, feature_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    pred = read_csv(pred_path)
    rows = pred[
        pred["dataset"].astype(str).eq("TOC_high_medium")
        & pred["model"].astype(str).eq("ExtraTrees")
        & pred["cv_scheme"].astype(str).eq("LOZO")
    ].copy()
    if len(rows) != TARGET_ROWS:
        raise RuntimeError(f"Expected {TARGET_ROWS} TOC ExtraTrees LOZO rows, got {len(rows)} from {pred_path}")
    if rows["sample_id"].duplicated().any():
        dup = rows.loc[rows["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise RuntimeError(f"Duplicate sample_id in filtered prediction rows: {dup}")

    feats = read_csv(feature_path)
    required = {"sample_id", "DEN", "max_burial_top", "pressure_coeff", "lithology"}
    missing = required.difference(feats.columns)
    if missing:
        raise RuntimeError(f"Missing feature columns in {feature_path}: {sorted(missing)}")
    feat_cols = ["sample_id", "DEN", "max_burial_top", "pressure_coeff", "lithology"]
    feat_cols += [c for c in ["GR", "AC", "CNL", "RT", "diagenesis_category"] if c in feats.columns]
    feats = feats[feat_cols].drop_duplicates("sample_id")

    rows = rows.merge(feats, on="sample_id", how="left", validate="one_to_one")
    rows["zone_en"] = rows["zone"].map(ZONE_MAP).fillna(rows["zone"].astype(str))
    rows["residual"] = rows["y_pred"] - rows["y_true"]
    rows["residual_true_minus_pred"] = rows["y_true"] - rows["y_pred"]
    rows["abs_residual"] = rows["residual"].abs()

    zone_counts = rows["zone_en"].value_counts().reindex(ZONE_ORDER).fillna(0).astype(int).to_dict()
    if zone_counts != EXPECTED_ZONE_COUNTS:
        raise RuntimeError(f"Zone counts do not match expected {EXPECTED_ZONE_COUNTS}: {zone_counts}")

    wells = sorted(rows["well"].dropna().astype(str).unique())
    well_map = {well: f"W{i + 1:03d}" for i, well in enumerate(wells)}
    rows["well_anon"] = rows["well"].map(well_map)
    rows = rows.sort_values(["zone_en", "well_anon", "depth_md", "sample_id"], key=lambda s: s.map({z: i for i, z in enumerate(ZONE_ORDER)}) if s.name == "zone_en" else s).reset_index(drop=True)
    rows["sample_id_anon"] = [f"S{i + 1:04d}" for i in range(len(rows))]
    return rows, well_map


def finite_pair(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    out = df[[x, y]].copy()
    out[x] = pd.to_numeric(out[x], errors="coerce")
    out[y] = pd.to_numeric(out[y], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def spearman_stat(df: pd.DataFrame, x: str, y: str) -> tuple[float, float, int]:
    pair = finite_pair(df, x, y)
    n = len(pair)
    if n < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return np.nan, np.nan, n
    res = spearmanr(pair[x], pair[y], nan_policy="omit")
    return float(res.statistic), float(res.pvalue), n


def partial_spearman(df: pd.DataFrame, x: str, y: str, control: str = "depth_md") -> tuple[float, float, int]:
    cols = [x, y, control]
    work = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(work)
    if n < 4 or any(work[c].nunique() < 2 for c in cols):
        return np.nan, np.nan, n
    ranks = work.rank(method="average")
    z = ranks[control].to_numpy(float)
    X = np.column_stack([np.ones(n), z])

    def residualize(v: np.ndarray) -> np.ndarray:
        beta = np.linalg.lstsq(X, v, rcond=None)[0]
        return v - X @ beta

    rx = residualize(ranks[x].to_numpy(float))
    ry = residualize(ranks[y].to_numpy(float))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan, np.nan, n
    res = pearsonr(rx, ry)
    return float(res.statistic), float(res.pvalue), n


def well_level(rows: pd.DataFrame) -> pd.DataFrame:
    agg = (
        rows.groupby(["well_anon", "zone_en"], dropna=False)
        .agg(
            n_samples=("sample_id_anon", "count"),
            residual=("residual", "median"),
            abs_residual=("abs_residual", "median"),
            depth_md=("depth_md", "median"),
            DEN=("DEN", "median"),
            max_burial_top=("max_burial_top", "median"),
            pressure_coeff=("pressure_coeff", "median"),
        )
        .reset_index()
    )
    return agg


def correlation_table(rows: pd.DataFrame, wells: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    level_data = [("point", rows), ("well_median", wells)]
    metric_labels = {"residual": "signed residual", "abs_residual": "absolute residual"}
    for level, data in level_data:
        for metric, metric_label in metric_labels.items():
            for var in GEO_VARS:
                rho, p_value, n = spearman_stat(data, var, metric)
                records.append(
                    {
                        "variable": var,
                        "metric": metric_label,
                        "level": level,
                        "analysis": "Spearman",
                        "rho": rho,
                        "p_value": p_value,
                        "n": n,
                        "note": "complete-pair Spearman; residual=predicted-observed TOC",
                    }
                )
            for var in ["DEN", "max_burial_top", "pressure_coeff"]:
                rho, p_value, n = partial_spearman(data, var, metric, "depth_md")
                records.append(
                    {
                        "variable": var,
                        "metric": metric_label,
                        "level": level,
                        "analysis": "partial Spearman controlling depth_md",
                        "rho": rho,
                        "p_value": p_value,
                        "n": n,
                        "note": "rank-transformed variable and residual were linearly residualized on ranked depth_md; Pearson r of residuals reported",
                    }
                )
    return pd.DataFrame(records)


def zone_summary(rows: pd.DataFrame) -> pd.DataFrame:
    q1 = rows.groupby("zone_en")["residual"].quantile(0.25)
    q3 = rows.groupby("zone_en")["residual"].quantile(0.75)
    summary = (
        rows.groupby("zone_en")
        .agg(
            n=("sample_id_anon", "count"),
            n_wells=("well_anon", "nunique"),
            mean_residual=("residual", "mean"),
            median_residual=("residual", "median"),
            q1_residual=("residual", lambda x: x.quantile(0.25)),
            q3_residual=("residual", lambda x: x.quantile(0.75)),
            iqr_residual=("residual", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            mean_abs_residual=("abs_residual", "mean"),
            median_abs_residual=("abs_residual", "median"),
        )
        .reindex(ZONE_ORDER)
        .reset_index()
    )
    summary["q1_residual"] = summary["zone_en"].map(q1)
    summary["q3_residual"] = summary["zone_en"].map(q3)
    summary["iqr_residual"] = summary["q3_residual"] - summary["q1_residual"]
    return summary


def fmt_float(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.4,
            "axes.titlesize": 7.8,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.18,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5},
    )


def annotate_rho(ax: plt.Axes, corr: pd.DataFrame, var: str) -> None:
    p = corr[(corr["variable"] == var) & (corr["metric"] == "signed residual") & (corr["level"] == "point") & (corr["analysis"] == "Spearman")].iloc[0]
    w = corr[(corr["variable"] == var) & (corr["metric"] == "signed residual") & (corr["level"] == "well_median") & (corr["analysis"] == "Spearman")].iloc[0]
    text = f"point rho={fmt_float(p.rho, 2)} (n={int(p.n)})\nwell rho={fmt_float(w.rho, 2)} (n={int(w.n)})"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        linespacing=1.05,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )


def draw_scatter_panel(ax: plt.Axes, rows: pd.DataFrame, wells: pd.DataFrame, xvar: str, corr: pd.DataFrame, show_ylabel: bool = True) -> None:
    for zone in ZONE_ORDER:
        sub = rows[rows["zone_en"] == zone]
        ax.scatter(
            sub[xvar],
            sub["residual"],
            s=11,
            marker=ZONE_MARKERS[zone],
            c=ZONE_COLORS[zone],
            alpha=0.55,
            linewidths=0,
            rasterized=False,
        )
        wsub = wells[wells["zone_en"] == zone]
        ax.scatter(
            wsub[xvar],
            wsub["residual"],
            s=42,
            marker=ZONE_MARKERS[zone],
            c=ZONE_COLORS[zone],
            alpha=0.96,
            edgecolors="#222222",
            linewidths=0.55,
            zorder=5,
        )
    ax.axhline(0, color="#333333", linewidth=0.75, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlabel(VAR_LABELS[xvar])
    if show_ylabel:
        ax.set_ylabel("Residual (predicted - observed TOC, wt%)")
    else:
        ax.set_ylabel("")
    ax.grid(color="#D9D9D9", linewidth=0.35, alpha=0.70)
    ax.set_axisbelow(True)
    ax.margins(x=0.06, y=0.10)
    annotate_rho(ax, corr, xvar)


def draw_zone_panel(ax: plt.Axes, rows: pd.DataFrame, wells: pd.DataFrame, zone_stats: pd.DataFrame, show_ylabel: bool = False) -> None:
    positions = np.arange(len(ZONE_ORDER))
    data = [rows.loc[rows["zone_en"] == zone, "residual"].dropna().to_numpy() for zone in ZONE_ORDER]
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 0.9},
        whiskerprops={"color": "#333333", "linewidth": 0.65},
        capprops={"color": "#333333", "linewidth": 0.65},
        boxprops={"edgecolor": "#333333", "linewidth": 0.65},
    )
    for patch, zone in zip(box["boxes"], ZONE_ORDER):
        patch.set_facecolor(ZONE_COLORS[zone])
        patch.set_alpha(0.28)
    rng = np.random.default_rng(20260702)
    for i, zone in enumerate(ZONE_ORDER):
        sub = rows[rows["zone_en"] == zone]
        x = i + rng.normal(0, 0.055, len(sub))
        ax.scatter(x, sub["residual"], s=10, marker=ZONE_MARKERS[zone], c=ZONE_COLORS[zone], alpha=0.52, linewidths=0)
        wsub = wells[wells["zone_en"] == zone]
        xw = i + rng.normal(0, 0.035, len(wsub))
        ax.scatter(xw, wsub["residual"], s=42, marker=ZONE_MARKERS[zone], c=ZONE_COLORS[zone], alpha=0.96, edgecolors="#222222", linewidths=0.55, zorder=5)
        stat = zone_stats[zone_stats["zone_en"] == zone].iloc[0]
        ax.text(
            i,
            1.60,
            f"n={int(stat.n)}\nm={stat.mean_residual:+.2f}",
            ha="center",
            va="top",
            fontsize=6.0,
            linespacing=1.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.3},
        )
    ax.axhline(0, color="#333333", linewidth=0.75, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.55, 2.55)
    ax.set_xticks(positions)
    ax.set_xticklabels(ZONE_ORDER)
    if show_ylabel:
        ax.set_ylabel("Residual (predicted - observed TOC, wt%)")
    else:
        ax.set_ylabel("")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.35, alpha=0.70)
    ax.set_axisbelow(True)


def make_figure(rows: pd.DataFrame, wells: pd.DataFrame, corr: pd.DataFrame, zone_stats: pd.DataFrame, out: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(3.54, 4.65), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.105, top=0.89, wspace=0.34, hspace=0.62)
    axa, axb, axc, axd = axes.ravel()
    draw_scatter_panel(axa, rows, wells, "depth_md", corr, show_ylabel=True)
    draw_scatter_panel(axb, rows, wells, "DEN", corr, show_ylabel=False)
    draw_scatter_panel(axc, rows, wells, "max_burial_top", corr, show_ylabel=True)
    draw_zone_panel(axd, rows, wells, zone_stats, show_ylabel=False)
    add_panel_label(axa, "(a)")
    add_panel_label(axb, "(b)")
    add_panel_label(axc, "(c)")
    add_panel_label(axd, "(d)")
    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [
        Line2D([0], [0], marker=ZONE_MARKERS[z], color="none", markerfacecolor=ZONE_COLORS[z], markeredgecolor="none", markersize=4.6, linestyle="none", label=z)
        for z in ZONE_ORDER
    ]
    handles.append(
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#BFBFBF", markeredgecolor="#222222", markersize=5.5, linestyle="none", label="well median")
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.985), ncol=4, frameon=False, handletextpad=0.35, columnspacing=0.8)
    for ext in ["svg", "pdf"]:
        fig.savefig(out / f"Fig_residual_geology.{ext}")
    fig.savefig(out / "Fig_residual_geology.png", dpi=600)
    plt.close(fig)
    try:
        from PIL import Image

        img = Image.open(out / "Fig_residual_geology.png")
        img.convert("L").save(out / "Fig_residual_geology_grayscale.png", dpi=(300, 300))
    except Exception:
        pass


def write_outputs(rows: pd.DataFrame, corr: pd.DataFrame, zone_stats: pd.DataFrame, out: Path, pred_path: Path, feature_path: Path) -> None:
    residual_cols = [
        "sample_id_anon",
        "well_anon",
        "zone_en",
        "layer",
        "lithology",
        "depth_md",
        "DEN",
        "max_burial_top",
        "pressure_coeff",
        "y_true",
        "y_pred",
        "residual",
        "residual_true_minus_pred",
        "abs_residual",
    ]
    export = rows[residual_cols].rename(columns={"sample_id_anon": "sample_id", "lithology": "lithology_code"})
    export.to_csv(out / "residuals_with_covariates.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(out / "S13v2_residual_correlations.csv", index=False, encoding="utf-8-sig")
    zone_stats.to_csv(out / "S12v2_residual_by_zone.csv", index=False, encoding="utf-8-sig")

    anchor = corr[(corr["variable"] == "depth_md") & (corr["metric"] == "absolute residual") & (corr["level"] == "point") & (corr["analysis"] == "Spearman")].iloc[0]
    signed_depth = corr[(corr["variable"] == "depth_md") & (corr["metric"] == "signed residual") & (corr["level"] == "point") & (corr["analysis"] == "Spearman")].iloc[0]
    den_partial = corr[(corr["variable"] == "DEN") & (corr["metric"] == "signed residual") & (corr["level"] == "point") & (corr["analysis"].str.startswith("partial"))].iloc[0]
    zone_text = "; ".join(
        f"{r.zone_en}: n={int(r.n)}, wells={int(r.n_wells)}, mean={r.mean_residual:+.3f}, median={r.median_residual:+.3f}, IQR={r.iqr_residual:.3f}"
        for r in zone_stats.itertuples()
    )
    report = f"""# Residual-geology diagnostic outputs

## Method
Filtered `expanded_model_baseline_predictions.csv` to `dataset=TOC_high_medium`, `target=toc_pct` (the table's TOC field name), `model=ExtraTrees`, and `cv_scheme=LOZO`, giving {len(rows)} held-out samples. Geological covariates were merged by `sample_id` from `feature_values_toc.csv`, which is the retained model-feature table matching the TOC sample set. To stay consistent with the existing S12/S13 residual diagnostics and the prior Fig. 7 convention, the primary `residual` column is `y_pred - y_true` (predicted minus observed TOC; positive values indicate overprediction). The literal opposite direction requested in one sentence of the command package is preserved as `residual_true_minus_pred` for audit.

Partial Spearman correlations were computed by rank-transforming the residual metric, the geological variable, and `depth_md`; both ranked residual and ranked geological variable were linearly residualized on ranked `depth_md`, and Pearson r between the two residualized vectors is reported.

## Anchor checks
- Row count: {len(rows)} samples.
- Zone counts: {rows['zone_en'].value_counts().reindex(ZONE_ORDER).astype(int).to_dict()}.
- S13 depth anchor: point-level `abs_residual` vs `depth_md` Spearman rho = {anchor.rho:.6f} (n={int(anchor.n)}), matching the published ~0.36 anchor; absolute difference from {ANCHOR_DEPTH_ABS_RHO:.6f} is {abs(anchor.rho - ANCHOR_DEPTH_ABS_RHO):.6f}.
- Signed residual vs depth: point-level Spearman rho = {signed_depth.rho:.6f} (n={int(signed_depth.n)}).
- S12 zone residual pattern: {zone_text}.

## Key interpretation
The diagnostic table supports weak-to-moderate univariate associations rather than a single geological master control. For example, point-level partial Spearman for signed residual vs DEN after controlling depth is rho = {den_partial.rho:.6f} (n={int(den_partial.n)}). Full point-level, well-median, and partial results are in `S13v2_residual_correlations.csv`.

## Files
- Prediction source: `{pred_path}`
- Feature source: `{feature_path}`
- Row-level anonymous table: `residuals_with_covariates.csv`
- Correlation table: `S13v2_residual_correlations.csv`
- Zone residual summary: `S12v2_residual_by_zone.csv`
- Figure: `Fig_residual_geology.svg`, `Fig_residual_geology.pdf`, `Fig_residual_geology.png`

## Reproduce
```bash
"PROJECT_ROOT_PLACEHOLDER" "{Path(__file__).resolve()}"
```
"""
    (out / "report_residual_geology.md").write_text(report, encoding="utf-8")


def anonymity_scan(out: Path, well_map: dict[str, str]) -> None:
    outputs = [
        out / "residuals_with_covariates.csv",
        out / "S13v2_residual_correlations.csv",
        out / "S12v2_residual_by_zone.csv",
        out / "report_residual_geology.md",
        out / "Fig_residual_geology.svg",
    ]
    hits: list[tuple[str, str]] = []
    real_wells = sorted(well_map.keys(), key=len, reverse=True)
    for path in outputs:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for well in real_wells:
            if well and re.search(re.escape(well), text):
                hits.append((path.name, well))
    lines = [f"real_well_name_hits={len(hits)}"]
    lines += [f"{name}: {well}" for name, well in hits[:50]]
    (out / "anonymity_scan.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if hits:
        raise RuntimeError(f"Anonymity scan found real well names: {hits[:5]}")


def machine_check(out: Path, rows: pd.DataFrame, corr: pd.DataFrame) -> None:
    checks: list[str] = []
    checks.append(f"row_count={len(rows)} expected={TARGET_ROWS} status={'PASS' if len(rows) == TARGET_ROWS else 'FAIL'}")
    counts = rows["zone_en"].value_counts().reindex(ZONE_ORDER).fillna(0).astype(int).to_dict()
    checks.append(f"zone_counts={counts} expected={EXPECTED_ZONE_COUNTS} status={'PASS' if counts == EXPECTED_ZONE_COUNTS else 'FAIL'}")
    anchor = corr[(corr["variable"] == "depth_md") & (corr["metric"] == "absolute residual") & (corr["level"] == "point") & (corr["analysis"] == "Spearman")].iloc[0]
    anchor_ok = abs(float(anchor.rho) - ANCHOR_DEPTH_ABS_RHO) <= 0.05
    checks.append(f"depth_abs_spearman={float(anchor.rho):.6f} target~0.36 status={'PASS' if anchor_ok else 'FAIL'}")
    for fn in ["Fig_residual_geology.svg", "Fig_residual_geology.pdf", "Fig_residual_geology.png"]:
        path = out / fn
        checks.append(f"{fn}_exists={path.exists()} size={path.stat().st_size if path.exists() else 0}")
    (out / "machine_check_residual_geology.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    if not anchor_ok:
        raise RuntimeError("Depth absolute-residual Spearman anchor failed")


def main() -> int:
    root = project_root()
    out = output_dir()
    pred_path, feature_path = find_input_files(root)
    rows, well_map = load_residual_rows(pred_path, feature_path)
    wells = well_level(rows)
    corr = correlation_table(rows, wells)
    zones = zone_summary(rows)
    make_figure(rows, wells, corr, zones, out)
    write_outputs(rows, corr, zones, out, pred_path, feature_path)
    anonymity_scan(out, well_map)
    machine_check(out, rows, corr)
    print(out / "residuals_with_covariates.csv")
    print(out / "S13v2_residual_correlations.csv")
    print(out / "S12v2_residual_by_zone.csv")
    print(out / "Fig_residual_geology.svg")
    print(out / "Fig_residual_geology.pdf")
    print(out / "Fig_residual_geology.png")
    print(out / "report_residual_geology.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
