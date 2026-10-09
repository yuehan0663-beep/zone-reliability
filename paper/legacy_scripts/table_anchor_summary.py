# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_derived"

def rows(name):
    with (DATA / name).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def pick(rows_, **conds):
    for row in rows_:
        if all(row.get(k) == v for k, v in conds.items()):
            return row
    raise SystemExit(f"row not found: {conds}")

def main():
    s24c = pick(rows("Table_S24c_schmoker_baseline_sensitivity.csv"), zone_en="North", variant="baseline")
    s19 = pick(rows("Table_S19_rejection_weight_sensitivity.csv"), target="total_gas_m3_t", cv_scheme="LOWO", method="global", weight_scheme="current_0.55_0.25_0.10_0.10")
    s1_toc = pick(rows("Table_S1_full_model_rank.csv"), dataset="TOC_high_medium", target="toc_pct", model="ExtraTrees", cv_scheme="LOZO")
    s1_gas = pick(rows("Table_S1_full_model_rank.csv"), dataset="total_gas_high_medium", target="total_gas_m3_t", model="RF", cv_scheme="LOZO")
    s21 = [r for r in rows("Table_S21b_adaptive_rejection_results.csv") if r.get("target_role") == "gas_main"]
    print("north_toc_s24c_pct", s24c["relative_improvement_pct"], s24c["improvement_ci_low_pct"], s24c["improvement_ci_high_pct"])
    print("gas_lowo_fixed20_pct", s19["relative_improvement_pct"])
    print("lozo_toc_r2", s1_toc["pooled_R2"])
    print("lozo_total_gas_r2", s1_gas["pooled_R2"])
    print("adaptive_rejection_gas_wins", sum(1 for r in s21 if r.get("win") == "True"))

if __name__ == "__main__":
    main()
