# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_derived"

def read_manifest():
    with (ROOT / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    rows = read_manifest()
    missing = []
    for row in rows:
        package_path = row.get("package_path", "")
        if package_path and package_path != "not_in_code_package":
            if not (ROOT / package_path).exists():
                missing.append(package_path)
    si_tables = sorted(DATA.glob("Table_S*.csv"))
    print(f"manifest_rows={len(rows)}")
    print(f"si_tables={len(si_tables)}")
    if missing:
        raise SystemExit("missing package paths: " + ", ".join(missing))
    print("PACKAGE INVENTORY PASS")

if __name__ == "__main__":
    main()
