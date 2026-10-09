# Requires the confidential raw dataset, available from the corresponding author on reasonable request. Paths de-identified; seeds/params retained for reproducibility.
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_derived"

def as_float(value):
    try:
        return float(value)
    except Exception:
        return None

def main():
    path = DATA / "oof_predictions.csv"
    if not path.exists():
        raise SystemExit("oof_predictions.csv is absent")
    groups = defaultdict(lambda: [0, 0.0])
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y = as_float(row.get("y_true") or row.get("observed") or row.get("truth"))
            pred = as_float(row.get("y_pred") or row.get("prediction") or row.get("pred"))
            if y is None or pred is None:
                continue
            key = (row.get("dataset", ""), row.get("target", ""), row.get("cv_scheme", ""), row.get("model", ""))
            groups[key][0] += 1
            groups[key][1] += abs(y - pred)
    print("dataset,target,cv_scheme,model,n,mae")
    for key, (n, err) in sorted(groups.items()):
        if n:
            print(",".join(key + (str(n), f"{err/n:.10g}")))

if __name__ == "__main__":
    main()
