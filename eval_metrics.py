import os
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
preds_dir = "preds_tempa_mscxrt_short"  # Folder containing per-example prediction CSVs

# ---------------------------------------------------------
# Load all individual prediction CSVs
# ---------------------------------------------------------
csv_files = sorted([
    os.path.join(preds_dir, f)
    for f in os.listdir(preds_dir)
    if f.endswith(".csv")
])

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in '{preds_dir}'")

dfs = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        print(f"[Warning] Could not read {f}: {e}")

data = pd.concat(dfs, ignore_index=True)
print(f"📄 Loaded {len(data)} total predictions from {len(csv_files)} CSV files.")

# ---------------------------------------------------------
# Normalize and clean labels
# ---------------------------------------------------------
data["true_comparison"] = data["true_comparison"].astype(str).str.strip().str.lower()
data["predicted_comparison"] = data["predicted_comparison"].astype(str).str.strip().str.lower()
data["disease_name"] = data["disease_name"].astype(str).str.strip().str.lower()

# ---------------------------------------------------------
# Define mapping (CheXagent → ground truth space)
# ---------------------------------------------------------
mapping = {
    "(a) worsening": "worsening",
    "worsening": "worsened",
    "(b) stable": "stable",
    "stable": "no change",
    "(c) improving": "improving",
    "improving": "improved",
    "error": "error"
}

data["predicted_mapped"] = data["predicted_comparison"]#.map(mapping).fillna("unknown")

# ---------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------
total = len(data)
num_errors = (data["predicted_mapped"] == "error").sum()
num_unknowns = (data["predicted_mapped"] == "unknown").sum()

valid_data = data[~data["predicted_mapped"].isin(["error", "unknown"])].copy()
valid_total = len(valid_data)

# ---------------------------------------------------------
# Per-class accuracy and breakdown of wrong predictions
# ---------------------------------------------------------
print("\n📊 Per-class Breakdown:")
for label in sorted(valid_data["true_comparison"].unique()):
    subset = valid_data[valid_data["true_comparison"] == label]
    n = len(subset)
    correct = (subset["true_comparison"] == subset["predicted_mapped"]).sum()
    acc = correct / n if n > 0 else 0.0

    print(f"  • {label:<10} — {acc*100:5.2f}% accuracy ({correct}/{n})")

    # Misclassifications
    wrong = subset[subset["true_comparison"] != subset["predicted_mapped"]]
    if len(wrong) > 0:
        counts = wrong["predicted_mapped"].value_counts()
        print("     ↳ Wrong predictions:")
        for pred_label, count in counts.items():
            print(f"       - {pred_label:<12} : {count}")
    else:
        print("     ↳ All correct ✅")

# ---------------------------------------------------------
# Accuracy by disease type
# ---------------------------------------------------------
print("\n🧬 Accuracy by Disease Type:")
disease_groups = valid_data.groupby("disease_name")

for disease, group in disease_groups:
    n = len(group)
    correct = (group["true_comparison"] == group["predicted_mapped"]).sum()
    acc = correct / n if n > 0 else 0.0
    print(f"  • {disease:<25} — {acc*100:5.2f}% ({correct}/{n})")

# ---------------------------------------------------------
# Correct overall accuracy calculation
# ---------------------------------------------------------
overall_correct = (valid_data["true_comparison"] == valid_data["predicted_mapped"]).sum()
accuracy = overall_correct / valid_total if valid_total > 0 else 0.0

# ---------------------------------------------------------
# Error analysis
# ---------------------------------------------------------
error_data = data[data["predicted_mapped"] == "error"]
print("\n🚨 Error Analysis:")
print(f"Total 'error' predictions: {num_errors}")

if num_errors > 0:
    print("\nBy disease type:")
    err_by_disease = error_data["disease_name"].value_counts()
    for disease, count in err_by_disease.items():
        print(f"  - {disease:<25} : {count}")

    print("\nBy true comparison type:")
    err_by_truth = error_data["true_comparison"].value_counts()
    for truth, count in err_by_truth.items():
        print(f"  - {truth:<10} : {count}")
else:
    print("  No 'error' predictions found ✅")

# ---------------------------------------------------------
# Print overall summary
# ---------------------------------------------------------
print("\n📈 Overall Summary:")
print(f"Total examples: {total}")
print(f"Valid predictions (excluding 'error'/'unknown'): {valid_total}")
print(f"Correct matches: {overall_correct}")
print(f"Exact match accuracy: {accuracy * 100:.2f}%")

print(f"\n⚠️  Errors: {num_errors}")
print(f"❓ Unknown labels (unmapped): {num_unknowns}")

# ---------------------------------------------------------
# Print label sets
# ---------------------------------------------------------
print("\n🩺 Predicted label set (raw):")
for label in sorted(set(data["predicted_comparison"])):
    print(f"  - {label}")

print("\n🩺 Predicted label set (mapped to true space):")
for label in sorted(set(data["predicted_mapped"])):
    print(f"  - {label}")

print("\n🩺 True label set:")
for label in sorted(set(data["true_comparison"])):
    print(f"  - {label}")

# ---------------------------------------------------------
# Save combined data
# ---------------------------------------------------------
output_path = "mscxrt_errors_chexagent.csv"
data.to_csv(output_path, index=False)
print(f"\n✅ Combined results with mappings saved to: {output_path}")

