"""Plot training/validation loss curves from the val_metrics.csv that
resume_train.py writes.

resume_train.py logs one row per epoch:
    epoch,val_total,val_global,val_local,val_mlm
This script turns that into PNG loss curves. Run it AFTER (or during) a
training run — it doesn't touch training and needs no GPU.

Usage:
    python biovilt/plot_metrics.py                      # defaults: logs/val_metrics.csv -> logs/
    python biovilt/plot_metrics.py --csv logs/val_metrics.csv --out-dir logs
    python biovilt/plot_metrics.py --show               # also open an interactive window

Outputs (in --out-dir):
    val_loss_total.png      total validation loss vs epoch
    val_loss_components.png global / local / mlm losses on one axis
"""
import argparse
import os
import sys

import pandas as pd

import matplotlib
# Use a non-interactive backend by default so it works over SSH / headless GCP.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="logs/val_metrics.csv",
                    help="Path to val_metrics.csv (default: logs/val_metrics.csv).")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write PNGs (default: same dir as --csv).")
    ap.add_argument("--show", action="store_true",
                    help="Also display the figures in an interactive window.")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"Metrics file not found: {args.csv}\n"
                 f"(It is written by resume_train.py to LOG_DIR/val_metrics.csv.)")

    df = pd.read_csv(args.csv)
    if df.empty:
        sys.exit(f"{args.csv} has no rows yet — let training run at least one epoch.")

    expected = {"epoch", "val_total", "val_global", "val_local", "val_mlm"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"{args.csv} missing columns: {sorted(missing)}. "
                 f"Found: {list(df.columns)}")

    out_dir = args.out_dir or (os.path.dirname(args.csv) or ".")
    os.makedirs(out_dir, exist_ok=True)

    if args.show:
        matplotlib.use("TkAgg", force=True)

    # ---- 1) Total validation loss ----
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["epoch"], df["val_total"], marker="o", color="tab:blue",
             label="val_total")
    best_idx = df["val_total"].idxmin()
    ax1.scatter([df.loc[best_idx, "epoch"]], [df.loc[best_idx, "val_total"]],
                color="red", zorder=5,
                label=f"best (epoch {int(df.loc[best_idx, 'epoch'])}, "
                      f"{df.loc[best_idx, 'val_total']:.4f})")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation loss")
    ax1.set_title("Total validation loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    p1 = os.path.join(out_dir, "val_loss_total.png")
    fig1.tight_layout()
    fig1.savefig(p1, dpi=150)

    # ---- 2) Loss components ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for col, color in [("val_global", "tab:green"),
                       ("val_local", "tab:orange"),
                       ("val_mlm", "tab:purple")]:
        ax2.plot(df["epoch"], df[col], marker="o", label=col, color=color)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation loss")
    ax2.set_title("Validation loss components (global / local / mlm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    p2 = os.path.join(out_dir, "val_loss_components.png")
    fig2.tight_layout()
    fig2.savefig(p2, dpi=150)

    print(f"[plot] {len(df)} epochs read from {args.csv}")
    print(f"[plot] wrote {p1}")
    print(f"[plot] wrote {p2}")
    print(f"[plot] best epoch = {int(df.loc[best_idx, 'epoch'])} "
          f"(val_total = {df.loc[best_idx, 'val_total']:.4f})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
