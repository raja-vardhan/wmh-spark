import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run_dir = Path(__file__).resolve().parents[1]
summary = json.loads((run_dir / "benchmarks" / "summary.json").read_text())
baselines = {b["name"]: b["mean_dsc"] for b in summary["baseline_summaries"]}

names = ["all_zero", "all_positive", "spatial_prior_only", "RF model (ours)"]
values = [
    baselines["all_zero"],
    baselines["all_positive"],
    baselines["spatial_prior_only"],
    summary["mean_dsc"],
]
colors = ["#cccccc", "#cccccc", "#7aaed8", "#d9531e"]

fig, ax = plt.subplots(figsize=(6.4, 4.0))
xs = np.arange(len(names))
bars = ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.7)
ax.set_xticks(xs)
ax.set_xticklabels(names, rotation=12)
ax.set_ylabel("Mean Dice (n = 10 test subjects)")
ax.set_ylim(0, max(values) * 1.25)
ax.set_title("WMH segmentation: model vs trivial baselines")
for bar, v in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + max(values) * 0.02,
        f"{v:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.grid(axis="y", linestyle=":", alpha=0.5)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("baseline_comparison.png"), dpi=130)
print("wrote baseline_comparison.png")
