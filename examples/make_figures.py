"""Regenerate the README figures from synthetic data.

pip install -e ".[viz]"
python examples/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

from poliseye import area_typologies, synthetic_presence
from poliseye.mobility import daily_profile

OUT = Path(__file__).parents[1] / "docs" / "assets"
COLORS = {
    "workplace": "#2563eb",
    "residential": "#16a34a",
    "leisure": "#f59e0b",
    "transport hub": "#dc2626",
}

wide = synthetic_presence(n_days=28)
typ = area_typologies(wide)
prof = daily_profile(wide)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
hours = [s / 4 for s in range(96)]
for ax, day_type, title in zip(
    axes,
    ["business", "non_business"],
    ["Business days", "Weekends and holidays"],
    strict=True,
):
    p = prof.xs(day_type)
    for cell in wide.columns:
        label = typ.loc[cell, "typology"]
        ax.plot(
            hours,
            p[cell] / prof.xs("business")[cell].mean(),
            lw=2.2,
            color=COLORS.get(label, "#555"),
            label=label,
        )
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=8)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("presence / business-day mean")
axes[1].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle(
    "Typical daily rhythm by area typology (synthetic illustration)",
    x=0.01,
    ha="left",
    fontsize=12,
    fontweight="bold",
)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "typology_profiles.png", dpi=160)
print("wrote", OUT / "typology_profiles.png")
