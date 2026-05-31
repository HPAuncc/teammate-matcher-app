"""
generate_logo.py — builds the Teammate Matcher app mark.
=========================================================
Draws a small "connected team" glyph (three linked nodes on a rounded navy
tile) and saves it as a transparent PNG used for the browser favicon and the
hero logo. Run once; the PNG is committed so the app has no runtime dependency
on matplotlib for branding.

    python scripts/generate_logo.py
"""

import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

NAVY = "#1E3A5F"
BLUE = "#2563EB"
WHITE = "#FFFFFF"

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def build(path: str, px: int = 512) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=px / 5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_aspect("equal")

    # Rounded tile
    ax.add_patch(
        FancyBboxPatch(
            (0.10, 0.10), 0.80, 0.80,
            boxstyle="round,pad=0,rounding_size=0.22",
            linewidth=0, facecolor=NAVY,
        )
    )

    # Three team nodes + connecting lines (a small network)
    pts = [(0.50, 0.70), (0.33, 0.39), (0.67, 0.39)]
    for a, b in itertools.combinations(pts, 2):
        ax.plot(
            [a[0], b[0]], [a[1], b[1]],
            color=WHITE, lw=6, alpha=0.45, solid_capstyle="round", zorder=1,
        )
    for (x, y) in pts:
        ax.add_patch(Circle((x, y), 0.115, facecolor=WHITE, edgecolor=BLUE, lw=5, zorder=2))

    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    build(os.path.join(ASSETS, "logo.png"), px=512)
