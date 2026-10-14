"""
Redraw the constrained-tagging FSM (paper Fig. 3) with text that is legible at column width.

The thesis figure was a 746 x 463 px screenshot of a diagram tool. Scaled to the 3.03 in column, its state names
printed at about 4 pt and its edge labels at about 3 pt. This draws the same machine directly at print size, so the
font sizes below are (up to the final 0.89 scale) the sizes on paper.

Same states, edges, labels and colours as the thesis version (thesis-latex/figures/FSM_diagram.png). The two
corner icons that marked start and end are replaced by the standard notation, which survives the small size: an
incoming "start" arrow on OUTSIDE and a double border on END OF STREAM.

Usage
-----
    python scripts/analysis/plot_fsm_diagram.py            # writes paper/figures/fsm_diagram.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "figures" / "fsm_diagram.png"

# Drawn at 3.4 in and included at \columnwidth (3.03 in), so everything prints at 0.89 of these sizes.
WIDTH, HEIGHT = 3.4, 2.22
STATE_PT, LABEL_PT = 9.0, 7.0
LINE = 0.9  # line width in pt

# name: (centre x, centre y, width, height, fill, text colour) -- in inches of the drawing
STATES = {
    "eos":   (0.33, 1.62, 0.56, 0.40, "#6e6e6e", "white"),
    "out":   (1.50, 1.62, 0.74, 0.32, "#a9d8e8", "black"),
    "tag":   (2.95, 1.62, 0.80, 0.32, "#f8d5c0", "black"),
    "close": (1.40, 0.62, 0.86, 0.32, "#f0c4f2", "black"),
    "text":  (2.95, 0.62, 0.80, 0.32, "#c3f5d3", "black"),
}
NAMES = {"eos": "End of\nStream", "out": "Outside", "tag": "Tag Block", "close": "Span Close", "text": "Span Text"}


def edge(s, side, offset=0.0):
    """Point on one side of a state box; `offset` slides along that side from its centre."""
    x, y, w, h = STATES[s][:4]
    return {"top": (x + offset, y + h / 2), "bottom": (x + offset, y - h / 2),
            "left": (x - w / 2, y + offset), "right": (x + w / 2, y + offset)}[side]


def arrow(ax, points):
    """Orthogonal polyline through `points`, arrowhead on the last segment."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color="black", lw=LINE, solid_capstyle="butt")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color="black", lw=LINE,
                                shrinkA=0, shrinkB=0))


def loop(ax, s, side, x0, x1, reach):
    """Self-loop leaving and re-entering the top or bottom of a state."""
    sign = 1 if side == "top" else -1
    a, b = edge(s, side, x0), edge(s, side, x1)
    arrow(ax, [a, (a[0], a[1] + sign * reach), (b[0], b[1] + sign * reach), b])


def label(ax, x, y, text, ha="center", va="center"):
    ax.text(x, y, text, fontsize=LABEL_PT, fontweight="bold", ha=ha, va=va, linespacing=1.05)


def main():
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.axis("off")

    for s, (x, y, w, h, fill, fg) in STATES.items():
        rounding = h / 2 if s in ("eos", "out") else 0.05   # the two stadium shapes of the original
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                    facecolor=fill, edgecolor="black", lw=LINE))
        if s == "eos":  # accepting state: inner border
            ax.add_patch(FancyBboxPatch((x - w / 2 + 0.035, y - h / 2 + 0.035), w - 0.07, h - 0.07,
                                        boxstyle=f"round,pad=0,rounding_size={rounding - 0.035}",
                                        facecolor="none", edgecolor="white", lw=LINE * 0.8))
        # The two-line END OF STREAM sits inside a double border, so it is set one point smaller.
        ax.text(x, y, NAMES[s], fontsize=STATE_PT - (1 if s == "eos" else 0), fontweight="bold", color=fg, ha="center", va="center",
                linespacing=1.0)

    # start -> OUTSIDE, entering the flat part of its top; the copy loop sits to its right
    start = edge("out", "top", -0.20)
    arrow(ax, [(start[0], 2.0), start])
    label(ax, start[0] - 0.04, 1.93, "start", ha="right")
    loop(ax, "out", "top", 0.0, 0.20, 0.12)
    label(ax, 1.60, 1.92, "Copy input\ntoken", va="bottom")

    # OUTSIDE -> END OF STREAM; the label fits between the two boxes, under the arrow
    arrow(ax, [edge("out", "left"), edge("eos", "right")])
    label(ax, 0.88, 1.56, "All input\nconsumed", va="top")

    # OUTSIDE -> TAG BLOCK, and the label loop
    arrow(ax, [edge("out", "right"), edge("tag", "left")])
    label(ax, 2.21, 1.58, "Span open\nfirst token", va="top")
    loop(ax, "tag", "top", -0.10, 0.10, 0.12)
    label(ax, 2.95, 1.92, "Emit label\nsequence", va="bottom")

    # TAG BLOCK -> SPAN TEXT
    arrow(ax, [edge("tag", "bottom"), edge("text", "top")])
    label(ax, 2.89, 1.12, "Label block\ncomplete", ha="right")

    # SPAN TEXT -> SPAN CLOSE, and the two loops on the bottom row
    arrow(ax, [edge("text", "left"), edge("close", "right")])
    label(ax, 2.19, 0.66, "Close tag\nfirst token", va="bottom")
    loop(ax, "text", "bottom", -0.10, 0.10, 0.12)
    label(ax, 2.95, 0.30, "Copy span\ntoken", va="top")
    loop(ax, "close", "bottom", -0.10, 0.10, 0.12)
    label(ax, 1.40, 0.30, "Emit close\nsequence", va="top")

    # SPAN CLOSE -> OUTSIDE, a straight climb inside the flat parts of both boxes
    arrow(ax, [edge("close", "top"), (STATES["close"][0], edge("out", "bottom")[1])])
    label(ax, 1.34, 1.12, "Close tag\ncomplete", ha="right")

    fig.savefig(OUT, dpi=600, facecolor="white")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
