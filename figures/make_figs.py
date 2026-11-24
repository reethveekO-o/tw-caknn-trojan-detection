"""Progression figures for the conference and journal papers (IEEE single column)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
F_COL, AP_COL, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"


def panel(ax, rows, title, xmax=100, highlight=()):
    """rows: list of (label, F, AP); drawn top-to-bottom in study order."""
    n = len(rows)
    y = np.arange(n)[::-1]
    h = 0.36
    for yi, (lab, f, ap) in zip(y, rows):
        ax.barh(yi + h / 2, f, h, color=F_COL, edgecolor="white", linewidth=0.6)
        ax.barh(yi - h / 2, ap, h, color=AP_COL, edgecolor="white", linewidth=0.6, hatch="////")
        ax.text(f + 0.8, yi + h / 2, f"{f:.1f}", va="center", fontsize=7, color=INK)
        ax.text(ap + 0.8, yi - h / 2, f"{ap:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    for t, (lab, _, _) in zip(ax.get_yticklabels(), rows):
        if lab in highlight:
            t.set_fontweight("bold")
    ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left", color=INK)
    ax.xaxis.grid(True, color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)


def legend(fig):
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=F_COL, label="F-measure"),
                        Patch(facecolor=AP_COL, hatch="////", edgecolor="white", label="Average precision")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))


# ---------------------------------------------------------------- conference
conf_a = [  # all 15 circuits, fixed contamination C, in study order
    ("Stacking LOF + CBLOF [8]", 61.5, 70.4),
    ("LOF", 60.4, 69.4),
    ("CBLOF", 60.9, 70.0),
    ("PSO-DBSCAN", 45.7, 57.3),
    ("BIRCH", 60.0, 69.0),
    ("PSO-DBSCAN + BIRCH", 58.3, 66.6),
    ("PSO-DBSCAN + CBLOF", 59.3, 67.2),
    ("LOF + BIRCH", 60.5, 69.3),
    ("Attention stacking (circuit held out)", 51.5, 51.4),
    ("Attention stacking (family held out)", 13.8, 18.1),
    ("CA kNN-50 (proposed)", 65.0, 77.0),
]
conf_b = [  # nine small circuits (TWEC-TD did not finish on the large ones)
    ("Stacking LOF + CBLOF [8]", 65.1, 67.8),
    ("TWEC-TD consensus clusters", 68.3, 80.3),
    ("Tensor-refined CA + kNN-50", 70.2, 78.5),
    ("Weighted CA + kNN-50", 70.1, 80.6),
    ("CA kNN-50 (proposed)", 70.7, 78.1),
]
fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 6.2), gridspec_kw={"height_ratios": [len(conf_a), len(conf_b)]})
panel(a, conf_a, "(a) All 15 circuits", highlight=("CA kNN-50 (proposed)",))
panel(b, conf_b, "(b) Nine small circuits", highlight=("CA kNN-50 (proposed)",))
b.set_xlabel("Score (%)")
legend(fig)
fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.0)
fig.savefig("fig_conf_progression.pdf")
fig.savefig("fig_conf_progression.png", dpi=300)

# ---------------------------------------------------------------- journal
jour = [  # all 15 circuits; last row uses the automatic flag count instead of fixed C
    ("PSO-DBSCAN", 45.7, 57.3),
    ("BIRCH", 60.0, 69.0),
    ("Stacking LOF + CBLOF [8]", 61.0, 69.9),
    ("CA kNN-50", 65.0, 77.0),
    ("+ 1-hop smoothing", 66.1, 78.0),
    ("+ hybrid Louvain/1-hop smoothing", 70.4, 85.3),
    ("+ testability weight", 71.7, 92.7),
    ("+ automatic flag count (TW-CA-kNN)", 85.1, 92.7),
]
fig, ax = plt.subplots(figsize=(3.5, 3.6))
panel(ax, jour, "", highlight=("+ automatic flag count (TW-CA-kNN)",))
ax.set_xlabel("Score (%)")
legend(fig)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("fig_journal_progression.pdf")
fig.savefig("fig_journal_progression.png", dpi=300)
print("ok")
