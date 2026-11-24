"""Slide-ready line chart: F-measure and AP across every method tried (16:9)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
F_COL, AP_COL, INK = "#2a78d6", "#eb6834", "#0b0b0b"

steps = [
    ("LOF", 60.4, 69.4),
    ("CBLOF", 60.9, 70.0),
    ("Stacking\nLOF+CBLOF", 61.5, 70.4),
    ("PSO-\nDBSCAN", 45.7, 57.3),
    ("BIRCH", 60.0, 69.0),
    ("PSO-\nDBSCAN\n+BIRCH", 58.3, 66.6),
    ("PSO-\nDBSCAN\n+CBLOF", 59.3, 67.2),
    ("LOF\n+BIRCH", 60.5, 69.3),
    ("Attention\nstacking", 51.5, 51.4),
    ("Attention\nstacking\n(KLOCO-CV)", 13.8, 18.1),
    ("TWEC-TD", 70.2, 78.5),
    ("CA kNN-50", 65.0, 77.0),
    ("+1-hop\nsmoothing", 66.1, 78.0),
    ("+hybrid\nsmoothing", 70.4, 85.3),
    ("+SCOAP\nweight", 71.7, 92.7),
    ("TW-CA-kNN", 85.1, 92.7),
]
x = list(range(len(steps)))
F = [s[1] for s in steps]
AP = [s[2] for s in steps]
last = len(steps) - 1

fig, ax = plt.subplots(figsize=(13.33, 6.0))
ax.plot(x, AP, color=AP_COL, lw=2, marker="s", ms=8, mec="white", mew=1.2, label="Average precision", zorder=3)
ax.plot(x, F, color=F_COL, lw=2, marker="o", ms=8, mec="white", mew=1.2, label="F-measure", zorder=3)

off_f = {8: (16, -4), 10: (18, -6)}
off_a = {8: (-24, -14), 9: (20, 0)}
for i in x[:-1]:
    ax.annotate(f"{F[i]:.1f}", (i, F[i]), textcoords="offset points", xytext=off_f.get(i, (0, -16)),
                ha="center", fontsize=9.5, color=F_COL, fontweight="bold")
    ax.annotate(f"{AP[i]:.1f}", (i, AP[i]), textcoords="offset points", xytext=off_a.get(i, (0, 9)),
                ha="center", fontsize=9.5, color=AP_COL, fontweight="bold")

# highlight the final, best result
ax.axvspan(last - 0.4, last + 0.6, color="#fff1c2", zorder=0)
ax.plot([last], [F[last]], marker="o", ms=15, color=F_COL, mec="white", mew=2, zorder=4)
ax.plot([last], [AP[last]], marker="s", ms=15, color=AP_COL, mec="white", mew=2, zorder=4)
ax.annotate(f"F {F[last]:.1f}", (last, F[last]), textcoords="offset points", xytext=(0, -26),
            ha="center", fontsize=12.5, color=F_COL, fontweight="bold")
ax.annotate(f"AP {AP[last]:.1f}", (last, AP[last]), textcoords="offset points", xytext=(0, 14),
            ha="center", fontsize=12.5, color=AP_COL, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in steps], fontsize=9.5, linespacing=1.05)
ax.get_xticklabels()[last].set_fontweight("bold")
ax.get_xticklabels()[last].set_fontsize(10)
ax.set_ylim(0, 105)
ax.set_yticks(range(0, 101, 20))
ax.set_ylabel("Score (%)", color=INK)
ax.yaxis.grid(True, color="#e4e3df", lw=0.8)
ax.set_axisbelow(True)
ax.set_xlim(-0.5, last + 0.7)
ax.legend(loc="upper left", frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig("slide_progression_line.png", dpi=200)
print("ok")
