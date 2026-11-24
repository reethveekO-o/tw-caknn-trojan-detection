from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.family":"serif","font.serif":["Liberation Serif"],"font.size":8,
    "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,"pdf.fonttype":42,
    "axes.spines.top":False,"axes.spines.right":False})
d=pd.read_csv(Path(__file__).resolve().parents[1] / 'results/progression/progression_per_circuit.csv')
p=d.pivot_table(index='circuit',columns='method',values='F',aggfunc='mean')*100
order=[c for c in p.index if c.startswith('RS232')]+[c for c in p.index if c.startswith('s1423')]+ \
      sorted([c for c in p.index if not c.startswith(('RS232','s1423'))])
p=p.loc[order]
ca=p['CA kNN-50'].values; st=p['Stacking LOF+CBLOF (Nie et al.)'].values
labels=[c.replace('_','-') for c in order]
y=np.arange(len(order))[::-1]; h=0.38
fig,ax=plt.subplots(figsize=(3.5,3.9))
ax.barh(y+h/2,st,h,color="#9a9994",edgecolor="white",lw=0.5,label="Stacking LOF + CBLOF [8]")
ax.barh(y-h/2,ca,h,color="#2a78d6",edgecolor="white",lw=0.5,label="CA kNN-50 (proposed)")
ax.set_yticks(y); ax.set_yticklabels(labels); ax.tick_params(axis='y',length=0)
ax.set_xlim(0,100); ax.set_xlabel("F-measure (%)")
ax.xaxis.grid(True,color="#e4e3df",lw=0.5); ax.set_axisbelow(True)
ax.axhline(y[8]-0.5,color="#8a8984",lw=0.6,ls=(0,(3,2)))
ax.legend(loc="upper center",bbox_to_anchor=(0.45,1.08),ncol=2,frameon=False)
fig.tight_layout()
fig.savefig("fig_conf_percircuit.pdf"); fig.savefig("fig_conf_percircuit.png",dpi=300)
print(p[['CA kNN-50','Stacking LOF+CBLOF (Nie et al.)']].round(1).to_string())
