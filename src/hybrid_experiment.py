"""
(1) Stacking LOF+CBLOF (Nie et al.) on label-free feature sets, for a fair comparison.
(2) Louvain + 1-hop hybrid smoothing for CA kNN-50.

Feature sets: XGB22 (label-informed reference), SCOAP6 and TAIL16+S (label-free).
Smoothing variants (all label-free; DRA Louvain picks gamma without labels):
  none        CA kNN-50 score only
  1hop        z' = 0.5 z + 0.5 mean z of directly wired nets
  louv        z' = 0.5 z + 0.5 mean z of the other nets in the DRA Louvain community
  louvF       Louvain community-deviation features (pool rebuilt on [Z, Z - community mean]) + louv
  hyb_size    louvF for circuits with fewer than 1000 nets, 1hop otherwise (the threshold
              sits in the empty gap between 576 and 2196 nets, so any value there gives the same split)
  hyb_mix     z' = 0.5 z + 0.25 mean(1-hop) + 0.25 mean(community)   (one rule for every circuit)
  hyb_mixF    hyb_mix on the Louvain-feature score
  hyb_avg     mean of the robust-z 1hop and louv scores
  hyb_in      1-hop restricted to wired neighbours inside the same Louvain community

Run: python hybrid_experiment.py DATA_DIR CELLS_CSV OUT_DIR
"""
import os, sys, time, json, warnings
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import LocalOutlierFactor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_experiment as F
import dra_louvain as L
warnings.filterwarnings('ignore')
ALPHA = 0.5


def louv_mean(z, lab):
    s = np.bincount(lab, weights=z); c = np.bincount(lab).astype(float)
    num, den = s[lab] - z, c[lab] - 1
    out = np.full_like(z, np.nan); has = den > 0
    out[has] = num[has] / den[has]
    return out


def hop_mean(z, nb):
    out = np.full_like(z, np.nan)
    for i, v in enumerate(nb):
        if len(v): out[i] = z[v].mean()
    return out


def blend(z, parts, weights):
    """z' = w0 z + sum w_k m_k; a missing neighbour mean gives its weight back to z."""
    out = weights[0] * z.copy()
    for m, w in zip(parts, weights[1:]):
        ok = ~np.isnan(m)
        out[ok] += w * m[ok]; out[~ok] += w * z[~ok]
    return F.robust_z(out)


def community_features(X, lab):
    Z = F.zscore(X); means = np.zeros((lab.max() + 1, Z.shape[1]))
    np.add.at(means, lab, Z); means /= np.bincount(lab)[:, None]
    return F.zscore(np.hstack([Z, Z - means[lab]]))


def stacking_scores(Z, r, lof):
    from pyod.models.cblof import CBLOF
    cb = CBLOF(n_clusters=8, alpha=0.9, beta=5, random_state=r).fit(Z).decision_scores_
    return 0.2 * rankdata(lof) + 0.8 * rankdata(cb)


def main(data_dir, cells_csv, out_dir):
    kw = dict(keep_default_na=False, na_values=[])
    S = pd.read_csv(os.path.join(data_dir, 'structural_features.csv'), **kw)
    T = pd.read_csv(os.path.join(data_dir, 'testability_features.csv'), **kw)
    M = S.merge(T.drop(columns='label'), on=['circuit', 'net']); M = M[M.net != '']
    E = pd.read_csv(os.path.join(data_dir, 'net_edges.csv'), **kw)
    EDGES = {c: list(zip(g.src, g.dst)) for c, g in E.groupby('circuit')}
    Dc = pd.read_csv(cells_csv, **kw)
    CELLS = {c: {L.norm(n): t for n, t in zip(g.net, g.driver_cell)} for c, g in Dc.groupby('circuit')}
    os.makedirs(out_dir, exist_ok=True)
    rows, comm = [], []
    for c, D in M.groupby('circuit', sort=False):
        t0 = time.time()
        y = D.label.to_numpy(int); nets = list(D.net); n = len(y)
        C = F.contamination(c); k = min(F.K_NN, n - 2)
        nb = F.neighbours(EDGES.get(c, []), nets)
        lab, g, _ = L.dra_louvain(nets, EDGES.get(c, []), CELLS.get(c, {}))
        comm.append(dict(circuit=c, nets=n, gamma=g, communities=int(lab.max() + 1),
                         largest_frac=round(np.bincount(lab).max() / n, 3)))
        nb_in = [v[lab[v] == lab[i]] for i, v in enumerate(nb)]
        small = n < 1000
        for fs in ['XGB22', 'SCOAP6', 'TAIL16+S']:
            feats = F.choose(fs, D); X = D[feats].to_numpy(float); Z = F.zscore(X)
            P = F.kmeans_pool(Z, np.random.default_rng(F.SEED))
            Pf = F.kmeans_pool(community_features(X, lab), np.random.default_rng(F.SEED))
            lof = -LocalOutlierFactor(n_neighbors=min(250, n - 1)).fit(Z).negative_outlier_factor_
            res = defaultdict(list)
            for r in range(F.RUNS):
                sel = np.random.default_rng(100 + r).choice(F.POOL, F.MES, replace=False)
                z = F.robust_z(F.knn_score(F.co_association(P[:, sel]), k))
                zf = F.robust_z(F.knn_score(F.co_association(Pf[:, sel]), k))
                h, lm = hop_mean(z, nb), louv_mean(z, lab)
                hf, lmf = hop_mean(zf, nb), louv_mean(zf, lab)
                v = {
                    'none': z,
                    '1hop': blend(z, [h], [0.5, 0.5]),
                    'louv': blend(z, [lm], [0.5, 0.5]),
                    'louvF': blend(zf, [lmf], [0.5, 0.5]),
                    'hyb_mix': blend(z, [h, lm], [0.5, 0.25, 0.25]),
                    'hyb_mixF': blend(zf, [hf, lmf], [0.5, 0.25, 0.25]),
                    'hyb_avg': F.robust_z(blend(z, [h], [0.5, 0.5]) + blend(z, [lm], [0.5, 0.5])),
                    'hyb_in': blend(z, [hop_mean(z, nb_in)], [0.5, 0.5]),
                }
                v['hyb_size'] = v['louvF'] if small else v['1hop']
                for name, s in v.items():
                    res[f'CA kNN-50 | {name}'].append(F.metrics(y, F.flag_top(s, C)))
                st = stacking_scores(Z, r, lof); zs = F.robust_z(st)
                res['Stacking | none'].append(F.metrics(y, F.flag_top(st, C)))
                res['Stacking | 1hop'].append(F.metrics(y, F.flag_top(blend(zs, [hop_mean(zs, nb)], [0.5, 0.5]), C)))
            for mth, vals in res.items():
                rows.append(dict(set=fs, circuit=c, method=mth,
                                 **{q: float(np.mean([d[q] for d in vals])) for q in vals[0]}))
            f = lambda m: 100 * np.mean([d['F'] for d in res[m]])
            print(f'{c:12s} {fs:8s} n={n:5d} g={g:<4} none={f("CA kNN-50 | none"):5.1f} 1hop={f("CA kNN-50 | 1hop"):5.1f} '
                  f'louvF={f("CA kNN-50 | louvF"):5.1f} mix={f("CA kNN-50 | hyb_mix"):5.1f} '
                  f'stack={f("Stacking | none"):5.1f} ({time.time()-t0:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'hybrid_per_circuit.csv'), index=False)
    pd.DataFrame(comm).to_csv(os.path.join(out_dir, 'hybrid_communities.csv'), index=False)
    R = pd.DataFrame(rows)
    print(R.groupby(['set', 'method']).F.mean().mul(100).round(2).unstack(0).to_string())


if __name__ == '__main__':
    main(*sys.argv[1:4])
