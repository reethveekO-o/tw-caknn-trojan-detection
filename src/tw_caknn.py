"""
TW-CA-kNN: testability-weighted CA kNN-50 for unsupervised hardware-Trojan detection.

  1. CA kNN-50 hybrid rarity: k-means pool -> co-association CA -> kNN-50 rarity
        s_i = 1 - mean of the 50 largest CA(i, j), smoothed with DRA Louvain communities
        (n < 1000) or directly wired nets (n >= 1000): s~ = 0.5 s + 0.5 mean(s of group).
        s~ lies in [0, 1]: the share of ensemble members that do NOT put the net with its
        50 closest matches.
  2. Testability hardness: h_i = percentile of log(1+CC_i) + log(1+CO_i) in the circuit,
        CC = sqrt(CC0^2 + CC1^2) (SCOAP). One-sided: only hard-to-control /
        hard-to-observe raises suspicion.
  3. Suspicion score: f = s~ * h^BETA (BETA = 2). Rare AND hard to test.
  4. Automatic flag set (no contamination C):
        seeds  = nets with f >= SEED_FRAC * max f
        grow   = add any net with f >= POOL_FRAC * max f that a majority of the ensemble
                 members cluster with an already-flagged net (CA >= 0.5); repeat until stable.

Evaluation (labels used only here), mean of 20 runs per circuit:
  ranking  : ROC AUC, AP, P@n (precision at n = number of Trojan nets) for TW-CA-kNN and
             baselines, plus a BETA sweep
  flag sets: TPR, TNR, precision, accuracy, F

Run: python tw_caknn.py DATA_DIR OUT_DIR [N_JOBS]
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ['PYTHONPATH'] = _HERE + os.pathsep + os.environ.get('PYTHONPATH', '')
warnings.filterwarnings('ignore')

SMALL_N = 1000
BETA = 2.0
SEED_FRAC, POOL_FRAC, CA_MAJ = 0.7, 0.5, 0.5
BETAS = [0, 0.5, 1, 2, 3, 4]


# --------------------------------------------------------------------------- method
def hardness(D):
    """Testability hardness percentile h in (0, 1] from SCOAP CC0, CC1, CO."""
    cc = np.sqrt(D.CC0.to_numpy(float) ** 2 + D.CC1.to_numpy(float) ** 2)
    co = D.CO.to_numpy(float)
    return rankdata(np.log1p(cc) + np.log1p(co), 'max') / len(D)


def smooth_raw(s, small, lab, nb):
    """Hybrid smoothing on the raw [0, 1] kNN score (Louvain if small, 1-hop if large)."""
    import hybrid_experiment as HX
    m = HX.louv_mean(s, lab) if small else HX.hop_mean(s, nb)
    return np.where(np.isnan(m), s, 0.5 * s + 0.5 * m)


def grow(S, seed, pool, th=CA_MAJ):
    fl = seed.copy()
    while True:
        new = fl | (pool & (S[:, fl] >= th).any(1))
        if new.sum() == fl.sum():
            return fl
        fl = new


def auto_flags(S, f):
    """Seeds within SEED_FRAC of the top score, grown by co-association majority."""
    mx = f.max()
    if mx <= 0:
        return np.zeros(len(f), bool)
    return grow(S, f >= SEED_FRAC * mx, f >= POOL_FRAC * mx)


# --------------------------------------------------------------------------- evaluation helpers
def rank_metrics(y, score):
    n_t = int(y.sum()); top = np.argsort(-score, kind='stable')[:n_t]
    return dict(AUC=roc_auc_score(y, score), AP=average_precision_score(y, score), Pn=y[top].mean())


def stacking_score(Z, seed):
    """LOF + CBLOF stacking baseline (Nie et al.)."""
    from sklearn.neighbors import LocalOutlierFactor
    from pyod.models.cblof import CBLOF
    lof = -LocalOutlierFactor(n_neighbors=min(250, len(Z) - 1)).fit(Z).negative_outlier_factor_
    cb = CBLOF(n_clusters=8, alpha=0.9, beta=5, random_state=seed).fit(Z).decision_scores_
    return 0.2 * rankdata(lof) + 0.8 * rankdata(cb)


def run_circuit(c, D, edges, cells):
    from threadpoolctl import threadpool_limits
    import fs_experiment as F, dra_louvain as L, hybrid_experiment as HX
    with threadpool_limits(1):
        t0 = time.time()
        y = D.label.to_numpy(int); nets = list(D.net); n = len(y)
        C = F.contamination(c); k = min(F.K_NN, n - 2)
        h = hardness(D)
        small = n < SMALL_N
        X = D[F.XGB22].to_numpy(float)
        lab, nb = None, None
        if small:
            lab, _, _ = L.dra_louvain(nets, edges, cells)
            P = F.kmeans_pool(HX.community_features(X, lab), np.random.default_rng(F.SEED))
        else:
            nb = F.neighbours(edges, nets)
            P = F.kmeans_pool(F.zscore(X), np.random.default_rng(F.SEED))
        Zs = F.zscore(X)
        rank_rows, set_rows = [], []
        for r in range(F.RUNS):
            sel = np.random.default_rng(100 + r).choice(F.POOL, F.MES, replace=False)
            S = F.co_association(P[:, sel])
            s = F.knn_score(S, k); np.fill_diagonal(S, 1.0)
            st = smooth_raw(s, small, lab, nb)
            f = st * h ** BETA
            stack = stacking_score(Zs, r)
            scorers = {'CA kNN-50 hybrid': st, 'Testability only (h)': h,
                       'Stacking LOF+CBLOF (Nie et al.)': stack, 'TW-CA-kNN': f}
            for b in BETAS:
                scorers[f'beta={b}'] = st * h ** b
            for nm, sc in scorers.items():
                rank_rows.append(dict(circuit=c, run=r, scorer=nm, **rank_metrics(y, sc)))
            sets = {'CA kNN-50 hybrid (C n)': F.flag_top(st, C),
                    'Testability only (C n)': F.flag_top(h, C),
                    'Stacking LOF+CBLOF (C n)': F.flag_top(stack, C),
                    'TW-CA-kNN (C n)': F.flag_top(f, C),
                    'TW-CA-kNN auto': auto_flags(S, f)}
            for nm, fl in sets.items():
                set_rows.append(dict(circuit=c, run=r, method=nm, **F.metrics(y, fl)))
            del S
        info = dict(circuit=c, nets=n, trojan_nets=int(y.sum()), seconds=round(time.time() - t0, 1))
        print(f'{c:12s} n={n:5d} {info["seconds"]}s', flush=True)
        return rank_rows, set_rows, info


def load_data(data_dir):
    import dra_louvain as L
    kw = dict(keep_default_na=False, na_values=[])
    S = pd.read_csv(os.path.join(data_dir, 'structural_features.csv'), **kw)
    T = pd.read_csv(os.path.join(data_dir, 'testability_features.csv'), **kw)
    M = S.merge(T.drop(columns='label'), on=['circuit', 'net']); M = M[M.net != '']
    E = pd.read_csv(os.path.join(data_dir, 'net_edges.csv'), **kw)
    EDGES = {c: list(zip(g.src, g.dst)) for c, g in E.groupby('circuit')}
    Dc = pd.read_csv(os.path.join(data_dir, 'net_driver_cells.csv'), **kw)
    CELLS = {c: {L.norm(n): t for n, t in zip(g.net, g.driver_cell)} for c, g in Dc.groupby('circuit')}
    return M, EDGES, CELLS


def main(data_dir='data', out_dir='results/tw_caknn', n_jobs=-1):
    M, EDGES, CELLS = load_data(data_dir)
    circuits = sorted(M.circuit.unique(), key=lambda c: -(M.circuit == c).sum())
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(run_circuit)(c, M[M.circuit == c].reset_index(drop=True),
                                                       EDGES.get(c, []), CELLS.get(c, {})) for c in circuits)
    RK = pd.DataFrame([x for o in out for x in o[0]]); ST = pd.DataFrame([x for o in out for x in o[1]])
    os.makedirs(out_dir, exist_ok=True)
    rk = RK.groupby(['scorer', 'circuit'], sort=False)[['AUC', 'AP', 'Pn']].mean().reset_index()
    st = ST.groupby(['method', 'circuit'], sort=False)[['TPR', 'TNR', 'P', 'A', 'F', 'TP', 'FP']].mean().reset_index()
    rk.to_csv(os.path.join(out_dir, 'ranking_per_circuit.csv'), index=False)
    st.to_csv(os.path.join(out_dir, 'flags_per_circuit.csv'), index=False)
    rs = rk.groupby('scorer', sort=False)[['AUC', 'AP', 'Pn']].mean().mul(100).round(2).rename(columns={'Pn': 'P@n'})
    ss = st.groupby('method', sort=False)[['TPR', 'TNR', 'P', 'A', 'F']].mean().mul(100).round(2)
    ss = ss.rename(columns={'P': 'Precision', 'A': 'Accuracy'})
    ss[['Trojans found', 'false alarms']] = st.groupby('method', sort=False)[['TP', 'FP']].sum().round(1)
    tests = []
    piv = rk.pivot(index='circuit', columns='scorer')
    for metric in ['AP', 'Pn']:
        for other in ['CA kNN-50 hybrid', 'Testability only (h)', 'Stacking LOF+CBLOF (Nie et al.)']:
            a, b = piv[metric]['TW-CA-kNN'], piv[metric][other]
            w, l = int((a > b + 1e-9).sum()), int((a < b - 1e-9).sum())
            tests.append(dict(metric=metric, comparison=f'TW-CA-kNN vs {other}', wins=w, losses=l,
                              ties=len(a) - w - l, p=round(wilcoxon(a, b).pvalue, 4) if (a != b).any() else 1.0))
    pf = st.pivot(index='circuit', columns='method', values='F')
    for other in ['CA kNN-50 hybrid (C n)', 'Testability only (C n)', 'Stacking LOF+CBLOF (C n)']:
        a, b = pf['TW-CA-kNN auto'], pf[other]; w, l = int((a > b + 1e-9).sum()), int((a < b - 1e-9).sum())
        tests.append(dict(metric='F', comparison=f'TW-CA-kNN auto vs {other}', wins=w, losses=l,
                          ties=len(a) - w - l, p=round(wilcoxon(a, b).pvalue, 4) if (a != b).any() else 1.0))
    TS = pd.DataFrame(tests)
    rs.to_csv(os.path.join(out_dir, 'ranking_summary.csv')); ss.to_csv(os.path.join(out_dir, 'flags_summary.csv'))
    TS.to_csv(os.path.join(out_dir, 'significance.csv'), index=False)
    pd.DataFrame([o[2] for o in out]).to_csv(os.path.join(out_dir, 'circuits.csv'), index=False)
    pd.set_option('display.width', 200)
    print(f'done in {time.time() - t0:.0f}s\n\nRANKING (mean over circuits, %)\n{rs.to_string()}\n\n'
          f'FLAG SETS\n{ss.to_string()}\n\nSIGNIFICANCE (paired Wilcoxon over circuits)\n{TS.to_string(index=False)}')
    return rk, st, rs, ss, TS


if __name__ == '__main__':
    import tw_caknn as TW
    a = sys.argv[1:]
    TW.main(a[0] if a else 'data', a[1] if len(a) > 1 else 'results/tw_caknn', int(a[2]) if len(a) > 2 else -1)
