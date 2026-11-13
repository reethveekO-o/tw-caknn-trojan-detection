"""
Benchmark of every method tried so far, plus other unsupervised detectors, on the XGB22
features (16 structural features selected by Nie et al. + 6 SCOAP). No step uses labels.

Flag-set metrics : TPR (= recall), TNR, precision, accuracy, F
Ranking metrics  : ROC AUC, AP (average precision), P@n (precision at n = number of Trojan nets)

Budget for detectors that only give a score: top floor(C n) nets, C = 0.05 RS232,
0.1 s1423, 0.005 s-series (Nie et al.'s protocol). TW-CA-kNN auto is evaluated on its own
flag set (no C).

Families
  CA kNN family   : CA kNN-50 (no smoothing, 1-hop, hybrid), TW-CA-kNN (C n budget, auto)
  References      : LOF + CBLOF stacking (Nie et al.), testability only
  Classical       : Isolation Forest, LOF, kNN distance, HBOS, ECOD, COPOD, OCSVM, PCA, CBLOF, LODA
  Feature extraction:
      Autoencoder   : MLP 22-16-8-16-22, reconstruction error
      AE-latent CA  : CA kNN-50 hybrid on the autoencoder's 8-d bottleneck
      Graph-emb CA  : CA kNN-50 hybrid on XGB22 + 8-d spectral embedding of the netlist graph
  "x h^2" rows    : the same detector's score, rank-normalised to [0, 1], times testability
                    hardness squared - shows whether CA kNN is the better rarity engine

Run: python benchmark_all.py DATA_DIR OUT_DIR [N_JOBS]
"""
import os, sys, time, warnings
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ['PYTHONPATH'] = _HERE + os.pathsep + os.environ.get('PYTHONPATH', '')
warnings.filterwarnings('ignore')
STOCH_RUNS = 5          # seeds for stochastic non-CA detectors


def rank_metrics(y, sc):
    top = np.argsort(-sc, kind='stable')[:int(y.sum())]
    return dict(AUC=roc_auc_score(y, sc), AP=average_precision_score(y, sc), Pn=y[top].mean())


def u01(x):
    return rankdata(x) / len(x)


def spectral_embedding(nb, dim=8):
    from scipy.sparse import csr_matrix, identity, diags
    from scipy.sparse.linalg import eigsh
    n = len(nb); rows = np.repeat(np.arange(n), [len(v) for v in nb])
    cols = np.concatenate([v for v in nb]) if rows.size else np.array([], int)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)); A = A.maximum(A.T)
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    Dm = diags(1 / np.sqrt(d)); Lsym = identity(n) - Dm @ A @ Dm
    vals, vecs = eigsh(Lsym + 1e-6 * identity(n), k=dim + 1, sigma=0, which='LM')
    return vecs[:, np.argsort(vals)][:, 1:]


def autoencoder(Z, seed):
    from sklearn.neural_network import MLPRegressor
    ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation='relu', max_iter=400, random_state=seed,
                      early_stopping=False, alpha=1e-4).fit(Z, Z)
    err = ((ae.predict(Z) - Z) ** 2).mean(1)
    h = Z
    for W, b in zip(ae.coefs_[:2], ae.intercepts_[:2]):  # forward to the 8-d bottleneck
        h = np.maximum(h @ W + b, 0)
    return err, h


def run_circuit(c, D, edges, cells):
    from threadpoolctl import threadpool_limits
    import fs_experiment as F, dra_louvain as L, hybrid_experiment as HX
    import tw_caknn as TW
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA as PPCA
    from pyod.models.cblof import CBLOF
    from pyod.models.loda import LODA
    with threadpool_limits(1):
        t0 = time.time()
        y = D.label.to_numpy(int); nets = list(D.net); n = len(y)
        C = F.contamination(c); m = max(1, int(np.floor(C * n))); k = min(F.K_NN, n - 2)
        h = TW.hardness(D); h2 = h ** 2
        X = D[F.XGB22].to_numpy(float); Z = F.zscore(X); small = n < 1000
        nb = F.neighbours(edges, nets)
        lab = L.dra_louvain(nets, edges, cells)[0] if small else None
        rng = lambda: np.random.default_rng(F.SEED)
        P_plain = F.kmeans_pool(Z, rng())
        P_hyb = F.kmeans_pool(HX.community_features(X, lab), rng()) if small else P_plain
        emb = spectral_embedding(nb)
        Xe = np.c_[X, emb]
        P_emb = F.kmeans_pool(HX.community_features(Xe, lab), rng()) if small else F.kmeans_pool(F.zscore(Xe), rng())
        ae_err, ae_lat = autoencoder(Z, 0)
        ae_lat = ae_lat[:, ae_lat.std(0) > 0] if (ae_lat.std(0) > 0).sum() >= 2 else Z
        P_ae = F.kmeans_pool(HX.community_features(ae_lat, lab), rng()) if small else F.kmeans_pool(F.zscore(ae_lat), rng())
        hop = lambda s: TW.smooth_raw(s, False, None, nb)
        hyb = lambda s: TW.smooth_raw(s, small, lab, nb)

        sets, ranks = defaultdict(list), defaultdict(list)
        def add(name, score=None, flags=None):
            if score is not None:
                ranks[name].append(rank_metrics(y, score))
                if flags is None: flags = F.flag_top(score, C)
            sets[name].append(F.metrics(y, flags))

        # ---------------- CA kNN family (20 runs) ----------------
        for r in range(F.RUNS):
            sel = np.random.default_rng(100 + r).choice(F.POOL, F.MES, replace=False)
            Sp = F.co_association(P_plain[:, sel]); sp = F.knn_score(Sp, k); del Sp
            add('CA kNN-50', sp)
            add('CA kNN-50 + 1-hop', hop(sp))
            S = F.co_association(P_hyb[:, sel]); s = F.knn_score(S, k); np.fill_diagonal(S, 1.0)
            st = hyb(s)
            add('CA kNN-50 + hybrid (Louvain / 1-hop)', st)
            f = st * h2
            add('TW-CA-kNN, C n budget', f)
            auto = TW.auto_flags(S, f); del S
            add('TW-CA-kNN auto', f, flags=auto)
            Se = F.co_association(P_emb[:, sel]); se = hyb(F.knn_score(Se, k)); del Se
            add('Graph-emb CA kNN-50 hybrid', se); add('Graph-emb CA kNN-50 hybrid x h^2', se * h2)
            Sa = F.co_association(P_ae[:, sel]); sa = hyb(F.knn_score(Sa, k)); del Sa
            add('AE-latent CA kNN-50 hybrid', sa); add('AE-latent CA kNN-50 hybrid x h^2', sa * h2)
            add('Stacking LOF+CBLOF (Nie et al.)', TW.stacking_score(Z, r))

        # ---------------- references ----------------
        add('Testability only (h)', h)

        # ---------------- other unsupervised detectors ----------------
        det = {'Isolation Forest': lambda s: IForest(n_estimators=200, random_state=s),
               'LOF (k=20)': lambda s: LOF(n_neighbors=20),
               'kNN distance (k=50, mean)': lambda s: KNN(n_neighbors=min(50, n - 2), method='mean'),
               'HBOS': lambda s: HBOS(), 'ECOD': lambda s: ECOD(), 'COPOD': lambda s: COPOD(),
               'OCSVM': lambda s: OCSVM(), 'PCA (recon.)': lambda s: PPCA(random_state=s),
               'CBLOF': lambda s: CBLOF(n_clusters=8, random_state=s), 'LODA': lambda s: LODA()}
        stoch = {'Isolation Forest', 'CBLOF', 'LODA'}
        for nm, mk in det.items():
            for sd in range(STOCH_RUNS if nm in stoch else 1):
                try:
                    if nm == 'LODA': np.random.seed(sd)
                    sc = mk(sd).fit(Z).decision_scores_
                except Exception as e:
                    print(c, nm, 'failed', e); continue
                sc = np.nan_to_num(sc, nan=np.nanmin(sc))
                add(nm, sc); add(nm + ' x h^2', u01(sc) * h2)
        for sd in range(STOCH_RUNS):
            err = ae_err if sd == 0 else autoencoder(Z, sd)[0]
            add('Autoencoder (recon.)', err); add('Autoencoder (recon.) x h^2', u01(err) * h2)

        rows = []
        for nm, v in sets.items():
            row = dict(circuit=c, method=nm, **{q: float(np.mean([d[q] for d in v])) for q in v[0]})
            if nm in ranks:
                row.update({q: float(np.mean([d[q] for d in ranks[nm]])) for q in ['AUC', 'AP', 'Pn']})
            rows.append(row)
        for nm in ranks:
            if nm not in sets:
                rows.append(dict(circuit=c, method=nm, **{q: float(np.mean([d[q] for d in ranks[nm]])) for q in ['AUC', 'AP', 'Pn']}))
        print(f'{c:12s} n={n:5d} {time.time() - t0:.0f}s', flush=True)
        return rows


GROUPS = [
    ('CA kNN family', ['CA kNN-50', 'CA kNN-50 + 1-hop', 'CA kNN-50 + hybrid (Louvain / 1-hop)',
                       'TW-CA-kNN, C n budget', 'TW-CA-kNN auto']),
    ('References', ['Stacking LOF+CBLOF (Nie et al.)', 'Testability only (h)']),
    ('Classical detectors', ['Isolation Forest', 'LOF (k=20)', 'kNN distance (k=50, mean)', 'HBOS', 'ECOD',
                             'COPOD', 'OCSVM', 'PCA (recon.)', 'CBLOF', 'LODA']),
    ('Feature extraction', ['Autoencoder (recon.)', 'AE-latent CA kNN-50 hybrid', 'Graph-emb CA kNN-50 hybrid']),
]


def main(data_dir='data', out_dir='results/benchmark', n_jobs=-1):
    import dra_louvain as L
    kw = dict(keep_default_na=False, na_values=[])
    S = pd.read_csv(os.path.join(data_dir, 'structural_features.csv'), **kw)
    T = pd.read_csv(os.path.join(data_dir, 'testability_features.csv'), **kw)
    M = S.merge(T.drop(columns='label'), on=['circuit', 'net']); M = M[M.net != '']
    E = pd.read_csv(os.path.join(data_dir, 'net_edges.csv'), **kw)
    EDGES = {c: list(zip(g.src, g.dst)) for c, g in E.groupby('circuit')}
    Dc = pd.read_csv(os.path.join(data_dir, 'net_driver_cells.csv'), **kw)
    CELLS = {c: {L.norm(n): t for n, t in zip(g.net, g.driver_cell)} for c, g in Dc.groupby('circuit')}
    circuits = sorted(M.circuit.unique(), key=lambda c: -(M.circuit == c).sum())
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(run_circuit)(c, M[M.circuit == c].reset_index(drop=True),
                                                       EDGES.get(c, []), CELLS.get(c, {})) for c in circuits)
    R = pd.DataFrame([x for o in out for x in o])
    os.makedirs(out_dir, exist_ok=True)
    R.to_csv(os.path.join(out_dir, 'all_per_circuit.csv'), index=False)
    cols = ['TPR', 'TNR', 'P', 'TPR', 'A', 'F', 'AUC', 'AP', 'Pn']
    agg = R.groupby('method', sort=False)[['TPR', 'TNR', 'P', 'A', 'F', 'AUC', 'AP', 'Pn']].mean().mul(100)
    agg.insert(3, 'Recall', agg['TPR'])
    small = R.circuit.str.startswith(('RS232', 's1423'))
    agg['F small (9)'] = R[small].groupby('method', sort=False).F.mean().mul(100)
    agg['F large (6)'] = R[~small].groupby('method', sort=False).F.mean().mul(100)
    agg = agg.rename(columns={'P': 'Precision', 'A': 'Accuracy', 'Pn': 'P@n'}).round(2)
    order = []
    for g, names in GROUPS:
        for nm in names:
            for v in [nm, nm + ' x h^2']:
                if v in agg.index: order.append((g, v))
    agg = agg.loc[[v for _, v in order]]
    agg.insert(0, 'group', [g for g, _ in order])
    agg.to_csv(os.path.join(out_dir, 'all_summary.csv'))
    pd.set_option('display.width', 250)
    print(f'done in {time.time() - t0:.0f}s\n{agg.to_string()}')
    return R, agg


if __name__ == '__main__':
    import benchmark_all as B
    a = sys.argv[1:]
    B.main(a[0] if a else 'data', a[1] if len(a) > 1 else 'results/benchmark', int(a[2]) if len(a) > 2 else -1)
