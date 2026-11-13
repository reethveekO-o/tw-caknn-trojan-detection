"""
Label-free feature-selection algorithms vs the label-informed XGB22 set, evaluated with both
the original CA kNN-50 hybrid (top floor(C n)) and TW-CA-kNN (auto flag count).

Every selector runs PER CIRCUIT and never sees labels. Candidate pool: 56 structural
features + 6 SCOAP, near-constant ones (one value on >= 99 % of nets) removed first.
Two modes per algorithm:
  <ALG>22    pick 22 features from all 62
  <ALG>16+S  pick 16 structural features, always keep the 6 SCOAP

Algorithms
  ALL   : no selection (every non-constant feature)
  RED   : redundancy filter, one medoid per |Spearman rho| >= 0.9 cluster   (fs_experiment)
  LS    : Laplacian Score, He et al. 2005                                    (fs_experiment)
  ECS   : ensemble-consensus score (k-means eta^2)                           (fs_experiment)
  TAIL  : excess kurtosis (heavy tails)                                      (fs_experiment)
  MCFS  : multi-cluster feature selection, Cai et al. 2010 - 5 spectral eigenvectors of a
          10-NN heat-kernel graph, LARS per eigenvector, score = max |coefficient|
  SPEC  : spectral feature selection, Zhao & Liu 2007 - normalised-Laplacian score phi2
  PFA   : principal feature analysis, Lu et al. 2007 - PCA (90 % variance), k-means on the
          loading rows, keep the feature nearest each centroid
  XGB22 : Nie et al.'s XGBoost-selected set (uses labels) - reference

Run: python fs_select_compare.py DATA_DIR OUT_DIR [N_JOBS] [RUNS]
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ['PYTHONPATH'] = _HERE + os.pathsep + os.environ.get('PYTHONPATH', '')
warnings.filterwarnings('ignore')

SETS = os.environ.get('FS_SETS', '').split(',') if os.environ.get('FS_SETS') else ['XGB22', 'ALL', 'RED', 'LS22', 'LS16+S', 'ECS22', 'ECS16+S', 'TAIL22', 'TAIL16+S',
        'MCFS22', 'MCFS16+S', 'SPEC22', 'SPEC16+S', 'PFA22', 'PFA16+S']


def knn_graph(Z, k=10):
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    n = len(Z)
    dist, ind = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    dist, ind = dist[:, 1:], ind[:, 1:]
    t = np.mean(dist ** 2) or 1.0
    W = csr_matrix((np.exp(-dist.ravel() ** 2 / t), (np.repeat(np.arange(n), k), ind.ravel())), shape=(n, n))
    return W.maximum(W.T).tocsr()


def norm_laplacian(W):
    from scipy.sparse import identity, diags
    d = np.asarray(W.sum(1)).ravel(); d[d == 0] = 1e-12
    Dm = diags(1 / np.sqrt(d))
    return identity(W.shape[0]) - Dm @ W @ Dm, d


def mcfs_scores(Z, n_eig=5):
    from scipy.sparse import identity
    from scipy.sparse.linalg import eigsh
    from sklearn.linear_model import Lars
    W = knn_graph(Z); L, d = norm_laplacian(W)
    # smallest eigenvectors of L_sym = largest of the normalised adjacency I - L_sym
    # (no shift-invert, which fails to converge on the 5k-net graphs)
    A = identity(L.shape[0]) - L
    vals, vecs = eigsh(A, k=n_eig + 1, which='LA', tol=1e-6, maxiter=20000,
                       v0=np.random.default_rng(0).standard_normal(L.shape[0]))
    Y = (vecs / np.sqrt(d)[:, None])[:, np.argsort(-vals)][:, 1:]
    sc = np.zeros(Z.shape[1])
    for k in range(Y.shape[1]):
        a = Lars(n_nonzero_coefs=min(Z.shape[1], 22), fit_intercept=True).fit(Z, Y[:, k]).coef_
        sc = np.maximum(sc, np.abs(a))
    return sc                                    # higher = better


def spec_scores(Z):
    W = knn_graph(Z); L, d = norm_laplacian(W)
    xi = np.sqrt(d) / np.linalg.norm(np.sqrt(d))
    out = []
    for j in range(Z.shape[1]):
        f = np.sqrt(d) * Z[:, j]; nf = np.linalg.norm(f)
        if nf == 0: out.append(np.inf); continue
        f = f / nf; num = f @ (L @ f); den = 1 - (f @ xi) ** 2
        out.append(num / den if den > 1e-12 else np.inf)
    return -np.array(out)                        # negate: higher = better


def pfa_select(Z, k_sel, seed=1):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    p = PCA().fit(Z); q = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.9) + 1)
    q = max(q, min(k_sel, Z.shape[1]))
    A = p.components_[:q].T                      # d x q loadings
    km = KMeans(k_sel, n_init=10, random_state=seed).fit(A)
    keep = []
    for g in range(k_sel):
        idx = np.where(km.labels_ == g)[0]
        if len(idx): keep.append(idx[np.argmin(((A[idx] - km.cluster_centers_[g]) ** 2).sum(1))])
    return keep


def choose(name, D):
    import fs_experiment as F
    if name in ('XGB22', 'ALL', 'RED', 'LS22', 'ECS22', 'TAIL22', 'TAIL16+S', 'ECS16+S'):
        return F.choose(name, D)
    alg, mode = (name[:-2], 22) if name.endswith('22') else (name[:-4], 16)
    pool = F.STRUCT + F.SCOAP if mode == 22 else F.STRUCT
    if name == 'LS16+S':
        return F.select_ls(D, F.STRUCT, 16) + F.SCOAP
    cand = F.non_constant(D, pool)
    Z = F.zscore(D[cand].to_numpy(float))
    if alg == 'MCFS': idx = np.argsort(-mcfs_scores(Z))[:mode]
    elif alg == 'SPEC': idx = np.argsort(-spec_scores(Z))[:mode]
    elif alg == 'PFA': idx = pfa_select(Z, mode)
    keep = {cand[i] for i in idx}
    sel = [c for c in pool if c in keep]
    return sel + (F.SCOAP if mode == 16 else [])


def rank_metrics(y, sc):
    from sklearn.metrics import roc_auc_score, average_precision_score
    top = np.argsort(-sc, kind='stable')[:int(y.sum())]
    return dict(AUC=roc_auc_score(y, sc), AP=average_precision_score(y, sc), Pn=y[top].mean())


def run_circuit(c, D, edges, cells, runs):
    from threadpoolctl import threadpool_limits
    import fs_experiment as F, dra_louvain as L, hybrid_experiment as HX, tw_caknn as TW
    with threadpool_limits(1):
        t0 = time.time()
        y = D.label.to_numpy(int); nets = list(D.net); n = len(y)
        C = F.contamination(c); k = min(F.K_NN, n - 2); small = n < 1000
        h2 = TW.hardness(D) ** 2
        nb = F.neighbours(edges, nets)
        lab = L.dra_louvain(nets, edges, cells)[0] if small else None
        rows, chosen = [], {}
        for name in SETS:
            try:
                cols = choose(name, D)
            except Exception as e:
                print(c, name, 'selection failed:', e, flush=True); continue
            chosen[name] = cols
            X = D[cols].to_numpy(float)
            P = F.kmeans_pool(HX.community_features(X, lab) if small else F.zscore(X), np.random.default_rng(F.SEED))
            acc = {'hyb': [], 'auto': [], 'rk_h': [], 'rk_tw': []}
            for r in range(runs):
                sel = np.random.default_rng(100 + r).choice(F.POOL, F.MES, replace=False)
                S = F.co_association(P[:, sel]); s = F.knn_score(S, k); np.fill_diagonal(S, 1.0)
                st = TW.smooth_raw(s, small, lab, nb); f = st * h2
                acc['hyb'].append(F.metrics(y, F.flag_top(st, C)))
                acc['auto'].append(F.metrics(y, TW.auto_flags(S, f)))
                acc['rk_h'].append(rank_metrics(y, st)); acc['rk_tw'].append(rank_metrics(y, f))
                del S
            mean = lambda L_: {q: float(np.mean([d[q] for d in L_])) for q in L_[0]}
            rows.append(dict(circuit=c, set=name, n_feat=len(cols), method='CA kNN-50 hybrid (C n)',
                             **mean(acc['hyb']), **mean(acc['rk_h'])))
            rows.append(dict(circuit=c, set=name, n_feat=len(cols), method='TW-CA-kNN auto',
                             **mean(acc['auto']), **mean(acc['rk_tw'])))
        print(f'{c:12s} n={n:5d} {time.time() - t0:.0f}s', flush=True)
        return rows, {c: chosen}


def main(data_dir='data', out_dir='results/feature_selection', n_jobs=-1, runs=10):
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
                                                       EDGES.get(c, []), CELLS.get(c, {}), runs) for c in circuits)
    R = pd.DataFrame([x for o in out for x in o[0]])
    os.makedirs(out_dir, exist_ok=True)
    R.to_csv(os.path.join(out_dir, 'fs_per_circuit.csv'), index=False)
    chosen = {}
    for o in out: chosen.update(o[1])
    json.dump(chosen, open(os.path.join(out_dir, 'selected_features.json'), 'w'), indent=1)
    cols = ['n_feat', 'TPR', 'TNR', 'P', 'A', 'F', 'AUC', 'AP', 'Pn']
    agg = R.groupby(['method', 'set'], sort=False)[cols].mean()
    agg[cols[1:]] *= 100
    agg = agg.rename(columns={'P': 'Precision', 'A': 'Accuracy', 'Pn': 'P@n'}).round(2)
    agg.to_csv(os.path.join(out_dir, 'fs_summary.csv'))
    pd.set_option('display.width', 250)
    print(f'done in {time.time() - t0:.0f}s\n{agg.to_string()}')


if __name__ == '__main__':
    import fs_select_compare as FS
    a = sys.argv[1:]
    FS.main(a[0] if a else 'data', a[1] if len(a) > 1 else 'results/feature_selection',
            int(a[2]) if len(a) > 2 else -1, int(a[3]) if len(a) > 3 else 10)
