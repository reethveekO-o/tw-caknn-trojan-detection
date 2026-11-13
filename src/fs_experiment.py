"""
Label-free feature selection for CA kNN-50 hardware-Trojan detection.

Feature sets (all chosen per circuit, WITHOUT labels; labels are used only for metrics):
  XGB22  : the 16 structural features Nie et al. picked with XGBoost (label-informed) + 6 SCOAP
           -> reference only
  ALL    : every non-constant feature (56 structural + 6 SCOAP)
  RED    : redundancy filter - drop near-constant features (one value on >= 99 % of nets),
           cluster the rest by 1 - |Spearman rho| (average linkage, cut at |rho| = 0.9),
           keep the medoid of each cluster
  LS22   : Laplacian Score (He, Cai & Niyogi, NIPS 2005) on the non-constant features,
           kNN graph k = 10, heat kernel; keep the 22 lowest scores
  ECS22  : ensemble-consensus score - for each feature, the share of its variance explained
           by the clusters of a k-means ensemble (mean eta^2 over 30 members,
           K ~ U[2, sqrt n]); keep the 22 highest

Detector (unchanged): k-means pool 100, K ~ U[2, sqrt n], M = 20, CA, kNN-50 score,
robust z, optional 1-hop smoothing (alpha 0.5), flag top floor(C n); 20 runs.

Run: python fs_experiment.py DATA_DIR OUT_DIR SET [SET ...]
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')

RUNS, POOL, MES, K_NN, ALPHA, SEED = 20, 100, 20, 50, 0.5, 1
STRUCT = [f'f{i}' for i in range(56)]
SCOAP = ['CC0', 'CC1', 'CO', 'SC0', 'SC1', 'SO']
XGB22 = ['f0', 'f1', 'f2', 'f3', 'f4', 'f28', 'f29', 'f34', 'f45', 'f46', 'f48',
         'f50', 'f52', 'f53', 'f54', 'f55'] + SCOAP
N_SEL = 22


def contamination(c):
    return 0.05 if c.startswith('RS232') else (0.1 if c.startswith('s1423') else 0.005)


# ----------------------------------------------------------------- detector
def zscore(X):
    s = X.std(0, ddof=1); s[s == 0] = 1
    return (X - X.mean(0)) / s


def kmeans_pool(Z, rng, size=POOL):
    n = len(Z); kmax = int(np.sqrt(n)); P = np.zeros((n, size), np.int32)
    for i in range(size):
        k = int(rng.integers(2, kmax + 1))
        P[:, i] = KMeans(k, n_init=1, random_state=int(rng.integers(1 << 30))).fit_predict(Z)
    return P


def co_association(BP):
    cols = []
    for k in range(BP.shape[1]):
        _, lab = np.unique(BP[:, k], return_inverse=True)
        cols.append(np.eye(lab.max() + 1, dtype=np.float32)[lab])
    H = np.hstack(cols)
    return (H @ H.T) / BP.shape[1]


def knn_score(S, k):
    np.fill_diagonal(S, -np.inf)
    return 1 - (-np.partition(-S, k, axis=1)[:, :k]).mean(1)


def robust_z(s):
    med = np.median(s); mad = np.median(np.abs(s - med)) * 1.4826
    if mad == 0: mad = np.std(s) or 1.0
    return (s - med) / mad


def neighbours(edges, nets):
    idx = {n: i for i, n in enumerate(nets)}; nb = [[] for _ in nets]
    for a, b in edges:
        if a in idx and b in idx: nb[idx[a]].append(idx[b]); nb[idx[b]].append(idx[a])
    return [np.array(sorted(set(v)), int) for v in nb]


def onehop(score, nb):
    z = robust_z(score); out = z.copy()
    for i, v in enumerate(nb):
        if len(v): out[i] = (1 - ALPHA) * z[i] + ALPHA * z[v].mean()
    return robust_z(out)


def flag_top(score, C):
    n = len(score); m = max(1, int(np.floor(C * n)))
    f = np.zeros(n, bool); f[np.argsort(-score, kind='stable')[:m]] = True
    return f


def metrics(y, flag):
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1); p = tp / max(tp + fp, 1)
    f = 2 * p * tpr / (p + tpr) if p + tpr > 0 else 0.0
    return dict(TPR=tpr, TNR=tnr, P=p, F=f, A=(tp + tn) / len(y), TP=tp, FP=fp, FN=fn, TN=tn)


# ------------------------------------------------------- label-free selection
def non_constant(D, cols, frac=0.99):
    keep = []
    for c in cols:
        v = D[c].to_numpy()
        top = pd.Series(v).value_counts(normalize=True).iloc[0]
        if top < frac and np.std(v) > 0: keep.append(c)
    return keep


def select_red(D, cols, rho_cut=0.9):
    cand = non_constant(D, cols)
    R = np.abs(spearmanr(D[cand].to_numpy()).correlation)
    R = np.nan_to_num(R, nan=0.0); np.fill_diagonal(R, 1.0)
    Dm = np.clip(1 - R, 0, None); np.fill_diagonal(Dm, 0)
    lab = fcluster(linkage(squareform(Dm, checks=False), 'average'), 1 - rho_cut, 'distance')
    keep = []
    for g in np.unique(lab):
        idx = np.where(lab == g)[0]
        medoid = idx[np.argmax(R[np.ix_(idx, idx)].mean(1))]       # most central member
        keep.append(cand[medoid])
    return [c for c in cols if c in keep]


def laplacian_score(Z, k=10):
    n = len(Z)
    dist, ind = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    dist, ind = dist[:, 1:], ind[:, 1:]
    t = np.mean(dist ** 2) or 1.0
    from scipy.sparse import csr_matrix
    W = csr_matrix((np.exp(-dist.ravel() ** 2 / t), (np.repeat(np.arange(n), k), ind.ravel())), shape=(n, n))
    W = W.maximum(W.T)
    d = np.asarray(W.sum(1)).ravel()
    scores = []
    for j in range(Z.shape[1]):
        f = Z[:, j]
        ft = f - (f @ d) / d.sum()
        den = (ft ** 2) @ d
        num = den - ft @ (W @ ft)                       # f~' L f~ with L = D - W
        scores.append(num / den if den > 0 else np.inf)
    return np.array(scores)


def select_ls(D, cols, k_sel=N_SEL):
    cand = non_constant(D, cols)
    s = laplacian_score(zscore(D[cand].to_numpy(float)))
    order = np.argsort(s)[:k_sel]
    keep = {cand[i] for i in order}
    return [c for c in cols if c in keep]


def select_ecs(D, cols, k_sel=N_SEL, members=30):
    cand = non_constant(D, cols)
    Z = zscore(D[cand].to_numpy(float))
    P = kmeans_pool(Z, np.random.default_rng(SEED + 7), size=members)
    tot = ((Z - Z.mean(0)) ** 2).sum(0)
    eta = np.zeros(len(cand))
    for m in range(members):
        lab = P[:, m]; within = np.zeros(len(cand))
        for g in np.unique(lab):
            Zg = Z[lab == g]; within += ((Zg - Zg.mean(0)) ** 2).sum(0)
        eta += 1 - within / np.where(tot > 0, tot, 1)
    eta /= members
    order = np.argsort(-eta)[:k_sel]
    keep = {cand[i] for i in order}
    return [c for c in cols if c in keep]


def select_tail(D, cols, k_sel=N_SEL):
    """Heavy-tail score: excess kurtosis of each (z-scored) feature. Features on which a
    few nets sit far from the bulk rank first - the property a rare-outlier detector needs."""
    cand = non_constant(D, cols)
    Z = zscore(D[cand].to_numpy(float))
    kurt = (Z ** 4).mean(0) - 3
    order = np.argsort(-kurt)[:k_sel]
    keep = {cand[i] for i in order}
    return [c for c in cols if c in keep]


def choose(set_name, D):
    cols = STRUCT + SCOAP
    if set_name == 'XGB22': return XGB22
    if set_name == 'XGB16': return XGB22[:16]                       # structural part only
    if set_name == 'SCOAP6': return SCOAP
    if set_name == 'TAIL22': return select_tail(D, cols)
    if set_name == 'TAIL16+S': return select_tail(D, STRUCT, 16) + SCOAP
    if set_name == 'ECS16+S': return select_ecs(D, STRUCT, 16) + SCOAP
    if set_name == 'ALL': return non_constant(D, cols, frac=1.01)   # only truly constant ones dropped
    if set_name == 'RED': return select_red(D, cols)
    if set_name == 'LS22': return select_ls(D, cols)
    if set_name == 'ECS22': return select_ecs(D, cols)
    raise ValueError(set_name)


# --------------------------------------------------------------------- main
def main(data_dir, out_dir, sets):
    kw = dict(keep_default_na=False, na_values=[])
    S = pd.read_csv(os.path.join(data_dir, 'structural_features.csv'), **kw)
    T = pd.read_csv(os.path.join(data_dir, 'testability_features.csv'), **kw)
    M = S.merge(T.drop(columns='label'), on=['circuit', 'net']); M = M[M.net != '']
    E = pd.read_csv(os.path.join(data_dir, 'net_edges.csv'), **kw)
    EDGES = {c: list(zip(g.src, g.dst)) for c, g in E.groupby('circuit')}
    os.makedirs(out_dir, exist_ok=True)
    for set_name in sets:
        rows, chosen = [], {}
        t_set = time.time()
        for c, D in M.groupby('circuit', sort=False):
            t0 = time.time()
            feats = choose(set_name, D); chosen[c] = feats
            X = D[feats].to_numpy(float); y = D.label.to_numpy(int); nets = list(D.net)
            n = len(y); C = contamination(c); k = min(K_NN, n - 2)
            nb = neighbours(EDGES.get(c, []), nets)
            P = kmeans_pool(zscore(X), np.random.default_rng(SEED))
            res = {'CA kNN-50': [], 'CA kNN-50 + 1-hop': []}
            for r in range(RUNS):
                sel = np.random.default_rng(100 + r).choice(POOL, MES, replace=False)
                s = knn_score(co_association(P[:, sel]), k)
                res['CA kNN-50'].append(metrics(y, flag_top(s, C)))
                res['CA kNN-50 + 1-hop'].append(metrics(y, flag_top(onehop(s, nb), C)))
            for mth, v in res.items():
                rows.append(dict(set=set_name, circuit=c, method=mth, n_feat=len(feats),
                                 **{q: float(np.mean([d[q] for d in v])) for q in v[0]}))
            print(f'{set_name:6s} {c:12s} n={n:5d} feats={len(feats):2d} '
                  f'F={100*np.mean([d["F"] for d in res["CA kNN-50"]]):5.1f} '
                  f'F+1hop={100*np.mean([d["F"] for d in res["CA kNN-50 + 1-hop"]]):5.1f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        R = pd.DataFrame(rows)
        R.to_csv(os.path.join(out_dir, f'fs_{set_name}_per_circuit.csv'), index=False)
        json.dump(chosen, open(os.path.join(out_dir, f'fs_{set_name}_features.json'), 'w'), indent=1)
        summ = R.groupby('method')[['TPR', 'TNR', 'P', 'F', 'A']].mean().mul(100).round(2)
        print(f'== {set_name} done in {time.time()-t_set:.0f}s\n{summ.to_string()}\n', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
