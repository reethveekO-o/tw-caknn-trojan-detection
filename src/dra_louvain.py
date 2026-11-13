import random
from collections import defaultdict
import numpy as np, pandas as pd, networkx as nx
from networkx.algorithms.community import modularity
GAMMA_GRID = [0.5, 1, 2, 4, 6, 8, 10, 15, 20]
MAX_FRAC = 0.30
def norm(s):
    return str(s).strip().lstrip('\\').strip()

def _one_level(G, m, partition, gamma, directed, rng):
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner = [{u} for u in G.nodes()]
    if directed:
        kin = dict(G.in_degree(weight='weight')); kout = dict(G.out_degree(weight='weight'))
        Sin, Sout = list(kin.values()), list(kout.values())
        nbrs = {}
        for u in G:
            nbrs[u] = defaultdict(float)
            for _, v, w in G.out_edges(u, data='weight'):
                if u != v: nbrs[u][v] += w
            for v, _, w in G.in_edges(u, data='weight'):
                if u != v: nbrs[u][v] += w
    else:
        deg = dict(G.degree(weight='weight')); Stot = list(deg.values())
        nbrs = {u: {v: d['weight'] for v, d in G[u].items() if v != u} for u in G}
    order = list(G.nodes); rng.shuffle(order)
    moves, improved, sweeps = 1, False, 0
    while moves > 0 and sweeps < 100:          # sweep cap
        sweeps += 1; moves = 0
        for u in order:
            best_gain, best = 0, node2com[u]
            w2c = defaultdict(float)
            for v, w in nbrs[u].items(): w2c[node2com[v]] += w
            if directed:
                di, do = kin[u], kout[u]
                Sin[best] -= di; Sout[best] -= do
                remove = -w2c[best] / m + gamma * (do * Sin[best] + di * Sout[best]) / m ** 2
            else:
                d = deg[u]; Stot[best] -= d
                remove = -w2c[best] / m + gamma * (Stot[best] * d) / (2 * m ** 2)
            for c, w in w2c.items():
                if directed:
                    gain = remove + w / m - gamma * (do * Sin[c] + di * Sout[c]) / m ** 2
                else:
                    gain = remove + w / m - gamma * (Stot[c] * d) / (2 * m ** 2)
                if gain > best_gain + 1e-12:     # a real gain, not rounding noise
                    best_gain, best = gain, c
            if directed: Sin[best] += di; Sout[best] += do
            else: Stot[best] += d
            if best != node2com[u]:
                com = G.nodes[u].get('nodes', {u})
                partition[node2com[u]].difference_update(com); inner[node2com[u]].remove(u)
                partition[best].update(com); inner[best].add(u)
                improved = True; moves += 1; node2com[u] = best
    return [p for p in partition if p], [p for p in inner if p], improved

def _aggregate(G, partition):
    H = G.__class__(); node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for u in part:
            node2com[u] = i; nodes.update(G.nodes[u].get('nodes', {u}))
        H.add_node(i, nodes=nodes)
    for u, v, w in G.edges(data='weight'):
        a, b = node2com[u], node2com[v]
        H.add_edge(a, b, weight=w + (H[a][b]['weight'] if H.has_edge(a, b) else 0))
    return H

def louvain(G, gamma, seed=0, threshold=1e-7):
    """Louvain on a (Di)Graph; directed graphs use directed modularity Q_dir."""
    rng = random.Random(seed)
    partition = [{u} for u in G.nodes()]
    if G.number_of_edges() == 0: return partition
    mod = modularity(G, partition, resolution=gamma, weight='weight')
    g = G.__class__(); g.add_nodes_from(G); g.add_weighted_edges_from(G.edges(data='weight', default=1))
    m = g.size(weight='weight'); directed = G.is_directed()
    partition, inner, improved = _one_level(g, m, partition, gamma, directed, rng)
    while True:
        best = [s.copy() for s in partition]
        new_mod = modularity(g, inner, resolution=gamma, weight='weight')
        if new_mod - mod <= threshold: return best
        mod = new_mod; g = _aggregate(g, inner)
        partition, inner, improved = _one_level(g, m, partition, gamma, directed, rng)
        if not improved: return [s.copy() for s in partition]

SEM_W = {'xor_mux': 2.0, 'and_or': 1.5, 'ff': 1.5, 'not_buf': 1.0, 'other': 1.0}

def sem_class(cell):
    c = str(cell).upper()
    if 'DFF' in c or 'LATCH' in c or c.startswith('LAT'): return 'ff'
    if 'XOR' in c or 'XNOR' in c or 'XNR' in c or 'MUX' in c or c.startswith('MX'): return 'xor_mux'
    if 'BUF' in c or c.startswith(('INV', 'NOT', 'I1S', 'IB1S', 'LSDN')): return 'not_buf'
    if c.startswith(('AND', 'OR', 'NAND', 'NOR', 'NND', 'NR', 'AO', 'OA', 'ISOLAND', 'ISOLOR')): return 'and_or'
    return 'other'

def build_graph(nets, edges, cells, directed=True, semantic=True):
    """Nodes = nets; edge u -> v if u drives a gate whose output is v, weighted by
    the class of the cell that drives v."""
    idx = {n: i for i, n in enumerate(nets)}
    G = nx.DiGraph() if directed else nx.Graph(); G.add_nodes_from(range(len(nets)))
    for a, b in edges:
        if a in idx and b in idx and a != b:
            w = SEM_W[sem_class(cells.get(norm(b), 'other'))] if semantic else 1.0
            u, v = idx[a], idx[b]
            if G.has_edge(u, v): G[u][v]['weight'] += w
            else: G.add_edge(u, v, weight=w)
    return G

def dra_louvain(nets, edges, cells):
    """Returns (community label per net, chosen gamma, gamma-sweep table)."""
    G = build_graph(nets, edges, cells); n = G.number_of_nodes()
    rows, labs = [], {}
    for g in GAMMA_GRID:
        comms = louvain(G, g)
        lab = np.empty(n, int)
        for k, c in enumerate(comms): lab[list(c)] = k
        Q = modularity(G, comms, weight='weight', resolution=g)
        rows.append(dict(gamma=g, Q=Q, n_comm=len(comms), largest_frac=np.bincount(lab).max() / n)); labs[g] = lab
    tab = pd.DataFrame(rows); ok = tab[tab.largest_frac <= MAX_FRAC]
    g = float(ok.loc[ok.Q.idxmax(), 'gamma']) if len(ok) else float(tab.loc[tab.largest_frac.idxmin(), 'gamma'])
    return labs[g], g, tab
