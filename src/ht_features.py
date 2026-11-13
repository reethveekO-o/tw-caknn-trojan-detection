"""
Hardware-Trojan feature extraction for the 15 Trust-Hub / TRIT-TS circuits
used in Nie et al. (J. Supercomputing 2025, "Stacking ensemble feature-attention
LOFs for detecting hardware Trojan").

Produces, per internal net of each gate-level netlist:
  structural_features.csv : circuit, net, label, f0..f55   (Table 3 of the paper)
  testability_features.csv: circuit, net, label, CC0, CC1, CO, SC0, SC1, SO (SCOAP)

Fixes relative to the earlier shared loader:
  * // and /* */ comments are removed before parsing. In the s-series TjIn
    netlists the Trojan gates follow "// Trigger" / "// Payload" comments,
    and those gates were silently dropped, leaving the Trojan disconnected.
  * Unconnected ports such as .QN() are ignored (they used to create one fake
    net named '' with thousands of fan-ins).
  * in_gate_x (f51-f55) is computed separately from fan_in_x (f0-f4).
  * SCOAP is computed from each cell's logic function with a configurable cap.

Usage (Colab or local):
  python ht_features.py /path/to/Stacking_Ensemble_TH  [out_dir]  [scoap_cap]
or import and call  extract_all(dataset_root, out_dir, scoap_cap)
"""
import os, re, sys, itertools, collections
from functools import lru_cache
import pandas as pd

INF = float('inf')
SENTINEL = 99            # "not reachable" value for the nearest-* features
MAX_LEVEL = 5            # x in [1, 5]

# =====================================================================
# 1. Netlist parsing
# =====================================================================
def strip_comments(raw):
    raw = re.sub(r'/\*.*?\*/', ' ', raw, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', raw)


def parse_statements(raw):
    text = re.sub(r'\s+', ' ', strip_comments(raw))
    return [s.strip() for s in text.split(';') if s.strip()]


def parse_verilog_netlist(filepath):
    """Returns (primary_inputs, primary_outputs, gates), gates = list of
    (cell_type, instance_name, {port: net})."""
    with open(filepath, 'r', errors='replace') as f:
        stmts = parse_statements(f.read())
    pis, pos, gates = [], [], []
    for st in stmts:
        if st.startswith('module') or st.startswith('endmodule'):
            continue
        if st.startswith('input ') or st.startswith('output '):
            d = 'input' if st.startswith('input') else 'output'
            body = st[len(d):].strip()
            m = re.match(r'\[(\d+):(\d+)\]\s*(.*)', body)
            names = [n.strip() for n in (m.group(3) if m else body).split(',') if n.strip()]
            for name in names:
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    for i in range(min(a, b), max(a, b) + 1):
                        (pis if d == 'input' else pos).append(f'{name}[{i}]')
                else:
                    (pis if d == 'input' else pos).append(name)
            continue
        if st.startswith('wire ') or st.startswith('reg '):
            continue
        if st.startswith('assign '):
            lhs, rhs = st[len('assign'):].split('=', 1)
            gates.append(('ASSIGN', None, {'IN': rhs.strip(), 'Q': lhs.strip()}))
            continue
        m = re.match(r'^(\S+)\s+(\S+)\s*\((.*)\)$', st)
        if not m:
            continue
        cell, inst, body = m.groups()
        ports = {}
        for pm in re.finditer(r'\.(\w+)\s*\(\s*(.*?)\s*\)', body):
            if pm.group(2) != '':
                ports[pm.group(1)] = pm.group(2)
        gates.append((cell, inst, ports))
    return pis, pos, gates


# =====================================================================
# 2. Cell library: logic function of every cell type in the dataset
#    (SAED 90nm for Trust-Hub, the TRIT-TS library for s1423).
# =====================================================================
def _and(*v): return int(all(v))
def _or(*v): return int(any(v))
def _xor(*v): return sum(v) % 2

def comb_function(cell):
    """Return (input_ports, f) for a combinational cell, or None."""
    c = cell.upper()
    base = re.sub(r'X\d+$', '', c)                        # SAED drive strength
    base = re.sub(r'S\d+$', '', base)                     # TRIT drive strength (nnd2s1)
    def ins(k, p='IN'): return [f'{p}{i}' for i in range(1, k + 1)]
    trit = cell[0].islower()
    P = 'DIN' if trit else 'IN'
    table = {
        'AND': lambda x: _and(*x), 'NAND': lambda x: 1 - _and(*x), 'NND': lambda x: 1 - _and(*x),
        'OR': lambda x: _or(*x), 'NOR': lambda x: 1 - _or(*x),
        'XOR': lambda x: _xor(*x), 'XNOR': lambda x: 1 - _xor(*x), 'XNR': lambda x: 1 - _xor(*x),
    }
    m = re.match(r'^(AND|NAND|NND|OR|NOR|XOR|XNOR|XNR)(\d)$', base)
    if m:
        k = int(m.group(2)); f = table[m.group(1)]
        return ins(k, P), lambda d, k=k, f=f: f([d[p] for p in ins(k, P)])
    # AND-OR / OR-AND families: digits give the group sizes, e.g. AO221 = (2,2,1)
    m = re.match(r'^(AOI|AO|OAI|OA)(\d+)$', base)
    if m:
        kind, groups = m.group(1), [int(g) for g in m.group(2)]
        k = sum(groups); names = ins(k, P)
        def f(d, kind=kind, groups=groups, names=names):
            vals = [d[p] for p in names]; out, i = [], 0
            for g in groups:
                part = vals[i:i + g]; i += g
                out.append(_and(*part) if kind.startswith('AO') else _or(*part))
            r = _or(*out) if kind.startswith('AO') else _and(*out)
            return 1 - r if kind.endswith('I') else r
        return names, f
    if base in ('INV', 'I1', 'IB1', 'HI1'):               # inverters
        p = 'DIN' if trit else 'IN'
        return [p], lambda d, p=p: 1 - d[p]
    if base in ('NBUFF', 'BUF', 'LSDN', 'ASSIGN'):
        p = {'LSDN': 'D', 'ASSIGN': 'IN'}.get(base, 'IN')
        return [p], lambda d, p=p: d[p]
    if base == 'LSDNEN':
        return ['D', 'ENB'], lambda d: _and(d['D'], d['ENB'])
    if base == 'ISOLAND':
        return ['D', 'ISO'], lambda d: _and(d['D'], d['ISO'])
    if base == 'ISOLOR':
        return ['D', 'ISO'], lambda d: _or(d['D'], d['ISO'])
    if base == 'MUX21':
        return ['IN1', 'IN2', 'S'], lambda d: d['IN2'] if d['S'] else d['IN1']
    if base == 'MXI21':
        return ['DIN1', 'DIN2', 'SIN'], lambda d: 1 - (d['DIN2'] if d['SIN'] else d['DIN1'])
    return None


def ff_function(cell):
    """Return (data_inputs, next_state f) for a flip-flop, or None.
    Clock pins are ignored; async set/reset and scan muxes are folded in."""
    c = cell.upper()
    if cell.startswith('sdff'):  # TRIT scan FF (lower-case library): checked before SAED's SDFF
        return ['DIN', 'SDIN', 'SSEL'], lambda d: d['SDIN'] if d['SSEL'] else d['DIN']
    if c.startswith('SDFF'):     # SAED scan FF: Q+ = SE ? SI : D
        return ['D', 'SI', 'SE'], lambda d: d['SI'] if d['SE'] else d['D']
    if c.startswith('DFFAR'):    # async reset, active low
        return ['D', 'RSTB'], lambda d: _and(d['D'], d['RSTB'])
    if c.startswith('DFFAS'):    # async set, active low
        return ['D', 'SETB'], lambda d: _or(d['D'], 1 - d['SETB'])
    if cell.startswith('dffles'):  # enable (active-low EB): approximated by DIN
        return ['DIN', 'EB'], lambda d: _and(d['DIN'], 1 - d['EB']) if False else d['DIN']
    if cell.startswith('dff'):
        return ['DIN'], lambda d: d['DIN']
    if c.startswith('DFF'):
        return ['D'], lambda d: d['D']
    return None


def is_mux(cell): return bool(re.match(r'^(MUX|mxi|mux)', cell, re.I))
def is_ff(cell): return ff_function(cell) is not None


@lru_cache(maxsize=None)
def cube_tables(cell):
    """Minimal input cubes (partial assignments) for SCOAP.
    Returns (inputs, force[0|1] -> list of cubes, sens[i] -> list of cubes)."""
    spec = comb_function(cell) or ff_function(cell)
    names, f = spec
    k = len(names)
    def ev(assign):
        free = [p for p in names if p not in assign]
        outs = set()
        for bits in itertools.product((0, 1), repeat=len(free)):
            d = dict(assign); d.update(zip(free, bits)); outs.add(f(d))
            if len(outs) > 1: return None
        return outs.pop()
    def minimal(cubes):
        cubes = sorted(cubes, key=len); keep = []
        for c in cubes:
            if not any(set(k.items()) <= set(c.items()) for k in keep): keep.append(c)
        return keep
    force = {0: [], 1: []}
    for vals in itertools.product((None, 0, 1), repeat=k):
        a = {p: v for p, v in zip(names, vals) if v is not None}
        r = ev(a)
        if r is not None: force[r].append(a)
    force = {v: minimal(force[v]) for v in (0, 1)}
    sens = {}
    for p in names:
        others = [q for q in names if q != p]; cubes = []
        for vals in itertools.product((None, 0, 1), repeat=len(others)):
            a = {q: v for q, v in zip(others, vals) if v is not None}
            a0 = dict(a); a0[p] = 0; a1 = dict(a); a1[p] = 1
            r0, r1 = ev(a0), ev(a1)
            if r0 is not None and r1 is not None and r0 != r1: cubes.append(a)
        sens[p] = minimal(cubes)
    return names, force, sens


# =====================================================================
# 3. Circuit model
# =====================================================================
class Circuit:
    def __init__(self, pis, pos, gates):
        self.pis, self.pos = set(pis), set(pos)
        self.driver = {}          # net -> gate index
        self.gates = []           # (cell, inst, {inport: net}, {outport: net})
        self.fanout = collections.defaultdict(list)   # net -> [(gate idx, port)]
        self.consts = {}
        cc = 0
        for cell, inst, ports in gates:
            ins, outs = {}, {}
            for p, net in ports.items():
                if p in ('Q', 'QN'):
                    outs[p] = net
                elif p == 'CLK':
                    continue          # clock pins carry no logic value
                else:
                    if net in ("1'b0", "1'b1"):
                        cc += 1; name = f'__const{cc}_{net[-1]}'
                        self.consts[name] = int(net[-1]); net = name
                    ins[p] = net
            gi = len(self.gates)
            self.gates.append((cell, inst, ins, outs))
            for net in outs.values(): self.driver[net] = gi
            for p, net in ins.items(): self.fanout[net].append((gi, p))
        # net-level adjacency (net -> nets) used by the structural features
        self.succ = collections.defaultdict(set); self.pred = collections.defaultdict(set)
        self.clk = {}             # gate index -> clock net (wiring only, no logic value)
        for gi, (cell, inst, ports) in enumerate(gates):
            if 'CLK' in ports: self.clk[gi] = ports['CLK']
        for gi, (cell, inst, ins, outs) in enumerate(self.gates):
            srcs = list(ins.values()) + ([self.clk[gi]] if gi in self.clk else [])
            for o in outs.values():
                for i in srcs:
                    self.succ[i].add(o); self.pred[o].add(i)
        self.nets = set(self.driver) | set(self.fanout) | self.pis | self.pos

    def cell_of(self, net):
        g = self.driver.get(net)
        return self.gates[g][0] if g is not None else None

    def internal_nets(self):
        """Nets driven by a gate that are not primary inputs/outputs."""
        return [n for n in self.driver if n not in self.pis and n not in self.pos
                and self.gates[self.driver[n]][0] != 'ASSIGN' or
                (n in self.driver and self.gates[self.driver[n]][0] == 'ASSIGN'
                 and n not in self.pos and n not in self.pis)]


# =====================================================================
# 4. Structural features (Table 3: f0-f55)
# =====================================================================
def bfs_levels(start, adj, max_level):
    """levels[x] = set of nets exactly x levels away (x = 1..max_level);
    also counts re-visits of already-seen nets (loops)."""
    seen = {start}; frontier = {start}; levels = []; loops = []
    for _ in range(max_level):
        nxt = set(); rev = 0
        for u in frontier:
            for v in adj.get(u, ()):
                if v in seen: rev += 1
                else: nxt.add(v)
        seen |= nxt; levels.append(nxt); loops.append(rev); frontier = nxt
    return levels, loops


def multi_source_dist(sources, adj):
    dist = {s: 0 for s in sources}; q = collections.deque(sources)
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return dist


def structural_features(C, nets):
    ff_nets = {n for n in C.driver if is_ff(C.cell_of(n))}
    mux_nets = {n for n in C.driver if is_mux(C.cell_of(n))}
    const = set(C.consts)
    # nearest-* distances with one multi-source BFS each (levels count gates)
    d_pi = multi_source_dist(list(C.pis), C.succ)                  # PI -> n
    d_po = multi_source_dist(list(C.pos), C.pred)                  # n -> PO
    d_ff_in = multi_source_dist(list(ff_nets), C.succ)             # FF output -> n
    d_ff_out = multi_source_dist([C.gates[g][3].get('Q') or next(iter(C.gates[g][3].values()))
                                  for g in {C.driver[n] for n in ff_nets}], C.pred)
    d_mx_in = multi_source_dist(list(mux_nets), C.succ)
    d_mx_out = multi_source_dist(list(mux_nets), C.pred)
    rows = []
    for n in nets:
        f = [0] * 56
        inl, inloop = bfs_levels(n, C.pred, MAX_LEVEL)
        outl, outloop = bfs_levels(n, C.succ, MAX_LEVEL)
        cum_in, cum_out = set(), set()
        for x in range(MAX_LEVEL):
            cum_in |= inl[x]; cum_out |= outl[x]
            f[0 + x] = len(inl[x])                                        # fan_in_x
            f[5 + x] = len(cum_in & ff_nets)                              # in_flipflop_x
            f[10 + x] = len(cum_out & ff_nets)                            # out_flipflop_x
            f[15 + x] = len(cum_in & mux_nets)                            # in_multiplexer_x
            f[20 + x] = len(cum_out & mux_nets)                           # out_multiplexer_x
            f[25 + x] = sum(inloop[:x + 1])                               # in_loop_x
            f[30 + x] = sum(outloop[:x + 1])                              # out_loop_x
            f[35 + x] = len(cum_in & const)                               # in_const_x
            f[40 + x] = 0                                                 # out_const_x (constants are sources)
            # in_gate_x: distinct gates x levels away on the input side
            if x == 0:
                f[51] = 1 if n in C.driver else 0
            else:
                f[51 + x] = len({C.driver[v] for v in inl[x - 1] if v in C.driver})
        f[45] = d_pi.get(n, SENTINEL)                                     # in_nearest_pin
        f[46] = d_po.get(n, SENTINEL)                                     # out_nearest_pin
        f[47] = d_ff_in[n] + 1 if n in d_ff_in else SENTINEL              # in_nearest_flipflop
        f[48] = d_ff_out[n] if n in d_ff_out and d_ff_out[n] > 0 else (
            min((d_ff_out.get(s, INF) + 1 for s in C.succ.get(n, ())), default=SENTINEL))
        f[48] = SENTINEL if f[48] == INF else int(f[48])                   # out_nearest_flipflop
        f[49] = d_mx_in[n] + 1 if n in d_mx_in else SENTINEL              # in_nearest_multiplexer
        f[50] = d_mx_out[n] if n in d_mx_out and d_mx_out[n] > 0 else (
            min((d_mx_out.get(s, INF) + 1 for s in C.succ.get(n, ())), default=SENTINEL))
        f[50] = SENTINEL if f[50] == INF else int(f[50])                   # out_nearest_multiplexer
        rows.append(f)
    return rows


# =====================================================================
# 5. SCOAP testability (CC0, CC1, CO, SC0, SC1, SO)
# =====================================================================
def scoap(C, cap):
    CC0, CC1, SC0, SC1 = {}, {}, {}, {}
    get = lambda d, n: d.get(n, INF)
    for p in C.pis: CC0[p] = CC1[p] = 1; SC0[p] = SC1[p] = 0
    for c, v in C.consts.items():
        CC0[c], CC1[c] = (1, INF) if v == 0 else (INF, 1); SC0[c] = SC1[c] = 0
    tables = [cube_tables(g[0]) if (comb_function(g[0]) or ff_function(g[0])) else None for g in C.gates]
    def cube_cost(cube, ins, CCd, SCd):
        cc = sc = 0
        for p, v in cube.items():
            net = ins.get(p)
            if net is None: continue          # unconnected optional pin
            cc += get(CCd[v], net); sc += get(SCd[v], net)
        return cc, sc
    # ---- controllability: relax to a fixed point (loops through flip-flops)
    for _ in range(400):
        changed = False
        for gi, (cell, inst, ins, outs) in enumerate(C.gates):
            t = tables[gi]
            if t is None: continue
            names, force, sens = t
            ff = is_ff(cell)
            val = {}
            for v in (0, 1):
                best_cc = best_sc = INF
                for cube in force[v]:
                    cc, sc = cube_cost(cube, ins, (CC0, CC1), (SC0, SC1))
                    best_cc = min(best_cc, cc); best_sc = min(best_sc, sc)
                val[v] = (best_cc + (0 if ff else 1), best_sc + (1 if ff else 0))
            for port, net in outs.items():
                v0, v1 = (val[1], val[0]) if (port == 'QN' and ff) else (val[0], val[1])
                for (d, s, new) in ((CC0, SC0, v0), (CC1, SC1, v1)):
                    if new[0] < get(d, net) - 1e-9: d[net] = new[0]; changed = True
                    if new[1] < get(s, net) - 1e-9: s[net] = new[1]; changed = True
        if not changed: break
    # ---- observability
    CO, SO = {}, {}
    for p in C.pos: CO[p] = 0; SO[p] = 0
    for _ in range(400):
        changed = False
        for gi, (cell, inst, ins, outs) in enumerate(C.gates):
            t = tables[gi]
            if t is None: continue
            names, force, sens = t
            ff = is_ff(cell)
            co_out = min((get(CO, n) for n in outs.values()), default=INF)
            so_out = min((get(SO, n) for n in outs.values()), default=INF)
            if co_out == INF: continue
            for p, net in ins.items():
                if p not in sens: continue
                best_co = best_so = INF
                for cube in sens[p]:
                    cc, sc = cube_cost(cube, ins, (CC0, CC1), (SC0, SC1))
                    best_co = min(best_co, cc); best_so = min(best_so, sc)
                new_co = co_out + best_co + (0 if ff else 1)
                new_so = so_out + best_so + (1 if ff else 0)
                if new_co < get(CO, net) - 1e-9: CO[net] = new_co; changed = True
                if new_so < get(SO, net) - 1e-9: SO[net] = new_so; changed = True
        if not changed: break
    c = lambda x: cap if x == INF or x > cap else x
    return {n: [c(get(CC0, n)), c(get(CC1, n)), c(get(CO, n)),
                c(get(SC0, n)), c(get(SC1, n)), c(get(SO, n))] for n in C.nets}


# =====================================================================
# 6. Labels (same three strategies as the shared loader)
# =====================================================================
def trojan_instances_from_text(raw, marker_list):
    pos, mk = -1, None
    for m in marker_list:
        p = raw.rfind(m)
        if p > pos: pos, mk = p, m
    if pos < 0: raise ValueError('Trojan section not found')
    names = []
    for st in parse_statements(raw[pos + len(mk):]):
        st = re.sub(r'-{5,}', ' ', st).strip()
        m = re.match(r'^(\S+)\s+(\w+)\s*\(', st)
        if m: names.append(m.group(2))
    return set(names)


CIRCUITS = [
    ('RS232-T1000', 'rs232'), ('RS232-T1100', 'rs232'), ('RS232-T1200', 'rs232'),
    ('RS232-T1300', 'rs232'), ('RS232-T1400', 'rs232'), ('RS232-T1500', 'rs232'),
    ('RS232-T1600', 'rs232'), ('s15850-T100', 'tjin'), ('s35932-T100', 'tjin'),
    ('s35932-T200', 'tjin'), ('s35932-T300', 'tjin'), ('s38417-T100', 'tjin'),
    ('s38417-T300', 'tjin'), ('s1423_T401', 'trit'), ('s1423_T402', 'trit'),
]


def find_file(root, *cands):
    for c in cands:
        p = os.path.join(root, *c)
        if os.path.exists(p): return p
    raise FileNotFoundError(f'None of {cands} under {root}')


def load_circuit(root, name, family):
    if family == 'rs232':
        base = find_file(root, (name, name), (name,))
        v = os.path.join(base, 'src', '90nm', 'uart.v')
        readme = [os.path.join(base, f) for f in os.listdir(base) if f.lower().startswith('read')][0]
        C = Circuit(*parse_verilog_netlist(v))
        troj = trojan_instances_from_text(open(readme, errors='replace').read(),
                                          ['Trojan Circuit', 'Trojan Taxonomy'])
        label = lambda n: int(C.gates[C.driver[n]][1] in troj)
    elif family == 'trit':
        base = find_file(root, (name,), (name, name))
        C = Circuit(*parse_verilog_netlist(os.path.join(base, f'{name}.v')))
        raw = open(os.path.join(base, 'log.txt'), errors='replace').read()
        troj = trojan_instances_from_text(raw, ['TROJAN BODY'])
        label = lambda n: int(C.gates[C.driver[n]][1] in troj)
    else:
        base = find_file(root, (name, name), (name,))
        scan = name.split('-')[0] + '_scan.v'
        C = Circuit(*parse_verilog_netlist(os.path.join(base, 'src', 'TjIn', scan)))
        F = Circuit(*parse_verilog_netlist(os.path.join(base, 'src', 'TjFree', scan)))
        free = set(F.driver)
        label = lambda n: int(n not in free)
    nets = sorted(n for n in C.driver if n not in C.pis and n not in C.pos)
    return C, nets, [label(n) for n in nets]


def extract_all(root, out_dir='.', scoap_cap=1000, verbose=True):
    S, T = [], []
    for name, fam in CIRCUITS:
        C, nets, y = load_circuit(root, name, fam)
        feats = structural_features(C, nets)
        sc = scoap(C, scoap_cap)
        for n, lab, f in zip(nets, y, feats):
            S.append([name, n, lab] + f); T.append([name, n, lab] + sc[n])
        if verbose:
            print(f'{name:13s} nets={len(nets):5d}  trojan={sum(y):3d}', flush=True)
    S = pd.DataFrame(S, columns=['circuit', 'net', 'label'] + [f'f{i}' for i in range(56)])
    T = pd.DataFrame(T, columns=['circuit', 'net', 'label', 'CC0', 'CC1', 'CO', 'SC0', 'SC1', 'SO'])
    os.makedirs(out_dir, exist_ok=True)
    S.to_csv(os.path.join(out_dir, 'structural_features.csv'), index=False)
    T.to_csv(os.path.join(out_dir, 'testability_features.csv'), index=False)
    return S, T


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    out = sys.argv[2] if len(sys.argv) > 2 else '.'
    cap = float(sys.argv[3]) if len(sys.argv) > 3 else 1000
    extract_all(root, out, cap)
