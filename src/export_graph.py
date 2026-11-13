"""
Export the netlist wiring and driver cells used by the graph steps (Louvain communities
and 1-hop smoothing).

  net_edges.csv        : circuit, src, dst - one row per (gate input net, gate output net)
                         where both nets are scored internal nets; clock pins included (in
                         s35932-T100 the Trojan reaches the design through a flip-flop clock)
  net_driver_cells.csv : circuit, net, driver_cell - the cell driving each scored net

Uses the same parser and the same Trust-Hub directory layout as ht_features.py.

Usage: python export_graph.py /path/to/trust_hub_root [out_dir]
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ht_features as H


def netlist_path(root, name, family):
    if family == 'rs232':
        base = H.find_file(root, (name, name), (name,))
        return os.path.join(base, 'src', '90nm', 'uart.v')
    if family == 'trit':
        base = H.find_file(root, (name,), (name, name))
        return os.path.join(base, f'{name}.v')
    base = H.find_file(root, (name, name), (name,))
    return os.path.join(base, 'src', 'TjIn', name.split('-')[0] + '_scan.v')


def export(root, out_dir='.'):
    E, Dc = [], []
    for name, fam in H.CIRCUITS:
        pis, pos, gates = H.parse_verilog_netlist(netlist_path(root, name, fam))
        C, nets, _ = H.load_circuit(root, name, fam)
        keep = set(nets)
        seen = set()
        for cell, inst, ports in gates:
            outs = [n for p, n in ports.items() if p in ('Q', 'QN')]
            ins = [n for p, n in ports.items() if p not in ('Q', 'QN') and n not in ("1'b0", "1'b1")]
            for o in outs:
                for i in ins:
                    if i in keep and o in keep and (i, o) not in seen:
                        seen.add((i, o)); E.append((name, i, o))
        Dc += [(name, n, C.cell_of(n)) for n in nets]
        print(f'{name:13s} edges={len(seen):6d} nets={len(nets):5d}', flush=True)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(E, columns=['circuit', 'src', 'dst']).to_csv(os.path.join(out_dir, 'net_edges.csv'), index=False)
    pd.DataFrame(Dc, columns=['circuit', 'net', 'driver_cell']).to_csv(os.path.join(out_dir, 'net_driver_cells.csv'), index=False)


if __name__ == '__main__':
    a = sys.argv[1:]
    export(a[0] if a else '.', a[1] if len(a) > 1 else '.')
