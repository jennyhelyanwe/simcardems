"""
Standalone, single-process entry point for single-cell pacing — run as
a subprocess (never in-process under mpirun), since cbcbeat's internal
SingleCellSolver mesh construction uses MPI.comm_world collectively and
cannot handle a 1-cell mesh split across more ranks than it has cells.
"""
import argparse
import json
import sys

import cbcbeat
import dolfin

from simcardems.models.fully_coupled_Tor_Land.cell_model import TorLandFull


class _StubCoupling:
    def __init__(self):
        self.lmbda_ep = dolfin.Constant(1.0)
        self.Zetas_ep = dolfin.Constant(0.0)
        self.Zetaw_ep = dolfin.Constant(0.0)


def pace_one_celltype(cell_params, celltype, pcl, max_beats, tol, dt,
                       stim_amp=53.0, stim_duration=1.0):
    import numpy as np

    params = dict(cell_params)
    params["celltype"] = celltype
    full_params = TorLandFull.default_parameters()
    full_params.update(params)

    stub_coupling = _StubCoupling()
    cell = TorLandFull(stub_coupling, params=full_params, init_conditions=None)

    time = dolfin.Constant(0.0)
    stimulus_expr = dolfin.Expression(
        "(fmod(time, pcl) >= start) && (fmod(time, pcl) <= start + duration) "
        "? amplitude : 0.0",
        time=time, pcl=pcl, start=0.0, duration=stim_duration, amplitude=stim_amp,
        degree=0,
    )
    cell.stimulus = stimulus_expr

    scheme_params = cbcbeat.SingleCellSolver.default_parameters()
    scheme_params["scheme"] = "GRL1"
    solver = cbcbeat.SingleCellSolver(cell, time, params=scheme_params)
    (vs_, vs) = solver.solution_fields()
    vs_.assign(cell.initial_conditions())

    state_names = list(TorLandFull.default_initial_conditions().keys())
    v_idx, ca_idx, xs_idx = (state_names.index(n) for n in ("v", "cai", "XS"))

    full_t, full_v, full_ca, full_xs = [], [], [], []
    last_beat_t = last_beat_v = last_beat_ca = last_beat_xs = None
    prev_ca_trace = prev_t_trace = None
    beats_used, final_delta = max_beats, float("nan")

    for beat in range(max_beats):
        t0, t1 = beat * pcl, (beat + 1) * pcl
        ca_trace, v_trace, xs_trace, t_trace = [], [], [], []

        for (interval, fields) in solver.solve((t0, t1), dt):
            arr = vs.vector().get_local()
            v_trace.append(arr[v_idx])
            ca_trace.append(arr[ca_idx])
            xs_trace.append(arr[xs_idx])
            t_trace.append(interval[1])

        ca_trace = np.array(ca_trace)
        t_trace = np.array(t_trace)

        full_t.extend(t_trace.tolist())
        full_v.extend(v_trace)
        full_ca.extend(ca_trace.tolist())
        full_xs.extend(xs_trace)
        last_beat_t = (t_trace - t_trace[0]).tolist()
        last_beat_v, last_beat_ca, last_beat_xs = v_trace, ca_trace.tolist(), xs_trace

        if prev_ca_trace is not None:
            prev_ca_resampled = np.interp(t_trace - t_trace[0],
                                           prev_t_trace - prev_t_trace[0], prev_ca_trace)
            ca_range = max(prev_ca_resampled.max() - prev_ca_resampled.min(), 1e-9)
            rel_diff = float(np.max(np.abs(ca_trace - prev_ca_resampled)) / ca_range)
            print(f"[{celltype}] beat {beat+1}: Ca dev={rel_diff:.3e}", file=sys.stderr)
            if rel_diff < tol:
                beats_used, final_delta = beat + 1, rel_diff
                break

        prev_ca_trace, prev_t_trace = ca_trace, t_trace
        vs_.assign(vs)

    final_state = {n: float(v) for n, v in zip(state_names, vs.vector().get_local())}

    return {
        "final_state": final_state, "beats_used": beats_used, "final_delta": final_delta,
        "full_t": full_t, "full_v": full_v, "full_ca": full_ca, "full_xs": full_xs,
        "last_beat_t": last_beat_t, "last_beat_v": last_beat_v,
        "last_beat_ca": last_beat_ca, "last_beat_xs": last_beat_xs,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cell-params-json", required=True)
    p.add_argument("--celltype", type=int, required=True)
    p.add_argument("--pcl", type=float, required=True)
    p.add_argument("--max-beats", type=int, required=True)
    p.add_argument("--tol", type=float, required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    cell_params = json.loads(args.cell_params_json)
    result = pace_one_celltype(cell_params, args.celltype, args.pcl,
                                args.max_beats, args.tol, args.dt)
    with open(args.out_json, "w") as f:
        json.dump(result, f)