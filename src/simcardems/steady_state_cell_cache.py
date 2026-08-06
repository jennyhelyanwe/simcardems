"""
src/simcardems/steady_state_cell_cache.py

Cache lookup + loading for single-cell steady-state initial conditions.
Pacing itself happens separately via pace_and_cache_steady_state.py
(run once, by hand, with plain python3 — never under mpirun, since
cbcbeat's SingleCellSolver builds its internal mesh on MPI.comm_world
collectively and cannot handle a 1-cell mesh split across more ranks
than it has cells). This module just reads that cache and applies it
to the real 3D EP mesh.
"""

import json
import hashlib
from pathlib import Path
from simcardems import utils

logger = utils.getLogger(__name__)

import dolfin

CACHE_ROOT = Path("steady_state_cache")
CACHE_ROOT.mkdir(exist_ok=True)

CELLTYPE_NAMES = {0: "Endo", 1: "Epi", 2: "Mid"}
CELLTYPE_COLORS = {0: "tab:blue", 1: "tab:red", 2: "tab:green"}


def _run_key(cell_params: dict, pcl: float, max_beats: int, tol: float, dt: float) -> str:
    """Hash over everything shared across all three celltypes for one
    pacing run — celltype itself is NOT part of this key, since all
    three are paced together under it."""
    payload = {
        "cell_params": {k: float(v) for k, v in sorted(cell_params.items())},
        "pcl": float(pcl), "max_beats": int(max_beats),
        "tol": float(tol), "dt": float(dt),
        "cell_model": "TorLandFull",
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _run_dir(key: str) -> Path:
    d = CACHE_ROOT / key
    d.mkdir(exist_ok=True)
    return d


def _plot_diagnostic(results: dict, land_params: dict, pcl: float, out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    Tref, rs = land_params["Tref"], land_params["rs"]

    fig = plt.figure(figsize=(20, 8))
    outer = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3)
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0], hspace=0.4)
    ax_full_v, ax_full_ca, ax_full_ta = (fig.add_subplot(inner[i]) for i in range(3))
    ax_final_v = fig.add_subplot(outer[1])
    ax_final_ca = fig.add_subplot(outer[2])
    ax_final_ta = fig.add_subplot(outer[3])

    for celltype, r in results.items():
        c = CELLTYPE_COLORS[celltype]
        name = r["name"]
        ta_full = (Tref / rs) * r["full_xs"]
        ta_last = (Tref / rs) * r["last_beat_xs"]

        ax_full_v.plot(r["full_t"], r["full_v"], color=c, label=name, linewidth=0.8)
        ax_full_ca.plot(r["full_t"], r["full_ca"], color=c, linewidth=0.8)
        ax_full_ta.plot(r["full_t"], ta_full, color=c, linewidth=0.8)

        ax_final_v.plot(r["last_beat_t"], r["last_beat_v"], color=c, label=name)
        ax_final_ca.plot(r["last_beat_t"], r["last_beat_ca"], color=c)
        ax_final_ta.plot(r["last_beat_t"], ta_last, color=c)

    ax_full_v.set_title("Full pacing history")
    ax_full_v.set_ylabel("V (mV)"); ax_full_v.legend(fontsize=8)
    ax_full_ca.set_ylabel("Cai (mM)")
    ax_full_ta.set_ylabel("Ta (kPa)"); ax_full_ta.set_xlabel("t (ms)")

    ax_final_v.set_title("Final steady-state beat — V")
    ax_final_v.set_xlabel("t (ms)"); ax_final_v.set_ylabel("V (mV)"); ax_final_v.legend(fontsize=8)
    ax_final_ca.set_title("Final beat — Cai")
    ax_final_ca.set_xlabel("t (ms)"); ax_final_ca.set_ylabel("Cai (mM)")
    ax_final_ta.set_title("Final beat — Ta")
    ax_final_ta.set_xlabel("t (ms)"); ax_final_ta.set_ylabel("Ta (kPa)")

    fig.suptitle(f"Single-cell steady-state diagnostic (PCL={pcl}ms)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_steady_state_cache(cell_params: dict, pcl: float, max_beats: int = 300,
                              tol: float = 1e-3, dt: float = 0.05) -> dict:
    """
    Looks up the cache produced by pace_and_cache_steady_state.py.
    Raises a clear error if it doesn't exist — this function NEVER
    paces anything itself; run the standalone script first.
    """
    key = _run_key(cell_params, pcl, max_beats, tol, dt)
    run_dir = _run_dir(key)
    png_path = run_dir / "diagnostic.png"
    state_paths = {ct: run_dir / f"celltype{ct}_state.json" for ct in CELLTYPE_NAMES}

    if not (png_path.exists() and all(p.exists() for p in state_paths.values())):
        raise FileNotFoundError(
            f"No steady-state cache found for key={key} (dir={run_dir}).\n"
            f"Run this first, with plain python3 (NOT mpirun):\n\n"
            f"    python3 pace_and_cache_steady_state.py "
            f"--pcl {pcl} --max-beats {max_beats} --tol {tol} --dt {dt}\n"
        )
    logger.info(f"[steady_state] Loaded cache (key={key}) from {run_dir}")

    return {
        "cell_init_files": {ct: str(p) for ct, p in state_paths.items()},
        "diagnostic_png": str(png_path),
    }


def apply_celltype_aware_initial_conditions(coupling, cell_fn, cell_init_files, state_names):
    """
    Overwrites coupling.ep_solver.vs in place: each DOF gets initial
    values from whichever celltype's steady-state cache matches that
    DOF's node, using cell_fn (already-built celltype assignment) to
    decide. cell_init_files: {celltype_int: path_to_json}.

    Safe under MPI: only operates on this rank's locally-owned cells,
    and vector().apply("insert") synchronizes ghost/shared DOFs across
    ranks at partition boundaries afterward.
    """
    celltype_states = {}
    for ct, path in cell_init_files.items():
        with open(path) as f:
            celltype_states[int(ct)] = json.load(f)

    vs = coupling.ep_solver.vs
    V = vs.function_space()
    mesh = V.mesh()

    cell_fn_arr = cell_fn.vector().get_local()
    vs_arr = vs.vector().get_local()
    dofmap = V.dofmap()

    for cell in dolfin.cells(mesh):
        ct = int(round(cell_fn_arr[cell.index()]))
        state_dict = celltype_states.get(ct, celltype_states[0])  # fallback to endo

        for i, name in enumerate(state_names):
            sub_dofs = V.sub(i).dofmap().cell_dofs(cell.index())
            for d in sub_dofs:
                if d < len(vs_arr):
                    vs_arr[d] = state_dict[name]

    vs.vector().set_local(vs_arr)
    vs.vector().apply("insert")
