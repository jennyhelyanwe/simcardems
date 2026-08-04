"""
simulations/rodero_05/pace_and_cache_steady_state.py

Standalone script: paces TorLandFull for all three celltypes to steady
state, saves the cache + diagnostic figure. Run with plain `python3`
(never mpirun) — cbcbeat's SingleCellSolver builds its internal mesh on
MPI.comm_world collectively and cannot handle a 1-cell mesh under
multiple ranks.

Usage:
    python3 pace_and_cache_steady_state.py --pcl 800 --max-beats 300 --tol 1e-3
"""
import argparse
import json

from simcardems.models.fully_coupled_Tor_Land.cell_model import TorLandFull
from simcardems.steady_state_cell_cache import (
    CELLTYPE_NAMES, _run_key, _run_dir, _plot_diagnostic,
)
from simcardems._pace_single_cell_cli import pace_one_celltype


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pcl", type=float, default=800.0)
    p.add_argument("--max-beats", type=int, default=300)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--cell-params-json", default="{}")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cell_params = json.loads(args.cell_params_json)

    full_params = TorLandFull.default_parameters()
    full_params.update(cell_params)
    land_params = {"Tref": full_params["Tref"], "rs": full_params["rs"]}

    key = _run_key(cell_params, args.pcl, args.max_beats, args.tol, args.dt)
    run_dir = _run_dir(key)
    png_path = run_dir / "diagnostic.png"
    state_paths = {ct: run_dir / f"celltype{ct}_state.json" for ct in CELLTYPE_NAMES}

    if not args.force and png_path.exists() and all(p.exists() for p in state_paths.values()):
        print(f"[pace_and_cache] Already cached at key={key}: {run_dir}")
        return

    print(f"[pace_and_cache] Pacing all three celltypes (key={key})...")
    results = {}
    for celltype in CELLTYPE_NAMES:
        print(f"[pace_and_cache] === {CELLTYPE_NAMES[celltype]} ===")
        result = pace_one_celltype(cell_params, celltype, args.pcl, args.max_beats,
                                    args.tol, args.dt)
        result["celltype"] = celltype
        result["name"] = CELLTYPE_NAMES[celltype]
        for k in ("full_t", "full_v", "full_ca", "full_xs",
                  "last_beat_t", "last_beat_v", "last_beat_ca", "last_beat_xs"):
            import numpy as np
            result[k] = np.array(result[k])
        results[celltype] = result

        with open(state_paths[celltype], "w") as f:
            json.dump(result["final_state"], f, indent=2)
        with open(run_dir / f"celltype{celltype}_meta.json", "w") as f:
            json.dump({
                "celltype": celltype, "name": result["name"],
                "beats_used": result["beats_used"],
                "final_ca_relative_deviation": result["final_delta"],
                "converged": result["beats_used"] < args.max_beats,
            }, f, indent=2)

    _plot_diagnostic(results, land_params, args.pcl, png_path)
    print(f"[pace_and_cache] Done. Saved: {run_dir}")
    print(f"[pace_and_cache] Diagnostic figure: {png_path}")


if __name__ == "__main__":
    main()