"""
Coordinate-based checkpointing.

dolfin.HDF5File.write/read on a MixedElement Function (vs, mech state)
relies on DOF-index correspondence between the WRITING and READING
processes' FunctionSpaces. That correspondence is NOT guaranteed across
independent process launches — SCOTCH's internal DOF reordering for mixed
spaces is a heuristic, not deterministic given the same mesh and rank
count. Confirmed directly: same-process round-trip of `vs` was
bit-perfect; a fresh process reading the same file got garbage (~1e239)
in every vs-derived field. Simple single-component scalar Functions were
unaffected — no reordering ambiguity for a trivial DOF numbering.
Fix: save/load every checkpointed Function by physical DOF coordinate.
"""

import glob
import numpy as np
import h5py
import dolfin
from scipy.spatial import cKDTree
from simcardems import utils

logger = utils.getLogger(__name__)


def _iter_scalar_leaves(fn, V, path_prefix=""):
    """Recurse on fn and V TOGETHER — no path-string reconstruction,
    no re-chaining .sub() calls after the fact (that was a bug:
    fn.sub(0, 1) parses as fn.sub(i=0, deepcopy=1), not a nested index)."""
    n_sub = V.num_sub_spaces()
    if n_sub == 0:
        yield (path_prefix or "root", fn, V)
    else:
        for i in range(n_sub):
            sub_path = f"{path_prefix}.{i}" if path_prefix else str(i)
            yield from _iter_scalar_leaves(fn.sub(i), V.sub(i), sub_path)


def save_function_by_coordinate(fn, path):
    comm = fn.function_space().mesh().mpi_comm()
    rank = dolfin.MPI.rank(comm)
    mesh = fn.function_space().mesh()

    with h5py.File(f"{path}.rank{rank}.h5", "w") as f:
        for leaf_path, fn_leaf, V_leaf in _iter_scalar_leaves(fn, fn.function_space()):
            standalone_V = dolfin.FunctionSpace(mesh, V_leaf.ufl_element())
            standalone_fn = dolfin.Function(standalone_V)
            dolfin.FunctionAssigner(standalone_V, V_leaf).assign(standalone_fn, fn_leaf)

            dofs = standalone_V.dofmap().dofs()
            if len(dofs) == 0:
                continue
            grp = f.create_group(leaf_path)
            grp.create_dataset("coords", data=standalone_V.tabulate_dof_coordinates())
            grp.create_dataset("vals", data=standalone_fn.vector().get_local())


def load_function_by_coordinate(fn, path, tol=1e-6):
    shard_paths = sorted(glob.glob(f"{path}.rank*.h5"))
    if not shard_paths:
        raise FileNotFoundError(f"No checkpoint shards found matching {path}.rank*.h5")

    saved = {}
    for shard_path in shard_paths:
        with h5py.File(shard_path, "r") as f:
            for leaf_path in f.keys():
                saved.setdefault(leaf_path, {"coords": [], "vals": []})
                saved[leaf_path]["coords"].append(f[leaf_path]["coords"][:])
                saved[leaf_path]["vals"].append(f[leaf_path]["vals"][:])

    mesh = fn.function_space().mesh()
    max_dist_seen = 0.0
    for leaf_path, fn_leaf, V_leaf in _iter_scalar_leaves(fn, fn.function_space()):
        if leaf_path not in saved:
            raise KeyError(f"Leaf '{leaf_path}' not in any checkpoint shard for {path}")

        standalone_V = dolfin.FunctionSpace(mesh, V_leaf.ufl_element())
        standalone_fn = dolfin.Function(standalone_V)
        my_coords = standalone_V.tabulate_dof_coordinates()

        saved_coords = np.vstack(saved[leaf_path]["coords"])
        saved_vals = np.concatenate(saved[leaf_path]["vals"])
        dist, nn_idx = cKDTree(saved_coords).query(my_coords)
        max_dist_seen = max(max_dist_seen, float(dist.max()) if len(dist) else 0.0)
        if np.any(dist > tol):
            logger.info(f"  [coord restore] WARNING '{leaf_path}': "
                        f"{np.sum(dist > tol)}/{len(dist)} unmatched (max dist={dist.max():.3e})")

        standalone_fn.vector().set_local(saved_vals[nn_idx])
        standalone_fn.vector().apply("insert")
        dolfin.FunctionAssigner(V_leaf, standalone_V).assign(fn_leaf, standalone_fn)

    return max_dist_seen


def _leaf_stats_direct(fn, idx):
    """Stats via the SAME FunctionAssigner-on-fn.sub(i) mechanism as
    save/load_function_by_coordinate — bypasses assign_ep() entirely,
    so this checks the real save/load path, not a separate,
    apparently partition-sensitive diagnostic copy."""
    mesh = fn.function_space().mesh()
    sub_V = fn.function_space().sub(idx)
    standalone_V = dolfin.FunctionSpace(mesh, sub_V.ufl_element())
    standalone_fn = dolfin.Function(standalone_V)
    dolfin.FunctionAssigner(standalone_V, sub_V).assign(standalone_fn, fn.sub(idx))
    return _field_stats(standalone_fn)


def _field_stats(fn, sub_idx=None, V_full=None):
    """max/min/mean of a Function (or a named sub-component of a mixed vs)."""
    if sub_idx is not None:
        dofs = V_full.sub(sub_idx).dofmap().dofs()
        vals = fn.vector().get_local(dofs) if len(dofs) > 0 else np.array([0.0])
    else:
        vals = fn.vector().get_local()
    local_max = float(vals.max()) if len(vals) > 0 else float('-inf')
    local_min = float(vals.min()) if len(vals) > 0 else float('inf')
    local_sum = float(vals.sum())
    local_n = len(vals)
    comm = dolfin.MPI.comm_world
    g_max = dolfin.MPI.max(comm, local_max)
    g_min = dolfin.MPI.min(comm, local_min)
    g_sum = dolfin.MPI.sum(comm, local_sum)
    g_n = dolfin.MPI.sum(comm, local_n)
    return {"max": g_max, "min": g_min, "mean": g_sum / g_n if g_n > 0 else 0.0}