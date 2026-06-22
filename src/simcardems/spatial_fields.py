"""
src/simcardems/spatial_fields.py

Generic per-node scalar/categorical field loading and EP-mesh mapping,
for cell-type (categorical, nearest-neighbor classification) and IKs
scale factor (continuous, nearest-neighbor interpolation -- consistent
with the activation-times approach, no smoothing).
"""

from typing import Optional

import dolfin
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd

def load_node_field(
    path: str,
    node_coords: np.ndarray,
    one_indexed: bool = False,
    value_column: int = 1,
):
    """
    Generic per-node field loader. Assumes a whitespace/csv-delimited
    file with node_id in column 0 and the field value in `value_column`.

    one_indexed: set True if the file's node IDs need a -1 shift to
                 align with node_coords (as confirmed needed for the
                 activation-times file; confirmed NOT needed for the
                 cell-type/IKs files, which are already 0-indexed).
    """
    df = pd.read_csv(path, header=None)
    data = df.to_numpy()
    node_ids = data[:, 0].astype(int)
    if one_indexed:
        node_ids = node_ids - 1
    values = data[:, value_column]
    coords_subset = node_coords[node_ids]
    return node_ids, values, coords_subset

def load_dense_node_field(path: str) -> np.ndarray:
    """
    For files with one value per line, implicitly indexed by line number
    (line i = node i, 0-indexed), covering the FULL node set with no
    separate node_id column -- e.g. cell-type and IKs-scale files.

    Returns: (n_nodes,) array, directly indexable by node_coords' own
    indexing (values[i] corresponds to node_coords[i]).
    """
    values = np.loadtxt(path)
    return values

def map_dense_field_to_ep_mesh(
    ep_mesh: dolfin.Mesh,
    node_coords_mech: np.ndarray,
    values_mech: np.ndarray,
) -> dolfin.Function:
    V = dolfin.FunctionSpace(ep_mesh, "P", 1)
    field_fn = dolfin.Function(V)
    tree = cKDTree(node_coords_mech)
    v2d = dolfin.vertex_to_dof_map(V)
    coords = ep_mesh.coordinates()

    # Vectorized nearest-neighbor query
    _, indices = tree.query(coords)

    # Vectorized assignment via get_local/set_local
    local_vec = field_fn.vector().get_local()
    local_size = len(local_vec)

    dofs = v2d[np.arange(ep_mesh.num_vertices())]
    mask = (dofs >= 0) & (dofs < local_size)
    local_vec[dofs[mask]] = values_mech[indices[mask]]

    field_fn.vector().set_local(local_vec)
    dolfin.MPI.comm_world.barrier()
    field_fn.vector().apply("insert")
    return field_fn

def map_dense_field_to_cell_meshfunction(
    ep_mesh: dolfin.Mesh,
    node_coords_mech: np.ndarray,
    values_mech: np.ndarray,
    label_to_celltype: dict,
) -> dolfin.MeshFunction:
    tdim = ep_mesh.topology().dim()
    cell_fn = dolfin.MeshFunction("size_t", ep_mesh, tdim)
    tree = cKDTree(node_coords_mech)
    for cell in dolfin.cells(ep_mesh):
        centroid = cell.midpoint().array()
        _, idx = tree.query(centroid)
        cell_fn[cell.index()] = label_to_celltype[int(values_mech[idx])]
    return cell_fn

def map_dense_field_to_dg0_function(
    ep_mesh: dolfin.Mesh,
    node_coords_mech: np.ndarray,
    values_mech: np.ndarray,
    label_to_celltype: dict,
) -> dolfin.Function:
    V = dolfin.FunctionSpace(ep_mesh, "DG", 0)
    fn = dolfin.Function(V)
    tree = cKDTree(node_coords_mech)
    dm = V.dofmap()

    # Vectorized: build all centroids at once
    cells = list(dolfin.cells(ep_mesh))
    centroids = np.array([c.midpoint().array() for c in cells])
    _, indices = tree.query(centroids)

    local_vec = fn.vector().get_local()
    local_size = len(local_vec)

    for i, cell in enumerate(cells):
        celltype_val = label_to_celltype[int(values_mech[indices[i]])]
        dof = dm.cell_dofs(cell.index())[0]
        if 0 <= dof < local_size:
            local_vec[dof] = celltype_val

    fn.vector().set_local(local_vec)
    dolfin.MPI.comm_world.barrier()
    fn.vector().apply("insert")
    return fn

def map_field_to_ep_mesh(
    ep_mesh: dolfin.Mesh,
    node_ids_mech: np.ndarray,
    values_mech: np.ndarray,
    coords_subset_mech: np.ndarray,
    categorical: bool = False,
    default_value: float = 0.0,
) -> dolfin.Function:
    """
    Nearest-neighbor map of a per-node field (defined on a subset or
    all of the mechanics-mesh nodes) onto every vertex of the EP mesh.

    categorical: if True, treated identically to the continuous case
                 mechanically (nearest-neighbor still just copies the
                 single nearest value, so the same logic IS correct for
                 categorical data -- the distinction only matters if
                 you ever average/interpolate, which nearest-neighbor
                 deliberately does not do). Kept as an explicit flag
                 for clarity/intent at call sites, and to guard against
                 a future change accidentally introducing smoothing for
                 a field where that would be meaningless.
    default_value: assigned to any EP-mesh vertex with no reasonably
                    close source point -- NOT currently distance-limited
                    (always finds SOME nearest point), so this is mostly
                    unused unless coords_subset_mech is empty. Kept for
                    API symmetry with interpolate_activation_to_ep_mesh's
                    np.inf default; 0.0 is a more sensible default here
                    since cell-type/IKs fields should always have full
                    coverage (no "never assigned" case, unlike the
                    deliberately-sparse endocardial activation field).
    """
    V = dolfin.FunctionSpace(ep_mesh, "P", 1)
    field_fn = dolfin.Function(V)
    field_fn.vector()[:] = default_value

    tree = cKDTree(coords_subset_mech)

    v2d = dolfin.vertex_to_dof_map(V)
    coords = ep_mesh.coordinates()

    for vid in range(ep_mesh.num_vertices()):
        point = coords[vid]
        _, idx = tree.query(point)
        nearest_value = values_mech[idx]
        dof = v2d[vid]
        field_fn.vector()[dof] = nearest_value

    field_fn.vector().apply("insert")
    return field_fn


def load_cell_type_field(path: str, node_coords: np.ndarray):
    """
    Convenience wrapper: cell type is column 1 (1=endo, 2=mid, 3=epi),
    0-indexed node IDs, no shift needed.
    """
    return load_node_field(path, node_coords, one_indexed=False, value_column=1)


def load_iks_scale_field(path: str, node_coords: np.ndarray):
    """
    Convenience wrapper: IKs scale factor, same indexing convention as
    cell type (0-indexed, presumably same file format -- CONFIRM the
    value_column index matches your actual file layout before trusting
    this default of column 1; if cell-type and IKs are in the SAME file
    as two different columns, adjust value_column per call rather than
    assuming both are column 1).
    """
    return load_node_field(path, node_coords, one_indexed=False, value_column=1)