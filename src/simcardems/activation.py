"""
src/simcardems/activation_times.py

Spatially-varying endocardial activation timing for the EP stimulus,
built from a sparse per-node activation-time field defined on a subset of
mechanics-mesh nodes, interpolated onto the EP mesh via nearest-neighbor
lookup, and applied as a precomputed dolfin.Function-based stimulus
(fast: vectorized numpy per timestep, no per-point Python eval in the
EP solve's loop).
"""

from typing import List, Tuple

import dolfin
import numpy as np
from scipy.spatial import cKDTree


def load_activation_times(
    path: str,
    node_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    path: text file with rows [node_id, amplitude, activation_time,
          duration, flag, scale]
    node_coords: (n_nodes, 3) array, SAME indexing as the node IDs in
                 the file (confirmed to match for this mesh)

    Returns:
        node_ids: (n_subset,) int array
        activation_times: (n_subset,) float array
        coords_subset: (n_subset, 3) float array, node_coords[node_ids]
    """
    data = np.loadtxt(path)
    node_ids = data[:, 0].astype(int) - 1 # 1-indexed -> 0-indexed
    activation_times = data[:, 2]
    coords_subset = node_coords[node_ids]
    return node_ids, activation_times, coords_subset


def interpolate_activation_to_ep_mesh(
    ep_mesh: dolfin.Mesh,
    endo_marker_ep: List[int],
    ffun_ep: dolfin.MeshFunction,
    node_ids_mech: np.ndarray,
    activation_times_mech: np.ndarray,
    coords_subset_mech: np.ndarray,
) -> dolfin.Function:
    """
    Build a P1 dolfin.Function on the EP mesh, giving each EP-mesh
    endocardial vertex an activation time via nearest-neighbor lookup
    against the sparse mechanics-mesh activation data. Non-endocardial
    EP vertices get np.inf (never directly stimulated; activate via
    propagation instead).
    """
    V = dolfin.FunctionSpace(ep_mesh, "P", 1)
    act_fn = dolfin.Function(V)
    act_fn.vector()[:] = np.inf

    tree = cKDTree(coords_subset_mech)

    v2d = dolfin.vertex_to_dof_map(V)
    ep_mesh.init(2, 0)

    endo_vertex_ids = set()
    for facet in dolfin.facets(ep_mesh):
        if ffun_ep[facet.index()] in endo_marker_ep:
            for v in facet.entities(0):
                endo_vertex_ids.add(v)

    coords = ep_mesh.coordinates()
    for vid in endo_vertex_ids:
        point = coords[vid]
        _, idx = tree.query(point)
        nearest_time = activation_times_mech[idx]
        dof = v2d[vid]
        act_fn.vector()[dof] = nearest_time

    act_fn.vector().apply("insert")
    return act_fn


def endocardial_stimulus_domain(
    mesh: dolfin.Mesh,
    ffun: dolfin.MeshFunction,
    endo_markers: List[int],
    layer_thickness: int = 1,
):
    """
    Build a geometry.StimulusDomain marking cells adjacent to the
    endocardial facets (ENDO_LV and/or ENDO_RV), for use as the
    stimulus region instead of BaseGeometry.default_stimulus_domain's
    whole-tissue default.
    """
    from . import geometry

    tdim = mesh.topology().dim()
    cell_domain = dolfin.MeshFunction("size_t", mesh, tdim)
    cell_domain.set_all(0)

    mesh.init(tdim - 1, tdim)

    marked_cells = set()
    for facet in dolfin.facets(mesh):
        if ffun[facet.index()] in endo_markers:
            for cell in dolfin.cells(facet):
                marked_cells.add(cell.index())

    for _ in range(layer_thickness - 1):
        grown = set(marked_cells)
        for cell_idx in marked_cells:
            cell = dolfin.Cell(mesh, cell_idx)
            for facet in dolfin.facets(cell):
                for neighbor in dolfin.cells(facet):
                    grown.add(neighbor.index())
        marked_cells = grown

    marker = 1
    for cell_idx in marked_cells:
        cell_domain[cell_idx] = marker

    return geometry.StimulusDomain(domain=cell_domain, marker=marker)


class PrecomputedStimulusUpdater:
    """
    Fast per-timestep stimulus update: builds the stimulus as a plain
    dolfin.Function (consumed natively by cbcbeat/UFL forms, no per-point
    Python eval in the solve loop), and updates its DOF vector via
    vectorized numpy at each timestep.
    """

    def __init__(
        self,
        act_fn: dolfin.Function,
        amplitude: float,
        duration: float,
        PCL: float,
    ):
        V = act_fn.function_space()
        self.stimulus_fn = dolfin.Function(V, name="I_s_spatial")

        self.act_values = act_fn.vector().get_local().copy()
        self.finite_mask = np.isfinite(self.act_values)

        self.amplitude = amplitude
        self.duration = duration
        self.PCL = PCL

    def update(self, t: float) -> None:
        """Call once per EP timestep with the current simulation time."""
        t_mod = t % self.PCL

        fire = np.zeros_like(self.act_values)
        active = self.finite_mask & (self.act_values <= t_mod) & (
            t_mod <= self.act_values + self.duration
        )
        fire[active] = self.amplitude

        self.stimulus_fn.vector().set_local(fire)
        self.stimulus_fn.vector().apply("insert")

    def get_stimulus(self) -> dolfin.Function:
        return self.stimulus_fn