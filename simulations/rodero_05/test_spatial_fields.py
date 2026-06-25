"""
Test script for loading cell-type and IKs-scale spatial fields,
mapping them onto the EP mesh, and visually checking the result.

EDIT the placeholders below once file format is confirmed.
"""

import numpy as np
import dolfin
import pyvista as pv
import pandas as pd

from simcardems.spatial_fields import (
    load_dense_node_field,
    map_dense_field_to_ep_mesh,
    map_dense_field_to_cell_meshfunction
)

CELL_TYPE_PATH = "./rodero_05_fine/rodero_05_fine_nodefield_cell-type.csv"
IKS_SCALE_PATH = "./rodero_05_fine/rodero_05_fine_nodefield_sf_IKs.csv"


from cardiac_geometries.geometry import Geometry
geo = Geometry.from_file("rodero_05_fine.h5")
from simcardems.bivgeometry import BiVentricularGeometry
biv_geo = BiVentricularGeometry.from_geometry(
    geo,
    ep_mesh=geo.mesh,
    ffun_ep=geo.ffun,
)
dir = './rodero_05_fine'
df = pd.read_csv(dir + '/rodero_05_fine_xyz.csv', header=None)
node_coords = df.to_numpy() * 10 # Because simcardems is in mm-kPa-ms-g units, and Alya meshes are in cm.

# ---------------------------------------------------------------------
# 1. Load raw fields
# ---------------------------------------------------------------------

ct_values = load_dense_node_field(
    path=CELL_TYPE_PATH
)

print("--- cell type raw data ---")
unique_ct, counts_ct = np.unique(ct_values, return_counts=True)
for val, cnt in zip(unique_ct, counts_ct):
    print(f"  cell_type={val}: {cnt} nodes")

iks_values = load_dense_node_field(
    path=IKS_SCALE_PATH
)
iks_fn = map_dense_field_to_ep_mesh(biv_geo.ep_mesh, node_coords, iks_values)


print("--- IKs scale raw data ---")
print("value range:", iks_values.min(), iks_values.max())
print("value mean:", iks_values.mean())
print("any NaN:", np.isnan(iks_values).any())

# ---------------------------------------------------------------------
# 2. Map onto EP mesh
# ---------------------------------------------------------------------

ct_fn = map_dense_field_to_ep_mesh(
    ep_mesh=biv_geo.ep_mesh,
    values_mech=ct_values,
    node_coords_mech=node_coords
)

iks_fn = map_dense_field_to_ep_mesh(
    ep_mesh=biv_geo.ep_mesh,
    values_mech=iks_values,
    node_coords_mech=node_coords
)

ct_vals_mapped = ct_fn.vector().get_local()
ct_vals_vertex_ordered = ct_vals_mapped[dolfin.vertex_to_dof_map(ct_fn.function_space())]

iks_vals_mapped = iks_fn.vector().get_local()

print("--- mapped onto EP mesh ---")
print("n EP-mesh vertices:", len(ct_vals_mapped))
unique_ct_mapped, counts_ct_mapped = np.unique(ct_vals_mapped, return_counts=True)
for val, cnt in zip(unique_ct_mapped, counts_ct_mapped):
    print(f"  cell_type={val}: {cnt} vertices")
print("IKs mapped range:", iks_vals_mapped.min(), iks_vals_mapped.max())
print("IKs mapped mean:", iks_vals_mapped.mean())

# ---------------------------------------------------------------------
# 3. Visual check via pyvista (offscreen, matching earlier pattern)
# ---------------------------------------------------------------------

def plot_field_on_mesh(mesh, values, title, output_path, cmap="viridis"):
    coords = mesh.coordinates()
    cells = mesh.cells()

    V = dolfin.FunctionSpace(mesh, "P", 1)
    v2d = dolfin.vertex_to_dof_map(V)
    values = values[v2d]  # convert DOF-ordered -> vertex-ordered

    n_cells = cells.shape[0]
    cell_array = np.hstack(
        [np.full((n_cells, 1), 4), cells]
    ).astype(np.int64).flatten()
    cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cell_array, cell_types, coords)
    grid.point_data[title] = values

    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.add_mesh(
        grid,
        scalars=title,
        cmap=cmap,
        scalar_bar_args={"title": title},
    )
    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.screenshot(output_path)
    print(f"Saved {output_path}")

    # Also clip through the mesh to see transmural variation (cell type
    # and IKs scale should both vary endo-to-epi, which a surface-only
    # plot won't show)
    plotter2 = pv.Plotter(off_screen=True, window_size=[1200, 900])
    clipped = grid.clip(normal="x", origin=grid.center)
    plotter2.add_mesh(clipped, scalars=title, cmap=cmap, scalar_bar_args={"title": title})
    plotter2.add_axes()
    plotter2.camera_position = "iso"
    plotter2.screenshot(output_path.replace(".png", "_clipped.png"))
    print(f"Saved {output_path.replace('.png', '_clipped.png')}")

#
# plot_field_on_mesh(biv_geo.ep_mesh, ct_vals_mapped, "cell_type", "cell_type.png", cmap="Set1")
# plot_field_on_mesh(biv_geo.ep_mesh, iks_vals_mapped, "iks_scale", "iks_scale.png", cmap="viridis")

cell_fn = map_dense_field_to_cell_meshfunction(
    biv_geo.ep_mesh, node_coords, ct_values, {1: 0, 2: 2, 3: 1}
)

vals = cell_fn.array()
unique, counts = np.unique(vals, return_counts=True)
for v, c in zip(unique, counts):
    print(f"celltype={v}: {c} cells")
print("n cells total:", biv_geo.ep_mesh.num_cells())

cellmodel = cbcbeat.MultiCellModel(models=(endo_model, epi_model, mid_model), keys=(0, 1, 2), markers=cell_fn)
print(cellmodel.num_states())
# need a representative v, s to test F/I -- e.g. from default initial conditions
ics = cellmodel.models()[0].initial_conditions()
