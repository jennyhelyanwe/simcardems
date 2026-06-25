import numpy as np
import pandas as pd
import dolfin
from simcardems.activation import load_activation_times, interpolate_activation_to_ep_mesh

# 1. Load the raw activation-time file
dir = './rodero_05_fine'
df = pd.read_csv(dir + '/rodero_05_fine_xyz.csv', header=None)
node_coords = df.to_numpy() * 10 # Because simcardems is in mm-kPa-ms-g units, and Alya meshes are in cm.

path = "./heart.endocardial-activation-times"
node_ids, activation_times, coords_subset = load_activation_times(path, node_coords)
activation_times = activation_times * 1000.0  # s -> ms

print("n entries in file:", len(node_ids))
print("node_id range:", node_ids.min(), node_ids.max())
print("activation_time range:", activation_times.min(), activation_times.max())
print("activation_time mean:", activation_times.mean())
print("any NaN in activation_times:", np.isnan(activation_times).any())
print("first 5 node_ids:", node_ids[:5])
print("first 5 activation_times:", activation_times[:5])
print("first 5 coords:", coords_subset[:5])

from cardiac_geometries.geometry import Geometry
geo = Geometry.from_file("rodero_05_fine.h5")
from simcardems.bivgeometry import BiVentricularGeometry
biv_geo = BiVentricularGeometry.from_geometry(
    geo,
    ep_mesh=geo.mesh,
    ffun_ep=geo.ffun,
)

lv_marker = biv_geo.markers["ENDO_LV"][0]
rv_marker = biv_geo.markers["ENDO_RV"][0]


act_fn = interpolate_activation_to_ep_mesh(
    ep_mesh=biv_geo.ep_mesh,
    endo_marker_ep=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
    ffun_ep=biv_geo.ffun_ep,
    node_ids_mech=node_ids,
    activation_times_mech=activation_times,
    coords_subset_mech=coords_subset,
)

vals = act_fn.vector().get_local()
finite_vals = vals[np.isfinite(vals)]

print("n EP-mesh vertices:", len(vals))
print("n finite (endocardial) vertices:", len(finite_vals))
print("finite value range:", finite_vals.min(), finite_vals.max())
print("finite value mean:", finite_vals.mean())

"""
Visualize the EP-mesh endocardial vertices used in interpolate_activation_to_ep_mesh,
colored by their assigned activation time, to check whether the extra
~3100 vertices beyond Alya's file are just edge/boundary nodes (expected)
or something more concerning (e.g. scattered valve-plug interior nodes).
"""

import numpy as np
import pyvista as pv
import dolfin


def plot_endocardial_activation(
    ep_mesh: dolfin.Mesh,
    endo_marker_ep: list,
    ffun_ep: dolfin.MeshFunction,
    act_fn: dolfin.Function,
    output_path: str = "endocardial_activation.png",
):
    """
    Plots ONLY the endocardial surface (the facets used to build
    endo_vertex_ids in interpolate_activation_to_ep_mesh), colored by
    activation time -- so you see exactly the surface + values the
    interpolation step actually operates on, not the whole mesh.
    """
    coords = ep_mesh.coordinates()
    V = act_fn.function_space()
    v2d = dolfin.vertex_to_dof_map(V)

    ep_mesh.init(2, 0)

    # Collect the endocardial facets (same filter as interpolate_activation_to_ep_mesh)
    endo_facets = []
    for facet in dolfin.facets(ep_mesh):
        if ffun_ep[facet.index()] in endo_marker_ep:
            endo_facets.append(tuple(facet.entities(0)))

    endo_facets = np.array(endo_facets)
    n_faces = endo_facets.shape[0]
    faces = np.hstack(
        [np.full((n_faces, 1), 3), endo_facets]
    ).astype(np.int64).flatten()

    mesh = pv.PolyData(coords, faces)

    # Per-vertex activation time, pulled via vertex_to_dof_map
    n_vertices = coords.shape[0]
    act_values = np.full(n_vertices, np.nan)
    for vid in range(n_vertices):
        dof = v2d[vid]
        val = act_fn.vector()[dof]
        if np.isfinite(val):
            act_values[vid] = val

    mesh.point_data["activation_time"] = act_values

    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.add_mesh(
        mesh,
        scalars="activation_time",
        cmap="viridis",
        nan_color="red",  # highlight any unexpected NaN gaps in red
        scalar_bar_args={"title": "activation time (ms)"},
    )
    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.screenshot(output_path)
    print(f"Saved {output_path}")

    # Side view too, useful for checking front/back of the endocardial surfaces
    plotter2 = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter2.add_mesh(
        mesh,
        scalars="activation_time",
        cmap="viridis",
        nan_color="red",
        scalar_bar_args={"title": "activation time (ms)"},
    )
    plotter2.camera_position = "xz"
    plotter2.screenshot(output_path.replace(".png", "_side.png"))
    print(f"Saved {output_path.replace('.png', '_side.png')}")
# Sanity check: does node_coords (used for the activation file) actually
# match the EP mesh's own vertex coordinates at the same index?

test_node_id = node_ids[0]  # e.g. 35
print("node_coords[test_node_id] (from activation-times loader):", node_coords[test_node_id])
print("ep_mesh vertex at same index:", biv_geo.ep_mesh.coordinates()[test_node_id])

# Repeat for a few more, including one from later in the array,
# to rule out a partial/offset mismatch rather than a uniform one
for tid in [node_ids[0], node_ids[100], node_ids[-1]]:
    print(f"node {tid}: file-coords={node_coords[tid]}  mesh-coords={biv_geo.ep_mesh.coordinates()[tid]}")

# import pyvista as pv
# import numpy as np
#
# def plot_raw_activation_points(coords_subset, activation_times, output_path="raw_activation_points.png"):
#     cloud = pv.PolyData(coords_subset)
#     cloud.point_data["activation_time"] = activation_times
#
#     plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
#     plotter.add_mesh(
#         cloud,
#         scalars="activation_time",
#         cmap="viridis",
#         point_size=8,
#         render_points_as_spheres=True,
#         scalar_bar_args={"title": "activation time (ms)"},
#     )
#     plotter.add_axes()
#     plotter.camera_position = "iso"
#     plotter.screenshot(output_path)
#     print(f"Saved {output_path}")
#
# plot_raw_activation_points(coords_subset, activation_times)
#
# node_ids_shifted = node_ids - 1
# coords_subset_shifted = node_coords[node_ids_shifted]
#
# plot_raw_activation_points(coords_subset_shifted, activation_times, output_path="raw_activation_points_shifted.png")

# # --- Example usage ----------------------------------------------------
# #
plot_endocardial_activation(
    ep_mesh=biv_geo.ep_mesh,
    endo_marker_ep=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
    ffun_ep=biv_geo.ffun_ep,
    act_fn=act_fn,
    output_path="endocardial_activation.png",
)