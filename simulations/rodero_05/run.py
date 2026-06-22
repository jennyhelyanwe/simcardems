from simcardems.activation import (
    load_activation_times,
    interpolate_activation_to_ep_mesh,
    endocardial_stimulus_domain,
)
from simcardems.spatial_fields import (
    map_dense_field_to_dg0_function,
    load_dense_node_field,
    map_dense_field_to_ep_mesh
)
from simcardems.config import Config
from simcardems.models import em_model
from simcardems.runner import Runner
import numpy as np


# # 1. Build geometry (same as your validated test.py setup)
# print('Build geometry... ')
# from cardiac_geometries.geometry import Geometry
# geo = Geometry.from_file("rodero_05_fine.h5")
# from simcardems.bivgeometry import BiVentricularGeometry
# biv_geo = BiVentricularGeometry.from_geometry(
#     geo,
#     ep_mesh=geo.mesh,
#     ffun_ep=geo.ffun,
# )
# import dolfin
# print(f"rank {dolfin.MPI.rank(dolfin.MPI.comm_world)}: local mesh vertices = {biv_geo.ep_mesh.num_vertices()}", flush=True)

import dolfin
import pulse
from simcardems.bivgeometry import BiVentricularGeometry
def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs)

# Load mesh via POSIX (no MPI_File_open)
mpi_print('Build geometry... ')
mesh = dolfin.Mesh(dolfin.MPI.comm_world, "rodero_05_mesh.xml")

# Load ffun
ffun = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
dolfin.File("rodero_05_ffun.xml") >> ffun

# Load microstructure
V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)
f0 = dolfin.Function(V)
s0 = dolfin.Function(V)
n0 = dolfin.Function(V)
dolfin.File("rodero_05_f0.xml") >> f0
dolfin.File("rodero_05_s0.xml") >> s0
dolfin.File("rodero_05_n0.xml") >> n0

# Load markers and info
markers = np.load("rodero_05_markers.npy", allow_pickle=True).item()
info = np.load("rodero_05_info.npy", allow_pickle=True).item()

# Build BiVentricularGeometry directly, bypassing Geometry.from_file entirely
biv_geo = BiVentricularGeometry(
    mechanics_mesh=mesh,
    ep_mesh=mesh,
    markers=markers,
    ffun=ffun,
    ffun_ep=ffun,
    microstructure=pulse.Microstructure(f0=f0, s0=s0, n0=n0),
    parameters=info,
)

mpi_print(f"rank {dolfin.MPI.rank(dolfin.MPI.comm_world)}: local mesh vertices = {mesh.num_vertices()}", flush=True)


# 1. Load the raw activation-time file
mpi_print('Load activation time file', flush=True)
import pandas as pd
node_coords = pd.read_csv("./rodero_05_fine/rodero_05_fine_xyz.csv", header=None).to_numpy() * 10.0
# node_coords = biv_geo.mechanics_mesh.coordinates()

# 2. Activation times, with the indexing fix and (parked) tm filter
path = "./heart.endocardial-activation-times"
node_ids, activation_times, coords_subset = load_activation_times(path, node_coords)
activation_times = activation_times * 1000.0  # s -> ms

act_fn = interpolate_activation_to_ep_mesh(
    ep_mesh=biv_geo.ep_mesh,
    endo_marker_ep=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
    ffun_ep=biv_geo.ffun_ep,
    node_ids_mech=node_ids,
    activation_times_mech=activation_times,
    coords_subset_mech=coords_subset,
)

# 3. Stimulus domain -- the cell-region marking, layer_thickness tunable
mpi_print(f"rank {dolfin.MPI.rank(dolfin.MPI.comm_world)}: Create stimulus domain", flush=True)
stim_domain = endocardial_stimulus_domain(
    mesh=biv_geo.ep_mesh,
    ffun=biv_geo.ffun_ep,
    endo_markers=[biv_geo.markers["ENDO_LV"][0], biv_geo.markers["ENDO_RV"][0]],
    layer_thickness=2,
)
biv_geo.stimulus_domain = stim_domain  # or however your loader injects this --
                                          # check whether BiVentricularGeometry.from_geometry
                                          # accepts stimulus_domain as a kwarg the way
                                          # LeftVentricularGeometry/load_geometry do, since
                                          # we never explicitly added that param to BiV's
                                          # constructor -- WORTH CHECKING before assuming
                                          # this attribute assignment is sufficient

# 4. Config -- short test window
mpi_print(f"rank {dolfin.MPI.rank(dolfin.MPI.comm_world)}: Configuring...", flush=True)
config = Config()
config.T = 50.0
config.dt = 0.05
config.geometry_path = "our_biv_geo.h5"
config.outdir = "test_run_output"
config.coupling_type = "fully_coupled_Tor_Land"
config.save_freq = 50
config.linear_mechanics_solver = "gmres"

# Load cell type field
CELL_TYPE_PATH = "./rodero_05_fine/rodero_05_fine_nodefield_cell-type.csv"
mpi_print(f"rank {dolfin.MPI.rank(dolfin.MPI.comm_world)}: Loading cell type...", flush=True)
ct_values = load_dense_node_field(
    path=CELL_TYPE_PATH
)
mpi_print(f"rank {dolfin.MPI.rank(dolfin.MPI.comm_world)}: Mapping cell type to EP mesh...", flush=True)
cell_fn = map_dense_field_to_dg0_function(
    biv_geo.ep_mesh, node_coords, ct_values, {1: 0, 2: 2, 3: 1}
)

# Load IKs spatial field
IKS_SCALE_PATH = "./rodero_05_fine/rodero_05_fine_nodefield_sf_IKs.csv"
mpi_print(f"rank {dolfin.MPI.rank(dolfin.MPI.comm_world)}: Loading sf IKs...", flush=True)
iks_values = load_dense_node_field(
    path=IKS_SCALE_PATH
)
mpi_print(f"rank {dolfin.MPI.rank(dolfin.MPI.comm_world)}: Mapping sf IKs to EP mesh...", flush=True)
iks_fn = map_dense_field_to_ep_mesh(biv_geo.ep_mesh, node_coords, iks_values)

# # Sanity check for whether celltype and sf IKs have been loaded in correctly
# def plot_field_mpi_safe(mesh, fn, title, output_path, cmap="viridis"):
#     if dolfin.MPI.rank(dolfin.MPI.comm_world) != 0:
#         return
#     import pyvista as pv
#     coords = mesh.coordinates()
#     cells = mesh.cells()
#     n_cells = cells.shape[0]
#     cell_array = np.hstack([np.full((n_cells, 1), 4), cells]).astype(np.int64).flatten()
#     cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
#     grid = pv.UnstructuredGrid(cell_array, cell_types, coords)
#
#     V = fn.function_space()
#     element = V.ufl_element()
#
#     if element.degree() == 0:
#         # DG0 -- cell-based field
#         vals = fn.vector().get_local()
#         dm = V.dofmap()
#         cell_vals = np.zeros(mesh.num_cells())
#         for cell in dolfin.cells(mesh):
#             dof = dm.cell_dofs(cell.index())[0]
#             if dof < len(vals):
#                 cell_vals[cell.index()] = vals[dof]
#         grid.cell_data[title] = cell_vals
#         scalars = title
#     else:
#         # P1 -- vertex-based field
#         vals = fn.vector().get_local()
#         v2d = dolfin.vertex_to_dof_map(V)
#         local_size = len(vals)
#         vertex_vals = np.zeros(mesh.num_vertices())
#         mask = (v2d >= 0) & (v2d < local_size)
#         vertex_vals[mask] = vals[v2d[mask]]
#         grid.point_data[title] = vertex_vals
#         scalars = title
#
#     clipped = grid.clip(normal="x", origin=grid.center)
#     plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
#     plotter.add_mesh(clipped, scalars=scalars, cmap=cmap, scalar_bar_args={"title": title})
#     plotter.add_axes()
#     plotter.screenshot(output_path)
#     print(f"Saved {output_path}", flush=True)
#
# dolfin.MPI.comm_world.barrier()
# plot_field_mpi_safe(biv_geo.ep_mesh, cell_fn, "cell_type", "cell_type_mpi.png", cmap="Set1")
# plot_field_mpi_safe(biv_geo.ep_mesh, iks_fn, "iks_scale", "iks_scale_mpi.png", cmap="viridis")
# dolfin.MPI.comm_world.barrier()
# quit()

# 5. Build coupling directly (bypasses Runner.__init__'s internal
#    setup_EM_model_from_config(self._config) call, which has no way
#    to receive our pre-built biv_geo/act_fn)
mpi_print(f"rank {dolfin.MPI.rank(dolfin.MPI.comm_world)}: Setting EM model for config...", flush=True)
coupling = em_model.setup_EM_model_from_config(
    config,
    geometry=biv_geo,
    activation_times=act_fn,
    celltype_function=cell_fn,
    iks_scale_function=iks_fn,
)

# 6. Hand the pre-built coupling to Runner via the alternate constructor
mpi_print('Run', flush=True)
runner = Runner.from_models(coupling=coupling, config=config)
runner.solve(T=config.T, save_freq=config.save_freq, show_progress_bar=True)
