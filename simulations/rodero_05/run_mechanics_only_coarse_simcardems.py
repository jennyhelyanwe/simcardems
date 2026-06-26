import os
cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import dolfin
dolfin.PETScOptions.set("ksp_type", "gmres")
dolfin.PETScOptions.set("pc_type", "jacobi")

import pulse
import numpy as np
from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry
from simcardems import mechanics_model, boundary_conditions

def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs, flush=True)

mpi_print("Loading geometry...")
geo = Geometry.from_file("rodero_05_coarse.h5")
biv_geo = BiVentricularGeometry.from_geometry(
    geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun,
)
mpi_print(f"Mesh elements: {biv_geo.mechanics_mesh.num_cells()}")

# Passive Guccione — no active model
material_parameters = pulse.Guccione.default_parameters()
material_parameters["CC"] = 2.0
material_parameters["bf"] = 8.0
material_parameters["bfs"] = 4.0
material_parameters["bt"] = 2.0

material = pulse.Guccione(
    parameters=material_parameters,
    f0=biv_geo.microstructure.f0,
    s0=biv_geo.microstructure.s0,
    n0=biv_geo.microstructure.n0,
)

# BCs — zero traction, spring on EPI
bcs = mechanics_model.resolve_boundary_conditions(
    geo=biv_geo,
    traction=0.0,
    spring=10.0,
)

# Use simcardems MechanicsProblem directly
problem = mechanics_model.create_problem(
    material=material,
    geo=biv_geo,
    traction=0.0,
    spring=10.0,
    linear_solver="mumps",
    use_custom_newton_solver=False,
)

mpi_print("Initial solve...")
problem.solve()
mpi_print("Initial solve done.")

# Fish out traction constants
lv_pressure = None
rv_pressure = None
for nbc in problem.bcs.neumann:
    if nbc.marker == biv_geo.markers["ENDO_LV"][0]:
        lv_pressure = nbc.traction
    elif nbc.marker == biv_geo.markers["ENDO_RV"][0]:
        rv_pressure = nbc.traction

mpi_print(f"LV pressure wired: {lv_pressure is not None}")
mpi_print(f"RV pressure wired: {rv_pressure is not None}")

n_steps = 10
for i in range(n_steps):
    p_lv = 0.5 * (i + 1) / n_steps
    p_rv = 0.17 * (i + 1) / n_steps
    if lv_pressure is not None:
        lv_pressure.assign(p_lv)
    if rv_pressure is not None:
        rv_pressure.assign(p_rv)
    mpi_print(f"Step {i+1}/{n_steps}: LVP={p_lv:.3f} RVP={p_rv:.3f}")
    try:
        nliter, nlconv = problem.solve()
        mpi_print(f"  converged={nlconv} iterations={nliter}")
        u, _ = problem.state.split(deepcopy=True)
        mpi_print(f"  Max displacement: {u.vector().norm('linf'):.6e} mm")
    except Exception as e:
        mpi_print(f"  Failed: {e}")
        break