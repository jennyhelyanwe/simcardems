import os
cache_dir = os.environ.get("FENICS_CACHE_DIR", os.path.expanduser("~/.cache"))
os.environ["XDG_CACHE_HOME"] = cache_dir

import dolfin
import pulse
import numpy as np
from cardiac_geometries.geometry import Geometry
from simcardems.bivgeometry import BiVentricularGeometry
from simcardems import boundary_conditions

def mpi_print(*args, **kwargs):
    if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
        print(*args, **kwargs, flush=True)

# ── Load geometry ──────────────────────────────────────────────────────────────
mpi_print("Loading geometry...")
geo = Geometry.from_file("rodero_05_coarse.h5")
biv_geo = BiVentricularGeometry.from_geometry(
    geo, ep_mesh=geo.mesh, ffun_ep=geo.ffun,
)
mpi_print(f"Mesh elements: {biv_geo.mechanics_mesh.num_cells()}")
mpi_print(f"Mesh vertices: {biv_geo.mechanics_mesh.num_vertices()}")

# ── Material ───────────────────────────────────────────────────────────────────
mpi_print("Setting up material...")
material_parameters = pulse.Guccione.default_parameters()
material_parameters["CC"] = 2.0
material_parameters["bf"] = 8.0
material_parameters["bfs"] = 8.0
material_parameters["bt"] = 8.0

material = pulse.Guccione(
    parameters=material_parameters,
    f0=biv_geo.microstructure.f0,
    s0=biv_geo.microstructure.s0,
    n0=biv_geo.microstructure.n0,
)

# ── Boundary conditions ────────────────────────────────────────────────────────
mpi_print("Setting up boundary conditions...")
bcs = boundary_conditions.create_biv_boundary_conditions(
    geo=biv_geo,
    traction_lv=0.0,
    traction_rv=0.0,
    spring=10.0,
)

lv_pressure = bcs.neumann[0].traction
rv_pressure = bcs.neumann[1].traction

# ── Problem ────────────────────────────────────────────────────────────────────
mpi_print("Setting up problem...")

class BiVMechanicsProblem(pulse.MechanicsProblem):
    def _set_dirichlet_bc(self):
        self._dirichlet_bc = []

    def _init_solver(self):
        self._problem = pulse.solver.NonlinearProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=self._dirichlet_bc,
        )
        dolfin.PETScOptions.set("ksp_type", "gmres")
        dolfin.PETScOptions.set("pc_type", "jacobi")
        dolfin.PETScOptions.set("ksp_max_it", "500")
        dolfin.PETScOptions.set("ksp_gmres_restart", "30")
        dolfin.PETScOptions.set("ksp_rtol", "1e-4")

        self.solver = dolfin.NewtonSolver()
        self.solver.parameters["linear_solver"] = "petsc"
        self.solver.parameters["preconditioner"] = "none"
        self.solver.parameters["error_on_nonconvergence"] = False
        self.solver.parameters["maximum_iterations"] = 20
        self.solver.parameters["relaxation_parameter"] = 0.1

    def solve(self):
        self.solver.solve(self._problem, self.state.vector())
        return 1, True

problem = BiVMechanicsProblem(biv_geo, material, bcs)

# ── Initial solve at zero load ─────────────────────────────────────────────────
mpi_print("Initial solve (zero load)...")
problem.solve()
mpi_print("Initial solve done.")

# ── Apply pressure incrementally ───────────────────────────────────────────────
mpi_print("Applying pressure incrementally...")
n_steps = 10
for i in range(n_steps):
    p_lv = 0.5 * (i + 1) / n_steps
    p_rv = 0.17 * (i + 1) / n_steps
    bcs.neumann[0].traction.assign(p_lv)
    bcs.neumann[1].traction.assign(p_rv)
    mpi_print(f"Step {i+1}/{n_steps}: LVP={p_lv:.3f} RVP={p_rv:.3f}")
    nliter, nlconv = problem.solve()
    mpi_print(f"  converged={nlconv} iterations={nliter}")
    u, p = problem.state.split(deepcopy=True)
    mpi_print(f"  Max displacement: {u.vector().norm('linf'):.6e} mm")

mpi_print("Pressure solve done.")
u, p = problem.state.split(deepcopy=True)
mpi_print(f"State vector norm: {problem.state.vector().norm('l2'):.6e}")
mpi_print(f"Max displacement: {u.vector().norm('linf'):.6e} mm")