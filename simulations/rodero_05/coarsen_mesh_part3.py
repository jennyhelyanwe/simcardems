import dolfin
import numpy as np
from scipy.spatial import cKDTree

# Load fine mesh with fibres
fine_mesh = dolfin.Mesh()
with dolfin.HDF5File(fine_mesh.mpi_comm(), "rodero_05_fine.h5", "r") as f:
    f.read(fine_mesh, "mesh", False)
    f0_fine = dolfin.Function(dolfin.VectorFunctionSpace(fine_mesh, "DG", 0))
    s0_fine = dolfin.Function(dolfin.VectorFunctionSpace(fine_mesh, "DG", 0))
    n0_fine = dolfin.Function(dolfin.VectorFunctionSpace(fine_mesh, "DG", 0))
    f.read(f0_fine, "microstructure/f0")
    f.read(s0_fine, "microstructure/s0")
    f.read(n0_fine, "microstructure/n0")

# Load coarse mesh
coarse_mesh = dolfin.Mesh()
with dolfin.HDF5File(coarse_mesh.mpi_comm(), "rodero_05_coarse.h5", "r") as f:
    f.read(coarse_mesh, "mesh", False)
    ffun_coarse = dolfin.MeshFunction("size_t", coarse_mesh, 2)
    f.read(ffun_coarse, "boundaries")

# Get fine mesh cell centres
fine_cell_centres = np.array([cell.midpoint().array() for cell in dolfin.cells(fine_mesh)])

# Build KD tree
tree = cKDTree(fine_cell_centres)

# Get coarse mesh cell centres
coarse_cell_centres = np.array([cell.midpoint().array() for cell in dolfin.cells(coarse_mesh)])

# Find nearest fine cell for each coarse cell
_, idx = tree.query(coarse_cell_centres)

# Extract fibre values at nearest fine cells
f0_vals = f0_fine.vector().get_local().reshape(-1, 3)
s0_vals = s0_fine.vector().get_local().reshape(-1, 3)
n0_vals = n0_fine.vector().get_local().reshape(-1, 3)

# Create DG0 functions on coarse mesh
DG0 = dolfin.VectorFunctionSpace(coarse_mesh, "DG", 0)
f0_coarse = dolfin.Function(DG0)
s0_coarse = dolfin.Function(DG0)
n0_coarse = dolfin.Function(DG0)

f0_coarse.vector().set_local(f0_vals[idx].flatten())
s0_coarse.vector().set_local(s0_vals[idx].flatten())
n0_coarse.vector().set_local(n0_vals[idx].flatten())

# Save everything to coarse h5
with dolfin.HDF5File(coarse_mesh.mpi_comm(), "rodero_05_coarse.h5", "a") as f:
    f.write(f0_coarse, "microstructure/f0")
    f.write(s0_coarse, "microstructure/s0")
    f.write(n0_coarse, "microstructure/n0")

print("Done. Fibres written to rodero_05_coarse.h5")