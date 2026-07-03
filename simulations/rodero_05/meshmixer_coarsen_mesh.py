"""
Coarse mesh generation pipeline for rodero_05.
Part 1 (not included): Meshmixer remeshing of rodero_05 surface to 4mm -> rodero_05_4_mm.obj

Steps 2-8:
    2. TetGen tetrahedralise with cavity holes, scale to mm
    3. Extract surface, nearest-neighbour marker transfer from fine mesh boundary facets
    4. Build dolfin mesh from VTU, assign ffun from markers
    5. Save to rodero_05_coarse_4mm.h5 with mesh, ffun, marker metadata
    6. KD-tree fibre interpolation from fine mesh
    7. Save fibres to h5
    8. Export XDMF for visualisation
"""

import numpy as np
import pyvista as pv
import tetgen
import dolfin
import h5py
import meshio
from scipy.spatial import cKDTree

# ── Step 2: TetGen tetrahedralisation ────────────────────────────────────────

print("Step 2: Tetrahedralising with TetGen...")

surf = pv.read("rodero_05_4_mm.obj")
bodies = surf.split_bodies()
bodies_sorted = sorted(bodies, key=lambda b: b.area, reverse=True)
epi, endo1, endo2 = bodies_sorted

def nudge_toward(point, target, factor=0.1):
    return np.array(point) + factor * (np.array(target) - np.array(point))

hole1 = nudge_toward(endo1.center, epi.center)
hole2 = nudge_toward(endo2.center, epi.center)
print(f"  Hole points: {hole1}, {hole2}")

tet = tetgen.TetGen(surf)
tet.add_hole(hole1)
tet.add_hole(hole2)
tet.tetrahedralize(switches="pq1.414/20")

coarse_pv = tet.grid
coarse_pv.points *= 10  # scale to mm
coarse_pv.save("rodero_05_coarse.vtu")
print(f"  TetGen mesh: {coarse_pv.n_points} nodes, {coarse_pv.n_cells} cells")

# ── Step 3: Nearest-neighbour marker transfer from fine mesh ──────────────────

print("Step 3: Transferring boundary markers from fine mesh...")

fine_mesh = dolfin.Mesh()
with dolfin.HDF5File(fine_mesh.mpi_comm(), "rodero_05_fine.h5", "r") as f:
    f.read(fine_mesh, "mesh", False)
    ffun_fine = dolfin.MeshFunction("size_t", fine_mesh, 2)
    f.read(ffun_fine, "meshfunctions/ffun")

fine_mesh.init()
fine_mesh.init(2, 3)

fine_boundary_centroids = []
fine_boundary_markers = []
for facet in dolfin.facets(fine_mesh):
    if facet.exterior():
        fine_boundary_centroids.append(facet.midpoint().array())
        fine_boundary_markers.append(ffun_fine[facet.index()])

fine_boundary_centroids = np.array(fine_boundary_centroids)
fine_boundary_markers = np.array(fine_boundary_markers)
print(f"  Fine marker counts: { {m: np.sum(fine_boundary_markers==m) for m in np.unique(fine_boundary_markers)} }")

tree_fine = cKDTree(fine_boundary_centroids)

coarse_surf = coarse_pv.extract_surface(algorithm='dataset_surface')
coarse_face_centres = coarse_surf.cell_centers().points
_, idx = tree_fine.query(coarse_face_centres)
coarse_markers = fine_boundary_markers[idx]
print(f"  Coarse marker counts: { {m: np.sum(coarse_markers==m) for m in np.unique(coarse_markers)} }")

# Build KD tree from coarse face centres for dolfin facet assignment
tree_coarse = cKDTree(coarse_face_centres)

# ── Step 4: Build dolfin mesh and assign ffun ─────────────────────────────────

print("Step 4: Building dolfin mesh and assigning boundary markers...")

m = meshio.read("rodero_05_coarse.vtu")
coords = m.points
cells = m.cells_dict["tetra"]

dolfin_mesh = dolfin.Mesh()
editor = dolfin.MeshEditor()
editor.open(dolfin_mesh, "tetrahedron", 3, 3)
editor.init_vertices(len(coords))
editor.init_cells(len(cells))
for i, pt in enumerate(coords):
    editor.add_vertex(i, pt)
for i, cell in enumerate(cells):
    editor.add_cell(i, cell)
editor.close()
dolfin_mesh.init()
dolfin_mesh.init(2, 3)

ffun = dolfin.MeshFunction("size_t", dolfin_mesh, 2, 0)
for facet in dolfin.facets(dolfin_mesh):
    if facet.exterior():
        mp = facet.midpoint().array()
        _, i = tree_coarse.query(mp)
        ffun[facet.index()] = coarse_markers[i]

print(f"  Dolfin marker counts: { {m: np.sum(ffun.array()==m) for m in np.unique(ffun.array())} }")
print(f"  Mesh: {dolfin_mesh.num_vertices()} vertices, {dolfin_mesh.num_cells()} cells")

# ── Step 5: Save mesh and ffun to h5 ─────────────────────────────────────────

print("Step 5: Saving mesh to rodero_05_coarse_4mm.h5...")

with dolfin.HDF5File(dolfin_mesh.mpi_comm(), "rodero_05_coarse_4mm.h5", "w") as f:
    f.write(dolfin_mesh, "mesh")
    f.write(ffun, "meshfunctions/ffun")

with h5py.File("rodero_05_coarse_4mm.h5", "a") as f:
    f.create_dataset("info/mesh_type", data="biv_ellipsoid")
    f.create_dataset("info/num_refinements", data=0)
    f.create_dataset("markers/BASE",    data=[10, 2])
    f.create_dataset("markers/ENDO_RV", data=[20, 2])
    f.create_dataset("markers/ENDO_LV", data=[30, 2])
    f.create_dataset("markers/EPI",     data=[40, 2])

print("  Saved mesh and ffun.")

# ── Step 6 & 7: Fibre interpolation from fine mesh ────────────────────────────

print("Step 6: Interpolating fibres from fine mesh...")

VFS_fine = dolfin.VectorFunctionSpace(fine_mesh, "DG", 0)
f0_fine = dolfin.Function(VFS_fine)
s0_fine = dolfin.Function(VFS_fine)
n0_fine = dolfin.Function(VFS_fine)

with dolfin.HDF5File(fine_mesh.mpi_comm(), "rodero_05_fine.h5", "r") as f:
    f.read(f0_fine, "microstructure/f0")
    f.read(s0_fine, "microstructure/s0")
    f.read(n0_fine, "microstructure/n0")

fine_cell_centres = np.array([cell.midpoint().array() for cell in dolfin.cells(fine_mesh)])
tree_fine_cells = cKDTree(fine_cell_centres)

coarse_cell_centres = np.array([cell.midpoint().array() for cell in dolfin.cells(dolfin_mesh)])
_, idx = tree_fine_cells.query(coarse_cell_centres)

f0_vals = f0_fine.vector().get_local().reshape(-1, 3)
s0_vals = s0_fine.vector().get_local().reshape(-1, 3)
n0_vals = n0_fine.vector().get_local().reshape(-1, 3)

VFS_coarse = dolfin.VectorFunctionSpace(dolfin_mesh, "DG", 0)
f0_coarse = dolfin.Function(VFS_coarse)
s0_coarse = dolfin.Function(VFS_coarse)
n0_coarse = dolfin.Function(VFS_coarse)

f0_coarse.vector().set_local(f0_vals[idx].flatten())
s0_coarse.vector().set_local(s0_vals[idx].flatten())
n0_coarse.vector().set_local(n0_vals[idx].flatten())

# Normalise
for func in [f0_coarse, s0_coarse, n0_coarse]:
    arr = func.vector().get_local().reshape(-1, 3)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = np.where(norms > 1e-10, arr / norms, arr)
    func.vector().set_local(arr.flatten())

print("Step 7: Saving fibres to h5...")

with dolfin.HDF5File(dolfin_mesh.mpi_comm(), "rodero_05_coarse_4mm.h5", "a") as f:
    f.write(f0_coarse, "microstructure/f0")
    f.write(s0_coarse, "microstructure/s0")
    f.write(n0_coarse, "microstructure/n0")

print("  Fibres saved.")

print(f"Final mesh: {dolfin_mesh.num_cells()} cells, {dolfin_mesh.num_vertices()} vertices")

# ── Step 8: Export XDMF for visualisation ────────────────────────────────────

print("Step 8: Exporting XDMF...")

with dolfin.XDMFFile("rodero_05_coarse_mesh_4mm.xdmf") as xf:
    xf.write(dolfin_mesh)

with dolfin.XDMFFile("rodero_05_coarse_boundaries_4mm.xdmf") as xf:
    xf.write(ffun)

print("Done. Output: rodero_05_coarse_4mm.h5, rodero_05_coarse_mesh_4mm.xdmf, rodero_05_coarse_boundaries_4mm.xdmf")