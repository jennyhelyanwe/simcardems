# facet_marker_check.py
import dolfin
import numpy as np


def check(mesh, ffun, markers=None):
    comm = mesh.mpi_comm()
    rank = dolfin.MPI.rank(comm)
    nprocs = dolfin.MPI.size(comm)

    if markers is not None:
        ids = sorted({v[0] for v in markers.values()})
        names = {v[0]: k for k, v in markers.items()}
    else:
        ids = [10, 20, 30, 40]
        names = {10: "BASE", 20: "ENDO_RV", 30: "ENDO_LV", 40: "EPI"}

    ds = dolfin.Measure("ds", domain=mesh, subdomain_data=ffun)
    one = dolfin.Constant(1.0)

    arr = ffun.array()
    local_counts = {m: int(np.count_nonzero(arr == m)) for m in ids}
    global_counts = {m: int(dolfin.MPI.sum(comm, float(local_counts[m]))) for m in ids}

    # assembled area is the invariant — this is the number that matters
    areas = {m: dolfin.assemble(one * ds(m)) for m in ids}

    if rank == 0:
        print(f"[FACET] nprocs={nprocs}")
        print(f"[FACET] {'name':9s} {'id':>3s} {'count(rough)':>14s} {'area(invariant)':>18s}")
        for m in ids:
            print(f"[FACET] {names.get(m,'?'):9s} {m:>3d} "
                  f"{global_counts[m]:>14d} {areas[m]:>18.6e}")
        print("[FACET] compare 'area' across nprocs — any drift => parallel marking is broken")

import dolfin
import numpy as np
MESH_DIR = 'meshes/'
RESOLUTION = '4mm'
import dolfin
import numpy as np

mesh = dolfin.Mesh()
with dolfin.HDF5File(mesh.mpi_comm(), MESH_DIR + "rodero_05_coarse_" + RESOLUTION + ".h5", "r") as f:
    f.read(mesh, "mesh", False)
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    f.read(ffun, "meshfunctions/ffun")

ffun_arr = ffun.array()

mesh.init(2, 0)
mesh.init(2, 3)
n_facets = mesh.num_entities(2)
conn_20 = mesh.topology()(2, 0)
conn_23 = mesh.topology()(2, 3)

facet_verts_arr = np.array([conn_20(i) for i in range(n_facets)])
exterior_mask = np.array([len(conn_23(i)) == 1 for i in range(n_facets)])

coords = mesh.coordinates()
facet_centroids = coords[facet_verts_arr].mean(axis=1)

print(f"Total facets: {n_facets}")
print(f"Total exterior facets: {exterior_mask.sum()}")

unassigned_mask = (ffun_arr == 0) & exterior_mask
print(f"Unassigned (marker=0) exterior facets: {unassigned_mask.sum()}")

if unassigned_mask.sum() > 0:
    unassigned_centroids = facet_centroids[unassigned_mask]
    print(f"Centroid range:")
    print(f"  x: {unassigned_centroids[:,0].min():.2f} to {unassigned_centroids[:,0].max():.2f}")
    print(f"  y: {unassigned_centroids[:,1].min():.2f} to {unassigned_centroids[:,1].max():.2f}")
    print(f"  z: {unassigned_centroids[:,2].min():.2f} to {unassigned_centroids[:,2].max():.2f}")