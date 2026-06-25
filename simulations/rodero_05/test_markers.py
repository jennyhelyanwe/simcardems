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