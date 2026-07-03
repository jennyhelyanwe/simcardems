"""
extract_p2.py
Extract P1 mesh geometry and per-timestep displacement fields from results.h5.
Displacement is P2 but interpolated to P1 corner nodes for visualisation.
Run inside Docker container.

Usage:
    python3 extract_p2.py --results biv_coarse_run_output/results.h5 --out p2_data/
"""

import argparse
import os
import numpy as np
import h5py
import dolfin


def get_timesteps(h5file, group):
    with h5py.File(h5file, 'r') as f:
        field_names = list(f[group].keys())
        if not field_names:
            return []
        return sorted(f[group][field_names[0]].keys(), key=float)


def main(h5file, outdir):
    os.makedirs(outdir, exist_ok=True)

    print('Loading mechanics mesh...')
    mesh = dolfin.Mesh()
    with dolfin.HDF5File(mesh.mpi_comm(), h5file, 'r') as f:
        f.read(mesh, 'geometry/mesh/mechanics', False)

    p1_coords = mesh.coordinates()
    p1_topo = mesh.cells()
    n_p1_nodes = len(p1_coords)
    print(f'  P1: {n_p1_nodes} nodes, {mesh.num_cells()} cells')

    print('Building P2 function space...')
    V = dolfin.VectorFunctionSpace(mesh, 'Lagrange', 2)
    u = dolfin.Function(V)

    # Build mapping from P1 corner nodes to P2 DOF indices
    dm = V.dofmap()
    p1_to_p2 = np.zeros(n_p1_nodes, dtype=int)
    for cell in dolfin.cells(mesh):
        p1_corners = p1_topo[cell.index()]  # 4 corner node indices
        p2_dofs = dm.cell_dofs(cell.index())[::3] // 3  # 10 P2 node indices
        # First 4 P2 nodes correspond to P1 corners
        for j in range(4):
            p1_to_p2[p1_corners[j]] = p2_dofs[j]

    np.save(os.path.join(outdir, 'p1_coords.npy'), p1_coords)
    np.save(os.path.join(outdir, 'p1_topo.npy'), p1_topo)
    print(f'  Saved p1_coords.npy ({n_p1_nodes} nodes) and p1_topo.npy')

    time_strs = get_timesteps(h5file, 'mechanics')
    print(f'  {len(time_strs)} timesteps: {time_strs}')

    for t_str in time_strs:
        with dolfin.HDF5File(mesh.mpi_comm(), h5file, 'r') as f:
            f.read(u, f'mechanics/u/{t_str}')
        p2_vals = u.vector().get_local().reshape(-1, 3)
        # Interpolate to P1 corner nodes
        u_p1 = p2_vals[p1_to_p2]
        fname = os.path.join(outdir, f'u_{t_str}.npy')
        np.save(fname, u_p1)
        print(f'  Saved u_{t_str}.npy, shape={u_p1.shape}, max_mag={np.linalg.norm(u_p1, axis=1).max():.3f} mm')

    np.save(os.path.join(outdir, 'timesteps.npy'), np.array([float(t) for t in time_strs]))
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--out', default='p2_data')
    args = parser.parse_args()
    main(args.results, args.out)