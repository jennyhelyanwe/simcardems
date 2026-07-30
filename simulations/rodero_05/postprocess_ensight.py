"""
postprocess_ensight.py
Convert simcardems results.h5 to EnSight Gold ASCII, parallelised over timesteps.
Runs inside Docker (needs dolfin for P1 mesh and displacement extraction).

Usage (parallel):
    mpirun -n 8 python3 postprocess_ensight.py --results biv_coarse_run_output/results.h5 --out ensight/ --tv rodero_05_coarse_tv.npy

Usage (serial):
    python3 postprocess_ensight.py --results ... --out ... --tv ...
"""

import argparse
import os
import numpy as np
import h5py
import dolfin
from scipy.spatial import cKDTree

try:
    from mpi4py import MPI as _MPI4PY
    _comm = _MPI4PY.COMM_WORLD
    RANK = _comm.Get_rank()
    SIZE = _comm.Get_size()
except ImportError:
    _comm = None
    RANK = 0
    SIZE = 1


def rprint(*args, **kwargs):
    if RANK == 0:
        print(*args, **kwargs, flush=True)


def get_timesteps(h5file, group):
    with h5py.File(h5file, 'r') as f:
        field_names = list(f[group].keys())
        if not field_names:
            return []
        return sorted(f[group][field_names[0]].keys(), key=float)


def load_mesh_h5py(h5file, mesh_group):
    with h5py.File(h5file, 'r') as f:
        coords = f[f'geometry/mesh/{mesh_group}/coordinates'][:]
        topo = f[f'geometry/mesh/{mesh_group}/topology'][:]
    return coords, topo


def cell_centres(coords, topo):
    return coords[topo].mean(axis=1)


def load_field_h5py(h5file, group, field_name, time_str):
    """Scalar per-cell field (e.g. lambda). No P2/vertex reordering
    ambiguity here — plain per-cell mean over cell dofs — so raw h5py
    reading is safe for this one."""
    path = f'{group}/{field_name}/{time_str}'
    with h5py.File(h5file, 'r') as f:
        grp = f[path]
        vector = grp['vector_0'][:]
        cell_dofs = grp['cell_dofs'][:]
        x_cell_dofs = grp['x_cell_dofs'][:]
        n_cells = len(grp['cells'])

    vals = np.zeros(n_cells)
    for i in range(n_cells):
        start = x_cell_dofs[i]
        end = x_cell_dofs[i + 1] if i + 1 < len(x_cell_dofs) else len(cell_dofs)
        dofs = cell_dofs[start:end]
        vals[i] = vector[dofs].mean()
    return vals


def extract_u_p1(hdf5_in, u_disp, p1_to_p2, time_str):
    """Extract P2 displacement via dolfin's own HDF5 read (correctly
    handles P2 vertex + edge-midpoint DOF ordering) and downsample to
    P1 corner nodes. hdf5_in and u_disp are created once outside the
    per-timestep loop and reused."""
    hdf5_in.read(u_disp, f'mechanics/u/{time_str}')
    p2_vals = u_disp.vector().get_local().reshape(-1, 3)
    return p2_vals[p1_to_p2]


def build_p1_to_p2_map(mesh):
    V = dolfin.VectorFunctionSpace(mesh, 'Lagrange', 2)
    dm = V.dofmap()
    p1_topo = mesh.cells()
    n_p1_nodes = mesh.num_vertices()
    p1_to_p2 = np.zeros(n_p1_nodes, dtype=np.int64)
    for cell in dolfin.cells(mesh):
        p1_corners = p1_topo[cell.index()]
        p2_dofs = dm.cell_dofs(cell.index())[::3] // 3
        for j in range(4):
            p1_to_p2[p1_corners[j]] = int(p2_dofs[j])
    return p1_to_p2


def write_geo_file(path, coords, topo):
    with open(path, 'w') as f:
        f.write('EnSight Gold geometry file\n')
        f.write('written by postprocess_ensight.py\n')
        f.write('node id assign\n')
        f.write('element id assign\n')
        f.write('part\n')
        f.write('         1\n')
        f.write('biv_mesh\n')
        f.write('coordinates\n')
        f.write(f'{len(coords):10d}\n')
        for i in range(3):
            for pt in coords:
                f.write(f'{pt[i]:12.5e}\n')
        f.write('tetra4\n')
        f.write(f'{len(topo):10d}\n')
        for tet in topo:
            f.write(''.join(f'{n+1:10d}' for n in tet) + '\n')


def write_geo_file_ep(path, coords, topo):
    """Same as write_geo_file but kept separate/named for clarity in this dual-mesh setup."""
    write_geo_file(path, coords, topo)


def write_scalar_file(path, vals, desc):
    with open(path, 'w') as f:
        f.write(f'{desc}\n')
        f.write('part\n')
        f.write('         1\n')
        f.write('tetra4\n')
        for v in vals:
            f.write(f'{float(v):12.5e}\n')


def write_vector_node_file(path, vals, desc):
    with open(path, 'w') as f:
        f.write(f'{desc}\n')
        f.write('part\n')
        f.write('         1\n')
        f.write('coordinates\n')
        for dim in range(3):
            for v in vals:
                f.write(f'{float(v[dim]):12.5e}\n')


def write_case_file(path, geo_file, scalar_vars, vector_vars, times_float):
    with open(path, 'w') as f:
        f.write('FORMAT\ntype: ensight gold\n\n')
        f.write('GEOMETRY\n')
        f.write(f'model: 1 {geo_file}\n\n')
        f.write('VARIABLE\n')
        for name, pattern in scalar_vars:
            f.write(f'scalar per element: 1 {name} {pattern}\n')
        for name, pattern in vector_vars:
            f.write(f'vector per node: 1 {name} {pattern}\n')
        f.write('\nTIME\n')
        f.write('time set: 1\n')
        f.write(f'number of steps: {len(times_float)}\n')
        f.write('filename start number: 1\n')
        f.write('filename increment: 1\n')
        f.write('time values:\n')
        for t in times_float:
            f.write(f'{t:12.5e}\n')


def my_share(indices, rank, size):
    """Round-robin split of a list of indices across ranks - keeps load
    balanced even if per-timestep cost varies slightly."""
    return [idx for idx in indices if idx % size == rank]


def convert(h5file, outdir, tv_file=None):
    if RANK == 0:
        os.makedirs(outdir, exist_ok=True)
    if _comm is not None:
        _comm.Barrier()  # make sure outdir exists before any rank writes into it

    rprint('Loading mechanics mesh via dolfin (MPI.comm_self on every rank)...')
    # comm_self: each rank gets its own full serial copy of the mesh, not a
    # distributed partition - required since h5py reads below are inherently
    # serial/global, not domain-decomposed.
    mesh = dolfin.Mesh(dolfin.MPI.comm_self)
    with dolfin.HDF5File(mesh.mpi_comm(), h5file, 'r') as f:
        f.read(mesh, 'geometry/mesh/mechanics', False)
    p1_coords = mesh.coordinates()
    p1_topo = mesh.cells()
    if RANK == 0:
        print(f'  P1 (mechanics): {len(p1_coords)} nodes, {len(p1_topo)} cells')

    p1_to_p2 = build_p1_to_p2_map(mesh)  # same on every rank, computed once

    # Reused across every timestep — avoids rebuilding the P2 space/reader
    # per iteration, and (critically) goes through dolfin's own HDF5 read
    # rather than a raw h5py array slice, so P2 vertex + edge-midpoint DOF
    # ordering is handled correctly.
    V_disp = dolfin.VectorFunctionSpace(mesh, 'Lagrange', 2)
    u_disp = dolfin.Function(V_disp)
    hdf5_in = dolfin.HDF5File(mesh.mpi_comm(), h5file, 'r')

    rprint('Loading EP mesh...')
    ep_coords, ep_topo = load_mesh_h5py(h5file, 'ep')
    if RANK == 0:
        print(f'  EP: {len(ep_coords)} nodes, {len(ep_topo)} cells')
    mech_centres = cell_centres(p1_coords, p1_topo)
    ep_centres = cell_centres(ep_coords, ep_topo)

    rprint('Building KD-tree (EP -> mechanics)...')
    tree = cKDTree(ep_centres)
    _, ep_to_mech_idx = tree.query(mech_centres)

    if RANK == 0:
        print('Writing geometry (mechanics)...')
        geo_file = 'biv.geo'
        write_geo_file(os.path.join(outdir, geo_file), p1_coords, p1_topo)

        print('Writing geometry (EP, native resolution)...')
        geo_file_ep = 'biv_ep.geo'
        write_geo_file_ep(os.path.join(outdir, geo_file_ep), ep_coords, ep_topo)
    else:
        geo_file = 'biv.geo'
        geo_file_ep = 'biv_ep.geo'

    mech_time_strs = get_timesteps(h5file, 'mechanics')
    ep_time_strs = get_timesteps(h5file, 'ep')
    all_time_strs = sorted(set(mech_time_strs) | set(ep_time_strs), key=float)
    times_float = [float(t) for t in all_time_strs]
    rprint(f'  {len(times_float)} timesteps, distributing across {SIZE} rank(s)')

    with h5py.File(h5file, 'r') as f:
        mech_fields = [fn for fn in f['mechanics'].keys() if fn != 'u']
        ep_fields = list(f['ep'].keys())

    scalar_vars = []          # mechanics-mesh case (includes EP interpolated down)
    vector_vars = []
    scalar_vars_ep = []       # EP-mesh case (EP fields at native resolution)

    my_timestep_indices = my_share(range(len(all_time_strs)), RANK, SIZE)

    # ── mech_u: vector per node, P2->P1 on mechanics mesh, via dolfin's
    #    own HDF5 read (fixed — previously used a raw h5py extraction that
    #    scrambled values at some nodes; confirmed via make_xdmffiles that
    #    the underlying simulation data was correct all along) ──────────
    var_name = 'mech_u'
    pattern = f'{var_name}.****'
    rprint(f'Processing {var_name}...')
    for i in my_timestep_indices:
        t_str = all_time_strs[i]
        out_path = os.path.join(outdir, f'{var_name}.{i+1:04d}')
        if t_str in mech_time_strs:
            u_vals = extract_u_p1(hdf5_in, u_disp, p1_to_p2, t_str)
        else:
            u_vals = np.zeros((len(p1_coords), 3))
        write_vector_node_file(out_path, u_vals, var_name)
    vector_vars.append((var_name, pattern))

    # ── Scalar mechanics fields (mechanics mesh only) ─────────────────────────
    for field_name in mech_fields:
        var_name = f'mech_{field_name}'
        pattern = f'{var_name}.****'
        rprint(f'Processing {var_name}...')
        for i in my_timestep_indices:
            t_str = all_time_strs[i]
            out_path = os.path.join(outdir, f'{var_name}.{i+1:04d}')
            if t_str in mech_time_strs:
                vals = load_field_h5py(h5file, 'mechanics', field_name, t_str)
            else:
                vals = np.zeros(len(p1_topo))
            write_scalar_file(out_path, vals, var_name)
        scalar_vars.append((var_name, pattern))

    # ── EP fields: write BOTH native (EP mesh) and interpolated (mechanics mesh) ──
    for field_name in ep_fields:
        var_name_ep = f'ep_{field_name}'
        pattern_ep = f'{var_name_ep}.****'
        rprint(f'Processing {var_name_ep} (native EP resolution)...')
        for i in my_timestep_indices:
            t_str = all_time_strs[i]
            out_path = os.path.join(outdir, f'{var_name_ep}.{i+1:04d}')
            if t_str in ep_time_strs:
                vals = load_field_h5py(h5file, 'ep', field_name, t_str)
            else:
                vals = np.zeros(len(ep_topo))
            write_scalar_file(out_path, vals, var_name_ep)
        scalar_vars_ep.append((var_name_ep, pattern_ep))

        var_name_mech = f'ep_on_mech_{field_name}'
        pattern_mech = f'{var_name_mech}.****'
        rprint(f'Processing {var_name_mech} (EP -> mechanics, downsampled)...')
        for i in my_timestep_indices:
            t_str = all_time_strs[i]
            out_path = os.path.join(outdir, f'{var_name_mech}.{i+1:04d}')
            if t_str in ep_time_strs:
                ep_vals = load_field_h5py(h5file, 'ep', field_name, t_str)
                vals = ep_vals[ep_to_mech_idx]
            else:
                vals = np.zeros(len(p1_topo))
            write_scalar_file(out_path, vals, var_name_mech)
        scalar_vars.append((var_name_mech, pattern_mech))

    # ── Material labels (mechanics mesh only) ─────────────────────────────────
    if tv_file and os.path.exists(tv_file):
        coarse_tv = np.load(tv_file)
        var_name = 'material_tv'
        pattern = f'{var_name}.****'
        rprint(f'Processing {var_name}...')
        for i in my_timestep_indices:
            out_path = os.path.join(outdir, f'{var_name}.{i+1:04d}')
            write_scalar_file(out_path, coarse_tv.astype(float), var_name)
        scalar_vars.append((var_name, pattern))

    hdf5_in.close()

    if _comm is not None:
        _comm.Barrier()  # make sure every rank's files exist before rank 0 writes the case file

    if RANK == 0:
        case_path = os.path.join(outdir, 'biv.case')
        write_case_file(case_path, geo_file, scalar_vars, vector_vars, times_float)
        print(f'Mechanics-mesh case written: {case_path}')

        case_path_ep = os.path.join(outdir, 'biv_ep.case')
        write_case_file(case_path_ep, geo_file_ep, scalar_vars_ep, [], times_float)
        print(f'EP-mesh (native resolution) case written: {case_path_ep}')
        print(f'Open either in ParaView: {case_path} (mechanics + downsampled EP), '
              f'{case_path_ep} (EP fields at full resolution)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results_4mm/biv_coarse_run_output/results.h5')
    parser.add_argument('--out', default='results_4mm/ensight')
    parser.add_argument('--tv', default='rodero_05_coarse_tv.npy', help='Path to coarse_tv.npy material labels')
    args = parser.parse_args()
    convert(args.results, args.out, args.tv)
