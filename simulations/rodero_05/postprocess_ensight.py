"""
postprocess_ensight.py
Convert simcardems results.h5 + extracted numpy data to EnSight Gold ASCII.
EP fields nearest-neighbour interpolated onto mechanics mesh.
Displacement written as vector per node on P1 mesh.

Usage:
    python3 postprocess_ensight.py --results biv_coarse_run_output/results.h5 --p2 p2_data/ --out ensight/
"""

import argparse
import os
import numpy as np
import h5py
from scipy.spatial import cKDTree


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
        end = x_cell_dofs[i+1] if i+1 < len(x_cell_dofs) else len(cell_dofs)
        dofs = cell_dofs[start:end]
        vals[i] = vector[dofs].mean()
    return vals


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


def write_scalar_file(path, vals, desc):
    with open(path, 'w') as f:
        f.write(f'{desc}\n')
        f.write('part\n')
        f.write('         1\n')
        f.write('tetra4\n')
        for v in vals:
            f.write(f'{float(v):12.5e}\n')


def write_vector_node_file(path, vals, desc):
    """vals shape: (n_nodes, 3) — vector per node."""
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


def convert(h5file, p2dir, outdir):
    os.makedirs(outdir, exist_ok=True)

    print('Loading P1 mesh from numpy files...')
    p1_coords = np.load(os.path.join(p2dir, 'p1_coords.npy'))
    p1_topo = np.load(os.path.join(p2dir, 'p1_topo.npy'))
    times_float = np.load(os.path.join(p2dir, 'timesteps.npy'))
    print(f'  P1: {len(p1_coords)} nodes, {len(p1_topo)} cells, {len(times_float)} timesteps')

    print('Loading EP mesh for interpolation...')
    ep_coords, ep_topo = load_mesh_h5py(h5file, 'ep')
    mech_centres = cell_centres(p1_coords, p1_topo)
    ep_centres = cell_centres(ep_coords, ep_topo)

    print('Building KD-tree...')
    tree = cKDTree(ep_centres)
    _, ep_to_mech_idx = tree.query(mech_centres)

    print('Writing geometry...')
    geo_file = 'biv.geo'
    write_geo_file(os.path.join(outdir, geo_file), p1_coords, p1_topo)

    mech_time_strs = get_timesteps(h5file, 'mechanics')
    ep_time_strs = get_timesteps(h5file, 'ep')
    all_time_strs = sorted(set(mech_time_strs) | set(ep_time_strs), key=float)
    print(f'  {len(all_time_strs)} timesteps')

    with h5py.File(h5file, 'r') as f:
        mech_fields = [fn for fn in f['mechanics'].keys() if fn != 'u']
        ep_fields = list(f['ep'].keys())

    scalar_vars = []
    vector_vars = []

    # Displacement u — vector per node
    var_name = 'mech_u'
    pattern = f'{var_name}.****'
    print(f'Processing {var_name}...')
    for i, t_str in enumerate(all_time_strs):
        out_path = os.path.join(outdir, f'{var_name}.{i+1:04d}')
        u_file = os.path.join(p2dir, f'u_{t_str}.npy')
        if os.path.exists(u_file):
            u_vals = np.load(u_file)
        else:
            u_vals = np.zeros((len(p1_coords), 3))
        write_vector_node_file(out_path, u_vals, var_name)
    vector_vars.append((var_name, pattern))

    # Scalar mechanics fields — per element
    for field_name in mech_fields:
        var_name = f'mech_{field_name}'
        pattern = f'{var_name}.****'
        print(f'Processing {var_name}...')
        for i, t_str in enumerate(all_time_strs):
            out_path = os.path.join(outdir, f'{var_name}.{i+1:04d}')
            if t_str in mech_time_strs:
                vals = load_field_h5py(h5file, 'mechanics', field_name, t_str)
            else:
                vals = np.zeros(len(p1_topo))
            write_scalar_file(out_path, vals, var_name)
        scalar_vars.append((var_name, pattern))

    # EP fields — per element, interpolated onto mechanics mesh
    for field_name in ep_fields:
        var_name = f'ep_{field_name}'
        pattern = f'{var_name}.****'
        print(f'Processing {var_name} (EP->mechanics)...')
        for i, t_str in enumerate(all_time_strs):
            out_path = os.path.join(outdir, f'{var_name}.{i+1:04d}')
            if t_str in ep_time_strs:
                ep_vals = load_field_h5py(h5file, 'ep', field_name, t_str)
                vals = ep_vals[ep_to_mech_idx]
            else:
                vals = np.zeros(len(p1_topo))
            write_scalar_file(out_path, vals, var_name)
        scalar_vars.append((var_name, pattern))

    case_path = os.path.join(outdir, 'biv.case')
    write_case_file(case_path, geo_file, scalar_vars, vector_vars, times_float)
    print(f'Done. Open in ParaView: {case_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--p2', required=True, help='Directory with extracted numpy files')
    parser.add_argument('--out', default='ensight')
    args = parser.parse_args()
    convert(args.results, args.p2, args.out)