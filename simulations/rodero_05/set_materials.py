import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Load fine mesh centroids (in cm, convert to mm)
fine_centres = pd.read_csv(
    './rodero_05_fine/rodero_05_fine_tetrahedron_centers.csv',
    header=None
).to_numpy() * 10.0

# Load material labels
tv = pd.read_csv(
    './rodero_05_fine/rodero_05_fine_elementfield_tv-element.csv',
    header=None
).to_numpy().flatten().astype(int)

print(f'Fine mesh: {len(fine_centres)} elements')
print(f'Unique material labels: {np.unique(tv)}')

# Load coarse mesh centroids from h5
import h5py
with h5py.File('./rodero_05_coarse_4mm.h5', 'r') as f:
    coords = f['mesh/coordinates'][:]
    topo = f['mesh/topology'][:]

coarse_centres = coords[topo].mean(axis=1)
print(f'Coarse mesh: {len(coarse_centres)} elements')

# Nearest neighbour mapping
tree = cKDTree(fine_centres)
_, idx = tree.query(coarse_centres)
coarse_tv = tv[idx]

print(f'Coarse material labels: {np.unique(coarse_tv)}')
print(f'Valve plug elements (7-10): {np.sum(coarse_tv >= 7)}')

np.save('./rodero_05_coarse_tv.npy', coarse_tv)
print('Saved rodero_05_coarse_tv.npy')