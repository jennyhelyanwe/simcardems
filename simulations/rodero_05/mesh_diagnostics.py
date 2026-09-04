import numpy as np
from simcardems import utils

logger = utils.getLogger(__name__)


def check_orthonormality(f0, s0, n0, label=""):
    def as_array(x):
        return x.vector().get_local().reshape(-1, 3) if hasattr(x, "vector") else x
    F, S, N = as_array(f0), as_array(s0), as_array(n0)
    f0n = np.linalg.norm(F, axis=1)
    dot_fs = np.abs(np.sum(F * S, axis=1))
    dot_fn = np.abs(np.sum(F * N, axis=1))
    dot_sn = np.abs(np.sum(S * N, axis=1))
    logger.info(f"{label} f0 norm: {f0n.min():.6f}-{f0n.max():.6f}, "
                f"max|f0.s0|={dot_fs.max():.3e}, max|f0.n0|={dot_fn.max():.3e}, max|s0.n0|={dot_sn.max():.3e}")

def tet_volumes(coords, cells_arr, signed=False):
    p0, p1, p2, p3 = (coords[cells_arr[:, i]] for i in range(4))
    vol = np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0)) / 6.0
    return vol if signed else np.abs(vol)

# def tet_volumes(coords, cells_arr):
#     p0, p1, p2, p3 = (coords[cells_arr[:, i]] for i in range(4))
#     return np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
#

def tet_quality_ratios(coords, cells_arr):
    p0, p1, p2, p3 = (coords[cells_arr[:, i]] for i in range(4))
    vol = np.abs(np.einsum('ij,ij->i', p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0

    def tri_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    surface_area = tri_area(p1, p2, p3) + tri_area(p0, p2, p3) + tri_area(p0, p1, p3) + tri_area(p0, p1, p2)
    inradius = 3 * vol / surface_area

    a, b, c = (np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p0, axis=1), np.linalg.norm(p3 - p0, axis=1))
    a1, b1, c1 = (np.linalg.norm(p3 - p2, axis=1), np.linalg.norm(p3 - p1, axis=1), np.linalg.norm(p2 - p1, axis=1))
    num = np.sqrt((a*a1+b*b1+c*c1) * (a*a1+b*b1-c*c1) * (a*a1-b*b1+c*c1) * (-a*a1+b*b1+c*c1))
    circumradius = num / (24 * vol)
    return inradius / circumradius