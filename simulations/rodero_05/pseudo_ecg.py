import numpy as np
import dolfin
from simcardems.postprocess import ecg_recovery


class PseudoECG:
    def __init__(self, leads: dict, sigma_b: float = 1.0):
        self.leads = leads
        self.sigma_b = sigma_b
        self._file = None

    def open_file(self, path: str):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0:
            self._file = open(path, "w")
            self._file.write("t_ms," + ",".join(self.leads.keys()) + "\n")
            self._file.flush()

    def compute(self, v: dolfin.Function, t: float):
        mesh = v.function_space().mesh()
        vals = []
        for name, (pos, neg) in self.leads.items():
            phi_pos = ecg_recovery(v=v, sigma_b=self.sigma_b, point=np.array(pos), mesh=mesh)
            phi_neg = ecg_recovery(v=v, sigma_b=self.sigma_b, point=np.array(neg), mesh=mesh)
            vals.append(phi_pos - phi_neg)
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._file is not None:
            self._file.write(f"{t:.3f}," + ",".join(f"{v:.8f}" for v in vals) + "\n")
            self._file.flush()

    def close(self):
        if dolfin.MPI.rank(dolfin.MPI.comm_world) == 0 and self._file is not None:
            self._file.close()