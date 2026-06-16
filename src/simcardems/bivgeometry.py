from typing import Dict
from typing import Tuple

import dolfin
import pulse
from cardiac_geometries.geometry import MeshTypes

from .geometry import BaseGeometry


class BiVentricularGeometry(BaseGeometry):
    @staticmethod
    def default_markers() -> Dict[str, Tuple[int, int]]:
        return {
            "BASE": (10, 2),
            "ENDO_RV": (20, 2),
            "ENDO_LV": (30, 2),
            "EPI": (40, 2),
        }

    def _default_microstructure(
        self,
        mesh: dolfin.Mesh,
        ffun: dolfin.MeshFunction,
    ) -> pulse.Microstructure:
        import ldrb

        markers = self.markers

        # ldrb.dolfin_ldrb expects a fiber_space string like "P_1" / "Quadrature_3"
        fiber_space = self.parameters["fiber_space"]

        # Map BaseGeometry/benchmark marker names onto ldrb's expected keys
        ldrb_markers = {
            "base": markers["BASE"][0],
            "lv": markers["ENDO_LV"][0],
            "rv": markers["ENDO_RV"][0],
            "epi": markers["EPI"][0],
        }

        angles = dict(
            alpha_endo_lv=self.parameters["fibers_angle_endo_lv"],
            alpha_epi_lv=self.parameters["fibers_angle_epi_lv"],
            beta_endo_lv=self.parameters["fibers_sheet_endo_lv"],
            beta_epi_lv=self.parameters["fibers_sheet_epi_lv"],
            alpha_endo_sept=self.parameters["fibers_angle_endo_sept"],
            alpha_epi_sept=self.parameters["fibers_angle_epi_sept"],
            beta_endo_sept=self.parameters["fibers_sheet_endo_sept"],
            beta_epi_sept=self.parameters["fibers_sheet_epi_sept"],
            alpha_endo_rv=self.parameters["fibers_angle_endo_rv"],
            alpha_epi_rv=self.parameters["fibers_angle_epi_rv"],
            beta_endo_rv=self.parameters["fibers_sheet_endo_rv"],
            beta_epi_rv=self.parameters["fibers_sheet_epi_rv"],
        )

        fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
            mesh=mesh,
            fiber_space=fiber_space,
            ffun=ffun,
            markers=ldrb_markers,
            **angles,
        )

        return pulse.Microstructure(f0=fiber, s0=sheet, n0=sheet_normal)

    def _default_ffun(self, mesh: dolfin.Mesh) -> dolfin.MeshFunction:
        raise NotImplementedError

    def _default_mesh(self) -> dolfin.Mesh:
        raise NotImplementedError

    @staticmethod
    def default_parameters():
        return {
            "num_refinements": 1,
            "fiber_space": "Quadrature_3",
            "mesh_type": MeshTypes.biv_ellipsoid.value,  # placeholder — check cardiac_geometries 1.1.7
            "fibers_angle_endo_lv": 30.0,
            "fibers_angle_epi_lv": -30.0,
            "fibers_sheet_endo_lv": 0.0,
            "fibers_sheet_epi_lv": 0.0,
            "fibers_angle_endo_sept": 60.0,
            "fibers_angle_epi_sept": -60.0,
            "fibers_sheet_endo_sept": 0.0,
            "fibers_sheet_epi_sept": 0.0,
            "fibers_angle_endo_rv": 80.0,
            "fibers_angle_epi_rv": -80.0,
            "fibers_sheet_endo_rv": 0.0,
            "fibers_sheet_epi_rv": 0.0,
        }

    def validate(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameters={self.parameters})"