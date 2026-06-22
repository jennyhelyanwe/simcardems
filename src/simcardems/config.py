import json
import logging
import typing
from dataclasses import dataclass, field
import dataclasses

import dolfin
import typing_extensions

from . import utils


@dataclass
class WindkesselParams:
    p_init: float = 0.0
    compliance: float = 0.0
    resistance: float = 0.0
    evolve: bool = True


@dataclass
class CycleParams:
    t_zero: float = 0.0
    t_prestress: float = 0.0
    preload_pressure: float = 0.0
    prestress_pressure: float = 0.0
    t_end_diastole: float = 0.0
    p_end_diastole: float = 0.0
    gain_contraction: typing.Tuple[float, float] = (0.0, 0.0)
    gain_relaxation: typing.Tuple[float, float] = (0.0, 0.0)
    p_fill: float = 0.0
    period: float = 1000.0
    filling_gain: bool = False
    windkessel: WindkesselParams = field(default_factory=WindkesselParams)


@dataclass
class Config:
    outdir: utils.PathLike = "results"
    outfilename: str = "results.h5"
    geometry_path: utils.PathLike = ""
    geometry_schema_path: typing.Optional[utils.PathLike] = None
    T: float = 1000
    dt: float = 0.05
    bnd_rigid: bool = False
    load_state: bool = False
    cell_init_file: utils.PathLike = ""
    show_progress_bar: bool = True
    save_freq: int = 1
    pre_stretch: typing.Optional[typing.Union[dolfin.Constant, float]] = None
    traction: typing.Union[dolfin.Constant, float] = None
    spring: typing.Union[dolfin.Constant, float] = None
    fix_right_plane: bool = False
    loglevel: int = logging.INFO
    num_refinements: int = 1
    set_material: str = ""
    debug_mode: bool = False
    drug_factors_file: utils.PathLike = ""
    popu_factors_file: utils.PathLike = ""
    disease_state: str = "healthy"
    dt_mech: float = 1.0
    mech_threshold: float = 0.05
    mechanics_solve_strategy: typing_extensions.Literal["fixed", "adaptive"] = "adaptive"
    ep_ode_scheme: str = "GRL1"
    ep_preconditioner: str = "sor"
    ep_theta: float = 0.5
    linear_mechanics_solver: str = "mumps"
    mechanics_use_continuation: bool = False
    mechanics_use_custom_newton_solver: bool = False
    PCL: float = 1000
    coupling_type: typing_extensions.Literal[
        "fully_coupled_ORdmm_Land",
        "fully_coupled_Tor_Land",
        "explicit_ORdmm_Land",
        "pureEP_ORdmm_Land",
    ] = "fully_coupled_ORdmm_Land"

    # --- BiV cardiac cycle control, per cavity ---
    cycle_lv: CycleParams = field(default_factory=CycleParams)
    cycle_rv: CycleParams = field(default_factory=CycleParams)

    def as_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if dataclasses.is_dataclass(v):
                result[k] = dataclasses.asdict(v)
            else:
                result[k] = v
        return result

    @classmethod
    def from_json(cls, path: utils.PathLike) -> "Config":
        config = cls()
        config.update_from_json(path)
        return config

    def update_from_json(self, path: utils.PathLike) -> None:
        """
        Populate this Config from a JSON parameter file. All values are
        taken as-is, no unit conversion -- the JSON is expected to already
        be in simcardems's units (kPa, mm, ms). Cavity-indexed arrays
        (cycle parameters) are mapped via 'cavity_bcs' (e.g. ["LV", "RV"]).

        Unrecognized keys are silently skipped.
        """
        data = json.loads(utils.Path(path).read_text())

        if "cavity_bcs" not in data:
            logging.getLogger(__name__).warning(
                "No 'cavity_bcs' found in %s -- skipping cycle fields", path
            )
            return

        order = data["cavity_bcs"]
        idx = {name: i for i, name in enumerate(order)}
        if "LV" not in idx or "RV" not in idx:
            raise ValueError(f"Expected 'LV' and 'RV' in cavity_bcs, got {order!r}")

        def get(key, cavity):
            if key not in data:
                return None
            return data[key][idx[cavity]]

        for cavity, cycle_attr in (("LV", "cycle_lv"), ("RV", "cycle_rv")):
            cycle: CycleParams = getattr(self, cycle_attr)

            def gv(key):
                return get(key, cavity)

            if (v := gv("prestress_t")) is not None:
                cycle.t_prestress = v
            if (v := gv("prestress_p")) is not None:
                cycle.prestress_pressure = v
            if (v := gv("diastasis_t")) is not None:
                cycle.t_zero = v
            if (v := gv("diastasis_p")) is not None:
                cycle.preload_pressure = v
            if (v := gv("end_diastole_t")) is not None:
                cycle.t_end_diastole = v
            if (v := gv("end_diastole_p")) is not None:
                cycle.p_end_diastole = v
            if (v := gv("gain_error_contraction")) is not None:
                cycle.gain_contraction = (v, cycle.gain_contraction[1])
            if (v := gv("gain_derror_contraction")) is not None:
                cycle.gain_contraction = (cycle.gain_contraction[0], v)
            if (v := gv("gain_error_relaxation")) is not None:
                cycle.gain_relaxation = (v, cycle.gain_relaxation[1])
            if (v := gv("gain_derror_relaxation")) is not None:
                cycle.gain_relaxation = (cycle.gain_relaxation[0], v)
            if (v := gv("filling_pressure_threshold")) is not None:
                cycle.p_fill = v
            if "cycle_length" in data:
                cycle.period = data["cycle_length"]

            if (v := gv("arterial_compliance")) is not None:
                cycle.windkessel.compliance = v
            if (v := gv("arterial_resistance")) is not None:
                cycle.windkessel.resistance = v
            if (v := gv("ejection_pressure_threshold")) is not None:
                cycle.windkessel.p_init = v


def default_parameters():
    return {k: v for k, v in Config.__dict__.items() if not k.startswith("_")}