"""SCM deformable-terrain backend for the default Chrono Go1 environment."""

from __future__ import annotations

import math

import numpy as np
import pychrono as chrono
import pychrono.parsers as parsers
import pychrono.vehicle as veh

from go1_env import (
    _CONTACT_DIAGNOSTIC_FORCE_LIMIT,
    _FOOT_COLLISION_RADIUS,
    _FOOT_FRICTION,
    _FOOT_RESTITUTION,
    _MATERIAL_COMPOSITION_RULE,
    _PHYSICS_SUBSTEPS,
    _PHYSICS_TIME_STEP,
    Go1Env,
)

_SCM_SOLVER_TYPE = "BARZILAIBORWEIN"
_SCM_SOLVER_ITERATIONS = 60
_SCM_TERRAIN_LENGTH = 20.0
_SCM_TERRAIN_WIDTH = 20.0
_SCM_GRID_SPACING = 0.02
_SCM_BEKKER_KPHI = 3e6
_SCM_BEKKER_KC = 0.0
_SCM_BEKKER_N = 1.1
_SCM_COHESION = 0.0
_SCM_FRICTION_ANGLE_DEG = 30.0
_SCM_JANOSI_SHEAR = 0.0
_SCM_ELASTIC_STIFFNESS = 2e9
_SCM_DAMPING = 3e4
_SCM_PLOT_MIN = 0.0
_SCM_PLOT_MAX = 0.01
_SCM_FRICTION = 0.9
_SCM_RESTITUTION = 0.1
_SCM_GN = 60.0
_SCM_KN = 2e5
_SCM_ACTIVE_DOMAIN_CENTER = (0.0, 0.0, 0.0)
_SCM_ACTIVE_DOMAIN_DIMS = (2.5, 1.5, 2.5)


def scm_env_metadata() -> dict:
    """Return SCM backend constants used by Go1SCMEnv."""
    return {
        "env_backend": "scm",
        "terrain": "scm",
        "scm_physics_dt": _PHYSICS_TIME_STEP,
        "scm_substeps": _PHYSICS_SUBSTEPS,
        "scm_solver_type": _SCM_SOLVER_TYPE,
        "scm_solver_iterations": _SCM_SOLVER_ITERATIONS,
        "scm_terrain_size": [_SCM_TERRAIN_LENGTH, _SCM_TERRAIN_WIDTH],
        "scm_grid_spacing": _SCM_GRID_SPACING,
        "scm_soil": {
            "bekker_kphi": _SCM_BEKKER_KPHI,
            "bekker_kc": _SCM_BEKKER_KC,
            "bekker_n": _SCM_BEKKER_N,
            "cohesion": _SCM_COHESION,
            "friction_angle_degrees": _SCM_FRICTION_ANGLE_DEG,
            "janosi_shear": _SCM_JANOSI_SHEAR,
            "elastic_stiffness": _SCM_ELASTIC_STIFFNESS,
            "damping": _SCM_DAMPING,
        },
        "scm_plot": {
            "type": "sinkage",
            "min": _SCM_PLOT_MIN,
            "max": _SCM_PLOT_MAX,
        },
        "scm_reference_frame": {
            "position": [0.0, 0.0, 0.0],
            "rotation": "QuatFromAngleX(-pi/2)",
            "normal": "+Y",
        },
        "scm_contact_material": {
            "friction": _SCM_FRICTION,
            "restitution": _SCM_RESTITUTION,
            "gn": _SCM_GN,
            "kn": _SCM_KN,
        },
        "scm_active_domain": {
            "body": "trunk",
            "center": list(_SCM_ACTIVE_DOMAIN_CENTER),
            "dims": list(_SCM_ACTIVE_DOMAIN_DIMS),
        },
    }


class Go1SCMEnv(Go1Env):
    """Default Go1 policy interface backed by Chrono SCM deformable terrain."""

    def __init__(self, *args, **kwargs):
        self._env_backend = "scm"
        self._terrain_type = "scm"
        self._contact_method = "SMC"
        self._friction_range = (_SCM_FRICTION, _SCM_FRICTION)
        self._synchronize_terrain = True
        self._scm_terrain = None
        super().__init__(*args, **kwargs)

    def _sample_ground_friction(self) -> float:
        return _SCM_FRICTION

    def _terrain_body_contact_force(self, body):
        if body is None or self._scm_terrain is None:
            return chrono.ChVector3d(0.0, 0.0, 0.0)
        force = chrono.ChVector3d(0.0, 0.0, 0.0)
        torque = chrono.ChVector3d(0.0, 0.0, 0.0)
        has_contact = self._scm_terrain.GetContactForceBody(body, force, torque)
        if not has_contact:
            return chrono.ChVector3d(0.0, 0.0, 0.0)
        return force

    @staticmethod
    def _force_norm(force) -> float:
        return float(math.sqrt(float(force.x) ** 2 + float(force.y) ** 2 + float(force.z) ** 2))

    def _foot_loads(self) -> np.ndarray:
        return np.array(
            [abs(float(self._terrain_body_contact_force(foot).y)) for foot in self._feet],
            dtype=np.float32,
        )

    def _max_nonfoot_load(self) -> float:
        if self._system is None:
            return 0.0
        bodies = {body.GetName(): body for body in self._system.GetBodies()}
        max_load = 0.0
        for leg in ("FR", "FL", "RR", "RL"):
            for group in ("calf", "thigh", "hip"):
                force = self._terrain_body_contact_force(bodies.get(f"{leg}_{group}"))
                max_load = max(max_load, abs(float(force.y)))
        return float(max_load)

    def _contact_diagnostic_count(self) -> tuple[float, dict[str, float]]:
        if self._system is None:
            return 0.0, {}
        bodies = {body.GetName(): body for body in self._system.GetBodies()}
        count = 0.0
        diagnostics: dict[str, float] = {}
        for leg in ("FR", "FL", "RR", "RL"):
            for group in ("thigh", "calf"):
                name = f"{leg}_{group}"
                force_norm = self._force_norm(self._terrain_body_contact_force(bodies.get(name)))
                diagnostics[f"collision_force_norm_{name}"] = float(force_norm)
                if force_norm > _CONTACT_DIAGNOSTIC_FORCE_LIMIT:
                    count += 1.0
        diagnostics["contact_diagnostic_force_limit"] = _CONTACT_DIAGNOSTIC_FORCE_LIMIT
        return float(count), diagnostics

    def _trunk_contact_force_norm(self) -> float:
        return self._force_norm(self._terrain_body_contact_force(self._trunk))

    def _new_system(self):
        system = chrono.ChSystemSMC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, -9.81, 0.0))
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.GetSolver().AsIterative().SetMaxIterations(_SCM_SOLVER_ITERATIONS)
        return system

    def _build_imported_system(self, actuation_type, include_terrain: bool = True):
        system = self._new_system()
        terrain = None
        if include_terrain:
            terrain = self._create_scm_terrain(system)

        parser = self._create_robot_parser(actuation_type)
        parser.PopulateSystem(system)
        self._configure_imported_bodies(system, parser)
        if terrain is not None:
            self._configure_scm_active_domain(terrain, parser)
        if self.visual_mesh_format == "none":
            self._clear_robot_visuals(parser)
        elif self.visual_mesh_format == "urdf":
            self._hide_viewer_sensor_visuals(parser)
        elif self.visual_mesh_format in ("obj", "obj_lod50"):
            self._replace_body_visuals_with_obj(parser)
            self._hide_viewer_sensor_visuals(parser)
        return system, terrain, parser

    def _create_scm_terrain(self, system):
        terrain = veh.SCMTerrain(system)
        terrain.SetReferenceFrame(
            chrono.ChCoordsysd(
                chrono.ChVector3d(0.0, 0.0, 0.0),
                chrono.QuatFromAngleX(-0.5 * math.pi),
            )
        )
        terrain.SetSoilParameters(
            _SCM_BEKKER_KPHI,
            _SCM_BEKKER_KC,
            _SCM_BEKKER_N,
            _SCM_COHESION,
            _SCM_FRICTION_ANGLE_DEG,
            _SCM_JANOSI_SHEAR,
            _SCM_ELASTIC_STIFFNESS,
            _SCM_DAMPING,
        )
        terrain.SetPlotType(veh.SCMTerrain.PLOT_SINKAGE, _SCM_PLOT_MIN, _SCM_PLOT_MAX)
        terrain.Initialize(_SCM_TERRAIN_LENGTH, _SCM_TERRAIN_WIDTH, _SCM_GRID_SPACING)
        self._scm_terrain = terrain
        return terrain

    def _configure_scm_active_domain(self, terrain, parser) -> None:
        trunk = parser.GetChBody("trunk")
        terrain.AddActiveDomain(
            trunk,
            chrono.ChVector3d(*_SCM_ACTIVE_DOMAIN_CENTER),
            chrono.ChVector3d(*_SCM_ACTIVE_DOMAIN_DIMS),
        )

    def _sample_reset_noise(self) -> None:
        super()._sample_reset_noise()
        self._reset_noise_sample["ground_friction"] = _SCM_FRICTION
        self._reset_noise_sample["ground_friction_range"] = [_SCM_FRICTION, _SCM_FRICTION]

    def _command_diagnostics(self) -> dict:
        diag = super()._command_diagnostics()
        diag["friction_min"] = _SCM_FRICTION
        diag["friction_max"] = _SCM_FRICTION
        return diag

    def _scm_info(self) -> dict:
        return scm_env_metadata()

    def _material_info(self) -> dict:
        return {
            "contact_method": self.contact_method,
            "material_composition_rule": _MATERIAL_COMPOSITION_RULE,
            "composition_strategy": "SCM deformable terrain plus SMC robot collision materials",
            "configured_ground_friction": _SCM_FRICTION,
            "configured_foot_friction": _FOOT_FRICTION,
            "effective_friction": _SCM_FRICTION,
            "static_friction": _SCM_FRICTION,
            "sliding_friction": _SCM_FRICTION,
            "static_sliding_note": "SCM shear behavior comes from soil parameters; friction here reports the demo material value.",
            "ground": {
                "friction": _SCM_FRICTION,
                "restitution": _SCM_RESTITUTION,
                "gn": _SCM_GN,
                "kn": _SCM_KN,
            },
            "feet": {
                "friction": _FOOT_FRICTION,
                "restitution": _FOOT_RESTITUTION,
                "collision_radius": _FOOT_COLLISION_RADIUS,
            },
        }

    def _info(self) -> dict:
        info = super()._info()
        info["env_backend"] = "scm"
        info["terrain"] = "scm"
        info["friction_range"] = (_SCM_FRICTION, _SCM_FRICTION)
        info["default_randomization"]["friction"] = _SCM_FRICTION
        info["scm"] = self._scm_info()
        return info
