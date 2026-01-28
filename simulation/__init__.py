# Author: Bradley R. Kinnard
# world model engine - physics simulation for predictive reasoning

from dataclasses import dataclass, field
from typing import Any, Protocol
from abc import ABC, abstractmethod
import time

import numpy as np
from scipy import integrate

from utils.helpers import get_logger

logger = get_logger(__name__)


# check available backends
_MUJOCO_AVAILABLE = False
_PYBULLET_AVAILABLE = False

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
    logger.info("MuJoCo available")
except ImportError:
    pass

try:
    import pybullet
    _PYBULLET_AVAILABLE = True
    logger.info("PyBullet available")
except ImportError:
    pass


@dataclass
class WorldState:
    """state of the simulated world."""
    positions: dict[str, np.ndarray]
    velocities: dict[str, np.ndarray]
    time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": {k: v.tolist() for k, v in self.positions.items()},
            "velocities": {k: v.tolist() for k, v in self.velocities.items()},
            "time": self.time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorldState":
        return cls(
            positions={k: np.array(v) for k, v in d.get("positions", {}).items()},
            velocities={k: np.array(v) for k, v in d.get("velocities", {}).items()},
            time=d.get("time", 0.0),
            metadata=d.get("metadata", {}),
        )


@dataclass
class SimulationResult:
    """result of a simulation run."""
    trajectory: list[WorldState]
    final_state: WorldState
    duration: float
    n_steps: int
    backend: str
    metrics: dict[str, float] = field(default_factory=dict)


class PhysicsBackend(ABC):
    """abstract base for physics backends."""

    @abstractmethod
    def step(self, state: WorldState, action: dict[str, Any], dt: float) -> WorldState:
        """advance simulation by one step."""
        pass

    @abstractmethod
    def simulate(
        self,
        initial_state: WorldState,
        actions: list[dict[str, Any]],
        dt: float,
    ) -> SimulationResult:
        """run full simulation."""
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        pass


class SciPyBackend(PhysicsBackend):
    """
    scipy-based physics simulation (always available).

    uses ODE integration for simple dynamics.
    """

    def __init__(self, gravity: float = 9.81):
        self._gravity = gravity

    def get_backend_name(self) -> str:
        return "scipy"

    def step(self, state: WorldState, action: dict[str, Any], dt: float) -> WorldState:
        """advance state by dt using simple dynamics."""
        new_positions = {}
        new_velocities = {}

        force = action.get("force", np.zeros(3))
        if isinstance(force, list):
            force = np.array(force)

        for name, pos in state.positions.items():
            vel = state.velocities.get(name, np.zeros_like(pos))

            # simple Euler integration with gravity and force
            mass = action.get("mass", 1.0)
            acc = force / mass
            if len(acc) >= 3:
                acc[2] -= self._gravity  # gravity in z

            new_vel = vel + acc * dt
            new_pos = pos + new_vel * dt

            new_positions[name] = new_pos
            new_velocities[name] = new_vel

        return WorldState(
            positions=new_positions,
            velocities=new_velocities,
            time=state.time + dt,
            metadata=state.metadata.copy(),
        )

    def simulate(
        self,
        initial_state: WorldState,
        actions: list[dict[str, Any]],
        dt: float,
    ) -> SimulationResult:
        """run simulation with scipy backend."""
        trajectory = [initial_state]
        current = initial_state

        for action in actions:
            current = self.step(current, action, dt)
            trajectory.append(current)

        return SimulationResult(
            trajectory=trajectory,
            final_state=current,
            duration=current.time - initial_state.time,
            n_steps=len(actions),
            backend="scipy",
        )


class MuJoCoBackend(PhysicsBackend):
    """
    MuJoCo physics backend for high-fidelity simulation.
    """

    def __init__(self, model_xml: str | None = None):
        if not _MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo not available")

        # default simple model
        if model_xml is None:
            model_xml = """
            <mujoco>
                <worldbody>
                    <body name="object" pos="0 0 1">
                        <joint type="free"/>
                        <geom type="sphere" size="0.1" mass="1"/>
                    </body>
                    <body name="ground">
                        <geom type="plane" size="10 10 0.1"/>
                    </body>
                </worldbody>
            </mujoco>
            """

        self._model = mujoco.MjModel.from_xml_string(model_xml)
        self._data = mujoco.MjData(self._model)

    def get_backend_name(self) -> str:
        return "mujoco"

    def _state_from_mujoco(self) -> WorldState:
        """extract state from MuJoCo data."""
        positions = {}
        velocities = {}

        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                positions[name] = self._data.xpos[i].copy()
                velocities[name] = self._data.cvel[i, 3:6].copy()  # linear velocity

        return WorldState(
            positions=positions,
            velocities=velocities,
            time=self._data.time,
        )

    def step(self, state: WorldState, action: dict[str, Any], dt: float) -> WorldState:
        """step MuJoCo simulation."""
        # apply controls
        if "ctrl" in action:
            ctrl = action["ctrl"]
            if isinstance(ctrl, list):
                ctrl = np.array(ctrl)
            n_ctrl = min(len(ctrl), self._model.nu)
            self._data.ctrl[:n_ctrl] = ctrl[:n_ctrl]

        # step simulation
        mujoco.mj_step(self._model, self._data)

        return self._state_from_mujoco()

    def simulate(
        self,
        initial_state: WorldState,
        actions: list[dict[str, Any]],
        dt: float,
    ) -> SimulationResult:
        """run full MuJoCo simulation."""
        # reset to initial state
        mujoco.mj_resetData(self._model, self._data)

        # set initial positions if provided
        for name, pos in initial_state.positions.items():
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                # note: this is simplified; real MuJoCo uses qpos
                pass

        trajectory = [self._state_from_mujoco()]

        for action in actions:
            state = self.step(trajectory[-1], action, dt)
            trajectory.append(state)

        return SimulationResult(
            trajectory=trajectory,
            final_state=trajectory[-1],
            duration=trajectory[-1].time - trajectory[0].time,
            n_steps=len(actions),
            backend="mujoco",
        )


class PyBulletBackend(PhysicsBackend):
    """
    PyBullet physics backend.
    """

    def __init__(self, use_gui: bool = False):
        if not _PYBULLET_AVAILABLE:
            raise RuntimeError("PyBullet not available")

        import pybullet as p

        self._p = p
        mode = p.GUI if use_gui else p.DIRECT
        self._client = p.connect(mode)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)

        # create ground plane
        p.createCollisionShape(p.GEOM_PLANE)
        self._ground = p.createMultiBody(0, 0)

        # create default object
        self._sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
        self._object = p.createMultiBody(
            1.0, self._sphere_shape, basePosition=[0, 0, 1]
        )

        self._dt = 1.0 / 240.0

    def get_backend_name(self) -> str:
        return "pybullet"

    def _state_from_pybullet(self) -> WorldState:
        """extract state from PyBullet."""
        pos, orn = self._p.getBasePositionAndOrientation(
            self._object, physicsClientId=self._client
        )
        vel, ang_vel = self._p.getBaseVelocity(
            self._object, physicsClientId=self._client
        )

        return WorldState(
            positions={"object": np.array(pos)},
            velocities={"object": np.array(vel)},
            time=0.0,  # pybullet doesn't track time directly
        )

    def step(self, state: WorldState, action: dict[str, Any], dt: float) -> WorldState:
        """step PyBullet simulation."""
        # apply force if specified
        if "force" in action:
            force = action["force"]
            if isinstance(force, list):
                force = force[:3]
            self._p.applyExternalForce(
                self._object,
                -1,
                force,
                [0, 0, 0],
                self._p.WORLD_FRAME,
                physicsClientId=self._client,
            )

        # step simulation
        n_substeps = max(1, int(dt / self._dt))
        for _ in range(n_substeps):
            self._p.stepSimulation(physicsClientId=self._client)

        return self._state_from_pybullet()

    def simulate(
        self,
        initial_state: WorldState,
        actions: list[dict[str, Any]],
        dt: float,
    ) -> SimulationResult:
        """run full PyBullet simulation."""
        # reset object position
        if "object" in initial_state.positions:
            self._p.resetBasePositionAndOrientation(
                self._object,
                initial_state.positions["object"].tolist(),
                [0, 0, 0, 1],
                physicsClientId=self._client,
            )
        if "object" in initial_state.velocities:
            self._p.resetBaseVelocity(
                self._object,
                initial_state.velocities["object"].tolist(),
                [0, 0, 0],
                physicsClientId=self._client,
            )

        trajectory = [self._state_from_pybullet()]

        for action in actions:
            state = self.step(trajectory[-1], action, dt)
            trajectory.append(state)

        return SimulationResult(
            trajectory=trajectory,
            final_state=trajectory[-1],
            duration=len(actions) * dt,
            n_steps=len(actions),
            backend="pybullet",
        )

    def __del__(self):
        if hasattr(self, "_client"):
            self._p.disconnect(self._client)


class WorldModelEngine:
    """
    unified world model engine with backend selection.

    automatically selects best available backend:
    1. MuJoCo (highest fidelity)
    2. PyBullet (good fidelity, easier setup)
    3. SciPy (always available, basic dynamics)
    """

    def __init__(self, preferred_backend: str | None = None):
        self._backend = self._select_backend(preferred_backend)
        logger.info(f"world model engine using backend: {self._backend.get_backend_name()}")

    def _select_backend(self, preferred: str | None) -> PhysicsBackend:
        """select best available backend."""
        if preferred == "mujoco" and _MUJOCO_AVAILABLE:
            return MuJoCoBackend()
        if preferred == "pybullet" and _PYBULLET_AVAILABLE:
            return PyBulletBackend()
        if preferred == "scipy":
            return SciPyBackend()

        # auto-select
        if _MUJOCO_AVAILABLE:
            return MuJoCoBackend()
        if _PYBULLET_AVAILABLE:
            return PyBulletBackend()

        return SciPyBackend()

    @property
    def backend_name(self) -> str:
        return self._backend.get_backend_name()

    def simulate(
        self,
        initial_state: WorldState,
        actions: list[dict[str, Any]],
        dt: float = 0.01,
    ) -> SimulationResult:
        """run simulation with current backend."""
        return self._backend.simulate(initial_state, actions, dt)

    def predict_outcome(
        self,
        current_state: WorldState,
        action_sequence: list[dict[str, Any]],
        dt: float = 0.01,
    ) -> dict[str, Any]:
        """predict outcome of action sequence."""
        result = self.simulate(current_state, action_sequence, dt)

        # compute summary metrics
        initial_pos = list(current_state.positions.values())[0] if current_state.positions else np.zeros(3)
        final_pos = list(result.final_state.positions.values())[0] if result.final_state.positions else np.zeros(3)

        displacement = np.linalg.norm(final_pos - initial_pos)
        final_height = final_pos[2] if len(final_pos) > 2 else 0.0

        return {
            "success": True,
            "displacement": float(displacement),
            "final_height": float(final_height),
            "duration": result.duration,
            "n_steps": result.n_steps,
            "backend": result.backend,
            "final_state": result.final_state.to_dict(),
        }

    def counterfactual(
        self,
        base_state: WorldState,
        base_actions: list[dict[str, Any]],
        alt_actions: list[dict[str, Any]],
        dt: float = 0.01,
    ) -> dict[str, Any]:
        """
        compare outcomes of two action sequences.

        answers: "what would have happened if I did alt_actions instead?"
        """
        base_result = self.simulate(base_state, base_actions, dt)
        alt_result = self.simulate(base_state, alt_actions, dt)

        base_final = list(base_result.final_state.positions.values())[0] if base_result.final_state.positions else np.zeros(3)
        alt_final = list(alt_result.final_state.positions.values())[0] if alt_result.final_state.positions else np.zeros(3)

        divergence = np.linalg.norm(alt_final - base_final)

        return {
            "base_outcome": base_result.final_state.to_dict(),
            "alt_outcome": alt_result.final_state.to_dict(),
            "divergence": float(divergence),
            "base_better": base_final[2] > alt_final[2] if len(base_final) > 2 else False,
        }
