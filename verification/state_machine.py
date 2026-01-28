# Author: Bradley R. Kinnard
# formal state machine definitions for belief/goal/contradiction systems

from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto
import hashlib
import json

from utils.helpers import get_logger

logger = get_logger(__name__)


class TransitionType(Enum):
    """enumeration of valid state transitions."""
    BELIEF_INSERT = auto()
    BELIEF_UPDATE = auto()
    BELIEF_DECAY = auto()
    BELIEF_DELETE = auto()
    GOAL_INSERT = auto()
    GOAL_UPDATE = auto()
    GOAL_COLLAPSE = auto()
    GOAL_DELETE = auto()
    CONTRADICTION_DETECT = auto()
    CONTRADICTION_RESOLVE = auto()


@dataclass(frozen=True)
class BeliefState:
    """immutable belief state representation for formal reasoning."""
    belief_id: str
    content_hash: str
    confidence: float  # must be in [0, 1]
    timestamp: float
    source: str = "unknown"

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "content_hash": self.content_hash,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BeliefState":
        return cls(
            belief_id=d["belief_id"],
            content_hash=d["content_hash"],
            confidence=d["confidence"],
            timestamp=d["timestamp"],
            source=d.get("source", "unknown"),
        )


@dataclass(frozen=True)
class GoalState:
    """immutable goal state for formal reasoning."""
    goal_id: str
    content_hash: str
    priority: float  # must be in [0, 1]
    status: str  # active, collapsed, abandoned
    parent_id: str | None = None

    def __post_init__(self):
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be in [0,1], got {self.priority}")
        if self.status not in ("active", "collapsed", "abandoned"):
            raise ValueError(f"invalid status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "content_hash": self.content_hash,
            "priority": self.priority,
            "status": self.status,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoalState":
        return cls(
            goal_id=d["goal_id"],
            content_hash=d["content_hash"],
            priority=d["priority"],
            status=d["status"],
            parent_id=d.get("parent_id"),
        )


@dataclass(frozen=True)
class Contradiction:
    """immutable contradiction record."""
    contradiction_id: str
    belief_a_id: str
    belief_b_id: str
    detected_at: float
    resolved: bool = False
    resolution_strategy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "belief_a_id": self.belief_a_id,
            "belief_b_id": self.belief_b_id,
            "detected_at": self.detected_at,
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy,
        }


@dataclass
class SystemState:
    """
    complete system state for formal verification.
    designed for deterministic hashing and comparison.
    """
    beliefs: dict[str, BeliefState] = field(default_factory=dict)
    goals: dict[str, GoalState] = field(default_factory=dict)
    contradictions: dict[str, Contradiction] = field(default_factory=dict)
    step_counter: int = 0
    seed: int = 42

    def digest(self) -> str:
        """compute deterministic hash of entire state."""
        state_repr = {
            "beliefs": {k: v.to_dict() for k, v in sorted(self.beliefs.items())},
            "goals": {k: v.to_dict() for k, v in sorted(self.goals.items())},
            "contradictions": {k: v.to_dict() for k, v in sorted(self.contradictions.items())},
            "step_counter": self.step_counter,
            "seed": self.seed,
        }
        canonical = json.dumps(state_repr, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def unresolved_contradictions(self) -> list[Contradiction]:
        """return list of unresolved contradictions."""
        return [c for c in self.contradictions.values() if not c.resolved]

    def active_goals(self) -> list[GoalState]:
        """return list of active goals."""
        return [g for g in self.goals.values() if g.status == "active"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "beliefs": {k: v.to_dict() for k, v in self.beliefs.items()},
            "goals": {k: v.to_dict() for k, v in self.goals.items()},
            "contradictions": {k: v.to_dict() for k, v in self.contradictions.items()},
            "step_counter": self.step_counter,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SystemState":
        return cls(
            beliefs={k: BeliefState.from_dict(v) for k, v in d.get("beliefs", {}).items()},
            goals={k: GoalState.from_dict(v) for k, v in d.get("goals", {}).items()},
            contradictions={k: Contradiction(**v) for k, v in d.get("contradictions", {}).items()},
            step_counter=d.get("step_counter", 0),
            seed=d.get("seed", 42),
        )


@dataclass(frozen=True)
class StateTransition:
    """
    immutable record of a state transition for trace verification.
    includes pre/post state digests for ZK-friendliness.
    """
    transition_type: TransitionType
    pre_state_digest: str
    post_state_digest: str
    input_digest: str
    step_number: int
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_type": self.transition_type.name,
            "pre_state_digest": self.pre_state_digest,
            "post_state_digest": self.post_state_digest,
            "input_digest": self.input_digest,
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "details": self.details,
        }

    def verify_chain(self, previous: "StateTransition | None") -> bool:
        """verify this transition chains correctly from previous."""
        if previous is None:
            return True
        return previous.post_state_digest == self.pre_state_digest


class TransitionEngine:
    """
    formal transition engine that tracks all state changes.
    ensures every mutation is recorded and verifiable.
    """

    def __init__(self, initial_state: SystemState | None = None):
        self._state = initial_state or SystemState()
        self._trace: list[StateTransition] = []

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def trace(self) -> list[StateTransition]:
        return self._trace.copy()

    def _record_transition(
        self,
        transition_type: TransitionType,
        pre_digest: str,
        input_data: dict[str, Any],
        details: dict[str, Any] | None = None,
    ) -> StateTransition:
        """record a transition after state mutation."""
        import time

        input_json = json.dumps(input_data, sort_keys=True, separators=(",", ":"))
        input_digest = hashlib.sha256(input_json.encode()).hexdigest()

        transition = StateTransition(
            transition_type=transition_type,
            pre_state_digest=pre_digest,
            post_state_digest=self._state.digest(),
            input_digest=input_digest,
            step_number=self._state.step_counter,
            timestamp=time.time(),
            details=details or {},
        )
        self._trace.append(transition)
        return transition

    def insert_belief(self, belief: BeliefState) -> StateTransition:
        """insert a new belief with full transition tracking."""
        pre_digest = self._state.digest()

        if belief.belief_id in self._state.beliefs:
            raise ValueError(f"belief {belief.belief_id} already exists")

        self._state.beliefs[belief.belief_id] = belief
        self._state.step_counter += 1

        return self._record_transition(
            TransitionType.BELIEF_INSERT,
            pre_digest,
            {"belief": belief.to_dict()},
            {"belief_id": belief.belief_id},
        )

    def update_belief(
        self,
        belief_id: str,
        new_confidence: float,
        timestamp: float,
    ) -> StateTransition:
        """update belief confidence with transition tracking."""
        pre_digest = self._state.digest()

        if belief_id not in self._state.beliefs:
            raise KeyError(f"belief {belief_id} not found")

        old = self._state.beliefs[belief_id]
        updated = BeliefState(
            belief_id=old.belief_id,
            content_hash=old.content_hash,
            confidence=max(0.0, min(1.0, new_confidence)),
            timestamp=timestamp,
            source=old.source,
        )
        self._state.beliefs[belief_id] = updated
        self._state.step_counter += 1

        return self._record_transition(
            TransitionType.BELIEF_UPDATE,
            pre_digest,
            {"belief_id": belief_id, "new_confidence": new_confidence},
            {"old_confidence": old.confidence, "new_confidence": updated.confidence},
        )

    def resolve_contradiction(
        self,
        contradiction_id: str,
        strategy: str,
    ) -> StateTransition:
        """mark contradiction as resolved."""
        pre_digest = self._state.digest()

        if contradiction_id not in self._state.contradictions:
            raise KeyError(f"contradiction {contradiction_id} not found")

        old = self._state.contradictions[contradiction_id]
        if old.resolved:
            raise ValueError(f"contradiction {contradiction_id} already resolved")

        resolved = Contradiction(
            contradiction_id=old.contradiction_id,
            belief_a_id=old.belief_a_id,
            belief_b_id=old.belief_b_id,
            detected_at=old.detected_at,
            resolved=True,
            resolution_strategy=strategy,
        )
        self._state.contradictions[contradiction_id] = resolved
        self._state.step_counter += 1

        return self._record_transition(
            TransitionType.CONTRADICTION_RESOLVE,
            pre_digest,
            {"contradiction_id": contradiction_id, "strategy": strategy},
            {"resolved": True},
        )

    def verify_trace_integrity(self) -> tuple[bool, str]:
        """verify entire trace chains correctly."""
        if not self._trace:
            return True, "empty trace"

        for i, t in enumerate(self._trace):
            prev = self._trace[i - 1] if i > 0 else None
            if not t.verify_chain(prev):
                return False, f"chain break at step {i}"

        return True, "trace valid"
