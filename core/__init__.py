# Author: Bradley R. Kinnard
# core module exports

from core.belief_ecology import BeliefEcology
from core.contradiction_tracer import ContradictionTracer
from core.goal_collapse import GoalCollapse
from core.meta_cognition import MetaCognition
from core.meta_evolution import (
    MetaEvolutionEngine,
    HyperparameterSet,
    EvolutionResult,
    BeliefCoherenceObjective,
    EfficiencyObjective,
    MetaEvolutionDisabledError,
    create_evolution_engine,
)
from core.world_model import (
    SimpleWorldModel,
    WorldModelContext,
    SimulationState,
    SimulationAction,
    WorldModelDisabledError,
    create_world_model,
)

# reasoning agent imports are done lazily to avoid circular imports
# use: from core.reasoning_agent import ReasoningAgent, AgentConfig, AgentState, ReasoningStep

__all__ = [
    "BeliefEcology",
    "ContradictionTracer",
    "GoalCollapse",
    "MetaCognition",
    "MetaEvolutionEngine",
    "HyperparameterSet",
    "EvolutionResult",
    "BeliefCoherenceObjective",
    "EfficiencyObjective",
    "MetaEvolutionDisabledError",
    "create_evolution_engine",
    "SimpleWorldModel",
    "WorldModelContext",
    "SimulationState",
    "SimulationAction",
    "WorldModelDisabledError",
    "create_world_model",
]
