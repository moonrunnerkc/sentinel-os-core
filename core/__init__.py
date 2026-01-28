# Author: Bradley R. Kinnard
# core module exports

from core.belief_ecology import BeliefEcology
from core.contradiction_tracer import ContradictionTracer
from core.goal_collapse import GoalCollapse
from core.meta_cognition import MetaCognition
from core.physics_sim import (
    simulate_causal_counterfactual,
    simulate_physics_environment,
    compute_causal_accuracy
)

__all__ = [
    "BeliefEcology",
    "ContradictionTracer",
    "GoalCollapse",
    "MetaCognition",
    "simulate_causal_counterfactual",
    "simulate_physics_environment",
    "compute_causal_accuracy"
]
