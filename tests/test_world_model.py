# Author: Bradley R. Kinnard
# tests for world model

import pytest
import numpy as np

from core.world_model import (
    WorldModelType,
    SimulationState,
    SimulationAction,
    SimulationResult,
    WorldModelInterface,
    SimpleWorldModel,
    DisabledWorldModel,
    WorldModelDisabledError,
    WorldModelContext,
    create_world_model,
)


class TestSimulationState:
    """test simulation state."""

    def test_state_creation(self):
        state = SimulationState(
            beliefs={"b1": 0.8, "b2": 0.6},
            goals={"g1": 0.9},
            resources={"r1": 0.5},
        )
        assert state.beliefs["b1"] == 0.8
        assert state.step == 0

    def test_state_to_dict(self):
        state = SimulationState(
            beliefs={"b1": 0.8},
            goals={"g1": 0.9},
            resources={"r1": 0.5},
        )
        d = state.to_dict()
        assert "beliefs" in d
        assert "goals" in d
        assert "resources" in d

    def test_state_digest_deterministic(self):
        state = SimulationState(
            beliefs={"b1": 0.8},
            goals={"g1": 0.9},
            resources={"r1": 0.5},
        )
        d1 = state.digest()
        d2 = state.digest()
        assert d1 == d2

    def test_state_digest_different_states(self):
        state1 = SimulationState(beliefs={"b1": 0.8}, goals={}, resources={})
        state2 = SimulationState(beliefs={"b1": 0.9}, goals={}, resources={})
        assert state1.digest() != state2.digest()

    def test_state_copy_independence(self):
        state = SimulationState(
            beliefs={"b1": 0.8},
            goals={"g1": 0.9},
            resources={"r1": 0.5},
        )
        copy = state.copy()
        copy.beliefs["b1"] = 0.1
        assert state.beliefs["b1"] == 0.8


class TestSimpleWorldModel:
    """test simple world model."""

    def test_model_type(self):
        model = SimpleWorldModel()
        assert model.model_type == WorldModelType.SIMPLE

    def test_simulate_empty_actions(self):
        model = SimpleWorldModel()
        state = SimulationState(
            beliefs={"b1": 0.8},
            goals={"g1": 0.9},
            resources={"r1": 0.5},
        )
        result = model.simulate(state, [], seed=42)
        assert result.steps == 0
        assert result.seed == 42
        assert result.deterministic

    def test_simulate_update_belief(self):
        model = SimpleWorldModel()
        state = SimulationState(
            beliefs={"b1": 0.5},
            goals={},
            resources={},
        )
        action = SimulationAction(
            action_type="update_belief",
            target_id="b1",
            delta=0.2,
        )
        result = model.simulate(state, [action], seed=42)
        # belief should increase (minus decay)
        assert result.final_state.beliefs["b1"] > 0.5

    def test_simulate_consume_resource(self):
        model = SimpleWorldModel()
        state = SimulationState(
            beliefs={},
            goals={},
            resources={"r1": 0.8},
        )
        action = SimulationAction(
            action_type="consume_resource",
            target_id="r1",
            delta=0.3,
        )
        result = model.simulate(state, [action], seed=42)
        assert result.final_state.resources["r1"] < 0.8
        assert result.reward > 0

    def test_simulate_determinism(self):
        model = SimpleWorldModel()
        state = SimulationState(
            beliefs={"b1": 0.5, "b2": 0.6},
            goals={"g1": 0.7},
            resources={"r1": 0.8},
        )
        actions = [
            SimulationAction("update_belief", "b1", 0.1),
            SimulationAction("consume_resource", "r1", 0.2),
        ]

        result1 = model.simulate(state, actions, seed=42)
        result2 = model.simulate(state, actions, seed=42)

        assert result1.final_digest == result2.final_digest
        assert result1.reward == result2.reward

    def test_simulate_different_seeds(self):
        model = SimpleWorldModel()
        state = SimulationState(
            beliefs={"b1": 0.5},
            goals={},
            resources={},
        )
        actions = [SimulationAction("update_belief", "b1", 0.1)]

        result1 = model.simulate(state, actions, seed=42)
        result2 = model.simulate(state, actions, seed=123)

        # different seeds may produce different results due to noise
        # but both should be valid
        assert result1.deterministic
        assert result2.deterministic

    def test_counterfactual(self):
        model = SimpleWorldModel()
        state = SimulationState(
            beliefs={"b1": 0.5},
            goals={},
            resources={},
        )
        action = SimulationAction("update_belief", "b1", 0.2)

        new_state, reward = model.counterfactual(state, action, seed=42)
        assert new_state.beliefs["b1"] != state.beliefs["b1"]
        assert isinstance(reward, float)

    def test_counterfactual_determinism(self):
        model = SimpleWorldModel()
        state = SimulationState(
            beliefs={"b1": 0.5},
            goals={},
            resources={},
        )
        action = SimulationAction("update_belief", "b1", 0.2)

        s1, r1 = model.counterfactual(state, action, seed=42)
        s2, r2 = model.counterfactual(state, action, seed=42)

        assert s1.digest() == s2.digest()
        assert r1 == r2


class TestDisabledWorldModel:
    """test disabled world model."""

    def test_model_type(self):
        model = DisabledWorldModel()
        assert model.model_type == WorldModelType.NONE

    def test_simulate_raises(self):
        model = DisabledWorldModel()
        state = SimulationState(beliefs={}, goals={}, resources={})

        with pytest.raises(WorldModelDisabledError):
            model.simulate(state, [], seed=42)

    def test_counterfactual_raises(self):
        model = DisabledWorldModel()
        state = SimulationState(beliefs={}, goals={}, resources={})
        action = SimulationAction("update_belief", "b1", 0.1)

        with pytest.raises(WorldModelDisabledError):
            model.counterfactual(state, action, seed=42)


class TestCreateWorldModel:
    """test factory function."""

    def test_disabled_returns_stub(self):
        config = {"world_model": {"enabled": False}}
        model = create_world_model(config)
        assert isinstance(model, DisabledWorldModel)

    def test_enabled_simple(self):
        config = {
            "world_model": {
                "enabled": True,
                "type": "simple",
            }
        }
        model = create_world_model(config)
        assert isinstance(model, SimpleWorldModel)

    def test_type_none(self):
        config = {
            "world_model": {
                "enabled": True,
                "type": "none",
            }
        }
        model = create_world_model(config)
        assert isinstance(model, DisabledWorldModel)

    def test_unknown_type_raises(self):
        config = {
            "world_model": {
                "enabled": True,
                "type": "neural",  # not supported
            }
        }
        with pytest.raises(ValueError, match="unknown world model type"):
            create_world_model(config)


class TestWorldModelContext:
    """test world model context."""

    def test_simulate_with_audit(self):
        model = SimpleWorldModel()
        ctx = WorldModelContext(model, base_seed=42)

        state = SimulationState(
            beliefs={"b1": 0.5},
            goals={},
            resources={},
        )
        actions = [SimulationAction("update_belief", "b1", 0.1)]

        result = ctx.simulate_with_audit(state, actions)
        assert result.steps == 1

        history = ctx.get_audit_history()
        assert len(history) == 1
        assert history[0]["seed"] == 42

    def test_audit_history_grows(self):
        model = SimpleWorldModel()
        ctx = WorldModelContext(model, base_seed=42)

        state = SimulationState(beliefs={}, goals={}, resources={})

        ctx.simulate_with_audit(state, [])
        ctx.simulate_with_audit(state, [])
        ctx.simulate_with_audit(state, [])

        history = ctx.get_audit_history()
        assert len(history) == 3
        # seeds should increment
        assert history[0]["seed"] == 42
        assert history[1]["seed"] == 43
        assert history[2]["seed"] == 44

    def test_counterfactual_branches(self):
        model = SimpleWorldModel()
        ctx = WorldModelContext(model, base_seed=42)

        state = SimulationState(
            beliefs={"b1": 0.5, "b2": 0.6},
            goals={},
            resources={},
        )
        actions = [
            SimulationAction("update_belief", "b1", 0.1),
            SimulationAction("update_belief", "b2", 0.2),
        ]

        branches = ctx.counterfactual_branches(state, actions)
        assert len(branches) == 2
        for action, new_state, reward in branches:
            assert isinstance(new_state, SimulationState)
            assert isinstance(reward, float)

    def test_verify_determinism(self):
        model = SimpleWorldModel()
        ctx = WorldModelContext(model, base_seed=42)

        state = SimulationState(
            beliefs={"b1": 0.5},
            goals={},
            resources={},
        )
        actions = [SimulationAction("update_belief", "b1", 0.1)]

        # first, get the expected digest
        result = model.simulate(state, actions, seed=100)
        expected = result.final_digest

        # verify
        valid, msg = ctx.verify_determinism(state, actions, seed=100, expected_digest=expected)
        assert valid, msg

    def test_verify_determinism_fails(self):
        model = SimpleWorldModel()
        ctx = WorldModelContext(model, base_seed=42)

        state = SimulationState(
            beliefs={"b1": 0.5},
            goals={},
            resources={},
        )
        actions = [SimulationAction("update_belief", "b1", 0.1)]

        valid, msg = ctx.verify_determinism(state, actions, seed=100, expected_digest="wrong")
        assert not valid
        assert "mismatch" in msg


class TestFallbackBehavior:
    """test fallback when world model is disabled."""

    def test_system_operates_without_world_model(self):
        # create disabled model
        config = {"world_model": {"enabled": False}}
        model = create_world_model(config)

        # should create DisabledWorldModel
        assert model.model_type == WorldModelType.NONE

        # calling simulate should fail explicitly
        state = SimulationState(beliefs={}, goals={}, resources={})
        with pytest.raises(WorldModelDisabledError):
            model.simulate(state, [], seed=42)

    def test_hard_fail_semantics(self):
        """world model should hard-fail, not silently succeed."""
        model = DisabledWorldModel()
        state = SimulationState(beliefs={}, goals={}, resources={})

        # must raise, not return None or empty result
        with pytest.raises(WorldModelDisabledError) as exc_info:
            model.simulate(state, [], seed=42)

        assert "disabled" in str(exc_info.value).lower()
