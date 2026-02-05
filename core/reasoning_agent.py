# Author: Bradley R. Kinnard
# reasoning agent - core cognitive loop with belief/goal integration

import time
import json
import uuid
import hashlib
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.belief_ecology import BeliefEcology
from core.goal_collapse import GoalCollapse
from core.contradiction_tracer import ContradictionTracer
from memory.episodic_replay import EpisodicReplay
from memory.persistent_memory import PersistentMemory
from privacy.budget import PrivacyAccountant, BudgetExhaustedError
from verification.state_machine import SystemState, BeliefState, GoalState, TransitionType, StateTransition
from verification.invariants import InvariantChecker, InvariantSeverity, InvariantViolation
from interfaces.ollama_llm import OllamaLLM, OllamaConfig, GenerationResult
from utils.helpers import get_logger, compute_hash


logger = get_logger(__name__)


# extraction prompts for structured LLM outputs
BELIEF_EXTRACTION_PROMPT = """Analyze the following user message and extract factual beliefs or propositions.
Return a JSON array of objects, each with:
- "content": the belief statement (string)
- "confidence": how confident the user seems (0.0 to 1.0)
- "source": "user_assertion" or "user_question" or "user_inference"

User message: {message}

Respond ONLY with a valid JSON array, no explanations."""

CONTRADICTION_CHECK_PROMPT = """Given these existing beliefs:
{beliefs}

And this new belief: "{new_belief}"

Does the new belief contradict any existing belief?
Return JSON: {{"contradicts": true/false, "conflicting_belief_id": "id or null", "reason": "explanation"}}"""

RESPONSE_GENERATION_PROMPT = """You are a reasoning assistant integrated with a belief and goal system.

Current beliefs:
{beliefs}

Active goals:
{goals}

Recent conversation:
{history}

User: {user_input}

Generate a helpful, coherent response that:
1. Respects the current belief state
2. Works toward active goals
3. Maintains consistency with previous statements"""

CHAIN_OF_THOUGHT_PROMPT = """Given this context, think step by step:

Beliefs: {beliefs}
Goals: {goals}
User input: {user_input}

Generate intermediate reasoning steps as JSON:
{{
    "observations": ["what you notice about the input"],
    "relevant_beliefs": ["beliefs that apply"],
    "goal_alignment": ["how this relates to goals"],
    "reasoning_chain": ["step 1", "step 2", ...],
    "conclusion": "final reasoning"
}}"""


@dataclass
class AgentConfig:
    """configuration for reasoning agent."""
    model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 512
    context_size: int = 4096
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    dp_noise_epsilon: float = 0.1
    max_history_turns: int = 10
    enable_verification: bool = True
    enable_chain_of_thought: bool = True
    data_dir: Path = field(default_factory=lambda: Path("data"))


@dataclass
class ReasoningStep:
    """single step in the reasoning process."""
    step_type: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    timestamp: float
    elapsed_ms: float


@dataclass
class AgentState:
    """serializable agent state for persistence."""
    beliefs: list[dict[str, Any]]
    goals: list[dict[str, Any]]
    episode_count: int
    total_interactions: int
    privacy_spent: float
    last_updated: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "beliefs": self.beliefs,
            "goals": self.goals,
            "episode_count": self.episode_count,
            "total_interactions": self.total_interactions,
            "privacy_spent": self.privacy_spent,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentState":
        return cls(
            beliefs=d.get("beliefs", []),
            goals=d.get("goals", []),
            episode_count=d.get("episode_count", 0),
            total_interactions=d.get("total_interactions", 0),
            privacy_spent=d.get("privacy_spent", 0.0),
            last_updated=d.get("last_updated", time.time()),
        )


class ReasoningAgent:
    """
    core reasoning agent integrating LLM with belief ecology, goals, and memory.
    implements the full cognitive loop:
    1. perception: extract beliefs from input
    2. reasoning: detect contradictions, evolve goals
    3. action: generate response aligned with state
    4. memory: record episode for replay
    """

    def __init__(self, config: AgentConfig | None = None):
        self._config = config or AgentConfig()

        # initialize components
        self._beliefs = BeliefEcology()
        self._goals = GoalCollapse()
        self._tracer = ContradictionTracer()
        self._episodes = EpisodicReplay(max_episodes=10000)
        self._memory = PersistentMemory()

        # privacy accounting
        self._accountant = PrivacyAccountant(
            total_epsilon=self._config.privacy_epsilon,
            total_delta=self._config.privacy_delta,
        )

        # llm adapter
        ollama_config = OllamaConfig(
            base_url=self._config.ollama_url,
            model=self._config.model,
            temperature=self._config.temperature,
            seed=self._config.seed,
            max_tokens=self._config.max_tokens,
            context_size=self._config.context_size,
        )
        self._llm = OllamaLLM(ollama_config)

        # conversation state
        self._history: list[dict[str, str]] = []
        self._reasoning_trace: list[ReasoningStep] = []
        self._total_interactions = 0

        # seed initial goals
        self._seed_initial_goals()

        logger.info(f"reasoning agent initialized, llm connected: {self._llm.is_connected()}")

    def _seed_initial_goals(self) -> None:
        """seed the default goals for the agent."""
        default_goals = [
            ("goal_helpful", "be helpful and answer questions accurately", 0.9),
            ("goal_truthful", "maintain truthfulness and avoid contradictions", 0.95),
            ("goal_coherent", "ensure response coherence with conversation history", 0.85),
            ("goal_safe", "avoid harmful or misleading information", 0.99),
        ]

        for goal_id, desc, priority in default_goals:
            try:
                self._goals.create_goal(goal_id, desc, priority)
            except Exception:
                pass  # goal may already exist from loaded state

    def is_ready(self) -> bool:
        """check if agent is ready for interaction."""
        return self._llm.is_connected()

    def _build_verification_state(self) -> SystemState:
        """build a SystemState from current agent state for verification."""
        beliefs = {}
        for bid, belief in self._beliefs._beliefs.items():
            beliefs[bid] = BeliefState(
                belief_id=bid,
                content_hash=compute_hash(belief["content"]),
                confidence=belief["confidence"],
                timestamp=belief.get("updated_at", time.time()),
                source=belief.get("source", "unknown"),
            )

        goals = {}
        for gid, goal in self._goals._goals.items():
            goals[gid] = GoalState(
                goal_id=gid,
                content_hash=compute_hash(goal["description"]),
                priority=goal["priority"],
                status="active",
                parent_id=goal.get("parent"),
            )

        return SystemState(
            beliefs=beliefs,
            goals=goals,
            step_counter=self._total_interactions,
            seed=self._config.seed,
        )

    def _verify_state(self) -> list[InvariantViolation]:
        """run invariant checks on current state."""
        if not self._config.enable_verification:
            return []

        state = self._build_verification_state()
        checker = InvariantChecker()
        violations = checker.check_all(state)

        for v in violations:
            if v.severity == InvariantSeverity.CRITICAL:
                logger.error(f"CRITICAL invariant violation: {v.invariant_name} - {v.message}")
            elif v.severity == InvariantSeverity.ERROR:
                logger.warning(f"invariant violation: {v.invariant_name} - {v.message}")

        return violations

    def process(self, user_input: str) -> str:
        """
        main processing loop for user input.
        returns generated response.
        """
        if not user_input.strip():
            return ""

        start = time.perf_counter()
        self._total_interactions += 1

        # pre-process verification
        if self._config.enable_verification:
            pre_violations = self._verify_state()
            if any(v.severity == InvariantSeverity.CRITICAL for v in pre_violations):
                logger.error("critical invariant violation before processing, aborting")
                return "I detected an internal consistency error. Please try again."

        # phase 1: perception - extract beliefs from input
        extracted = self._extract_beliefs(user_input)

        # phase 2: reasoning - process beliefs, detect contradictions
        contradictions = self._process_beliefs(extracted)

        # phase 3: goal evolution with DP noise
        self._evolve_goals()

        # phase 4: chain of thought (if enabled)
        reasoning = None
        if self._config.enable_chain_of_thought:
            reasoning = self._chain_of_thought(user_input)

        # phase 5: action - generate response
        response = self._generate_response(user_input, reasoning)

        # phase 6: memory - record episode
        episode = self._record_episode(user_input, response, extracted, contradictions)

        # post-process verification
        if self._config.enable_verification:
            post_violations = self._verify_state()
            if any(v.severity == InvariantSeverity.CRITICAL for v in post_violations):
                logger.error("critical invariant violation after processing")
                # could rollback here if needed

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"processed input in {elapsed:.1f}ms")

        return response

    def _extract_beliefs(self, message: str) -> list[dict[str, Any]]:
        """use LLM to extract beliefs from user message."""
        prompt = BELIEF_EXTRACTION_PROMPT.format(message=message)

        result = self._llm.generate_json(prompt, seed=self._config.seed)

        if not result or not isinstance(result, list):
            logger.debug("no beliefs extracted from message")
            return []

        beliefs = []
        for item in result:
            if isinstance(item, dict) and "content" in item:
                beliefs.append({
                    "content": item["content"],
                    "confidence": min(1.0, max(0.0, float(item.get("confidence", 0.5)))),
                    "source": item.get("source", "user_assertion"),
                })

        self._reasoning_trace.append(ReasoningStep(
            step_type="belief_extraction",
            input_data={"message": message},
            output_data={"beliefs": beliefs},
            timestamp=time.time(),
            elapsed_ms=0,
        ))

        logger.debug(f"extracted {len(beliefs)} beliefs")
        return beliefs

    def _process_beliefs(self, extracted: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """process extracted beliefs, check for contradictions."""
        contradictions = []

        for belief in extracted:
            belief_id = f"b_{uuid.uuid4().hex[:8]}"
            content = belief["content"]
            confidence = belief["confidence"]
            source = belief["source"]

            # check for contradictions with existing beliefs
            conflict = self._check_contradiction(content)

            if conflict and conflict.get("contradicts"):
                # register contradiction
                conflicting_id = conflict.get("conflicting_belief_id")
                if conflicting_id:
                    self._tracer.mark_contradictory(belief_id, conflicting_id)
                    contradictions.append({
                        "new_belief": belief_id,
                        "existing_belief": conflicting_id,
                        "reason": conflict.get("reason", ""),
                    })

                    # resolve by confidence (newer wins ties)
                    resolution = self._tracer.resolve_by_confidence(belief_id, conflicting_id)
                    logger.info(f"resolved contradiction: {resolution}")

            # add belief to ecology
            try:
                self._beliefs.create_belief(belief_id, content, confidence, source)
                self._tracer.register_belief(belief_id, content, confidence, source)
            except Exception as e:
                logger.warning(f"failed to create belief: {e}")

        return contradictions

    def _check_contradiction(self, new_belief: str) -> dict[str, Any] | None:
        """use LLM to check if new belief contradicts existing beliefs."""
        # get summary of current beliefs
        beliefs_summary = self._format_beliefs_summary()

        if not beliefs_summary:
            return None

        prompt = CONTRADICTION_CHECK_PROMPT.format(
            beliefs=beliefs_summary,
            new_belief=new_belief,
        )

        result = self._llm.generate_json(prompt, seed=self._config.seed)
        return result if isinstance(result, dict) else None

    def _evolve_goals(self) -> None:
        """evolve goals with differential privacy noise."""
        try:
            # spend privacy budget for noisy goal updates
            self._accountant.spend(
                epsilon=self._config.dp_noise_epsilon,
                mechanism="laplace",
                operation="goal_evolution",
            )

            # apply DP noise to goal priorities
            np.random.seed(self._config.seed + self._total_interactions)

            for goal_id in ["goal_helpful", "goal_truthful", "goal_coherent", "goal_safe"]:
                try:
                    goal = self._goals.get_goal(goal_id)
                    # laplace noise for DP
                    noise = np.random.laplace(0, 1 / self._config.dp_noise_epsilon)
                    # small bounded update
                    delta = noise * 0.01
                    self._goals.update_priority(goal_id, delta)
                except KeyError:
                    pass

        except BudgetExhaustedError:
            logger.warning("privacy budget exhausted, skipping goal evolution")

    def _chain_of_thought(self, user_input: str) -> dict[str, Any] | None:
        """generate chain of thought reasoning."""
        prompt = CHAIN_OF_THOUGHT_PROMPT.format(
            beliefs=self._format_beliefs_summary(),
            goals=self._format_goals_summary(),
            user_input=user_input,
        )

        result = self._llm.generate_json(prompt, seed=self._config.seed)

        if result:
            self._reasoning_trace.append(ReasoningStep(
                step_type="chain_of_thought",
                input_data={"user_input": user_input},
                output_data=result,
                timestamp=time.time(),
                elapsed_ms=0,
            ))

        return result

    def _generate_response(
        self,
        user_input: str,
        reasoning: dict[str, Any] | None
    ) -> str:
        """generate final response using LLM."""
        # build context
        system = RESPONSE_GENERATION_PROMPT.format(
            beliefs=self._format_beliefs_summary(),
            goals=self._format_goals_summary(),
            history=self._format_history(),
            user_input=user_input,
        )

        # add reasoning context if available
        if reasoning:
            conclusion = reasoning.get("conclusion", "")
            if conclusion:
                system += f"\n\nReasoning: {conclusion}"

        result = self._llm.generate(
            prompt=user_input,
            system=system,
            seed=self._config.seed,
        )

        response = result.text if result else "I encountered an error processing your request."

        # update history
        self._history.append({"role": "user", "content": user_input})
        self._history.append({"role": "assistant", "content": response})

        # trim history if needed
        max_turns = self._config.max_history_turns * 2
        if len(self._history) > max_turns:
            self._history = self._history[-max_turns:]

        return response

    def _record_episode(
        self,
        user_input: str,
        response: str,
        beliefs: list[dict[str, Any]],
        contradictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """record interaction as episode for replay."""
        episode = {
            "id": f"ep_{uuid.uuid4().hex[:12]}",
            "user_input": user_input,
            "response": response,
            "extracted_beliefs": beliefs,
            "contradictions": contradictions,
            "belief_count": self._beliefs.count(),
            "timestamp": time.time(),
            "interaction_number": self._total_interactions,
        }

        self._episodes.record_episode(episode)
        return episode

    def _format_beliefs_summary(self) -> str:
        """format current beliefs for prompt context."""
        lines = []
        for bid, belief in list(self._beliefs._beliefs.items())[:20]:  # limit to 20
            conf = belief["confidence"]
            content = belief["content"][:100]  # truncate long content
            lines.append(f"- [{bid}] ({conf:.2f}): {content}")

        return "\n".join(lines) if lines else "(no beliefs recorded)"

    def _format_goals_summary(self) -> str:
        """format active goals for prompt context."""
        lines = []
        for gid, goal in self._goals._goals.items():
            pri = goal["priority"]
            desc = goal["description"][:80]
            lines.append(f"- [{gid}] (priority {pri:.2f}): {desc}")

        return "\n".join(lines) if lines else "(no goals)"

    def _format_history(self) -> str:
        """format conversation history for context."""
        if not self._history:
            return "(no prior conversation)"

        lines = []
        for msg in self._history[-6:]:  # last 3 turns
            role = msg["role"].capitalize()
            content = msg["content"][:200]  # truncate
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def get_state(self) -> AgentState:
        """export current agent state for persistence."""
        beliefs = [
            {"id": bid, **belief}
            for bid, belief in self._beliefs._beliefs.items()
        ]
        goals = [
            {"id": gid, **goal}
            for gid, goal in self._goals._goals.items()
        ]

        return AgentState(
            beliefs=beliefs,
            goals=goals,
            episode_count=self._episodes.count(),
            total_interactions=self._total_interactions,
            privacy_spent=self._accountant.budget.spent_epsilon,
            last_updated=time.time(),
        )

    def save_state(self, path: Path | str | None = None) -> None:
        """persist agent state to disk."""
        path = path or (self._config.data_dir / "agent_state.json")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = self.get_state()
        with open(path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        # also save episodes
        self._episodes.save_to_disk(self._config.data_dir / "episodes" / "replay.json")

        logger.info(f"saved agent state to {path}")

    def load_state(self, path: Path | str | None = None) -> bool:
        """load agent state from disk."""
        path = path or (self._config.data_dir / "agent_state.json")
        path = Path(path)

        if not path.exists():
            logger.warning(f"state file not found: {path}")
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            state = AgentState.from_dict(data)

            # restore beliefs
            for belief in state.beliefs:
                bid = belief.pop("id")
                self._beliefs.create_belief(
                    bid,
                    belief["content"],
                    belief["confidence"],
                    belief["source"],
                )
                self._tracer.register_belief(
                    bid,
                    belief["content"],
                    belief["confidence"],
                    belief["source"],
                )

            # restore goals (skip defaults that already exist)
            for goal in state.goals:
                gid = goal.pop("id")
                if gid not in self._goals._goals:
                    self._goals.create_goal(
                        gid,
                        goal["description"],
                        goal["priority"],
                        goal.get("parent"),
                    )

            self._total_interactions = state.total_interactions

            # load episodes
            episodes_path = self._config.data_dir / "episodes" / "replay.json"
            if episodes_path.exists():
                self._episodes.load_from_disk(episodes_path)

            logger.info(f"loaded agent state: {len(state.beliefs)} beliefs, {self._episodes.count()} episodes")
            return True

        except Exception as e:
            logger.error(f"failed to load state: {e}")
            return False

    def get_relevant_episodes(self, query: str, n: int = 3) -> list[dict[str, Any]]:
        """retrieve relevant past episodes for context."""
        # simple keyword matching for now
        # could be enhanced with embeddings
        all_episodes = self._episodes.replay(n_episodes=100, seed=None)

        scored = []
        query_lower = query.lower()
        for ep in all_episodes:
            score = 0
            if query_lower in ep.get("user_input", "").lower():
                score += 2
            if query_lower in ep.get("response", "").lower():
                score += 1
            for belief in ep.get("extracted_beliefs", []):
                if query_lower in belief.get("content", "").lower():
                    score += 1
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:n] if _ > 0]

    def get_reasoning_trace(self) -> list[dict[str, Any]]:
        """return reasoning trace for inspection."""
        return [
            {
                "step_type": step.step_type,
                "input": step.input_data,
                "output": step.output_data,
                "timestamp": step.timestamp,
            }
            for step in self._reasoning_trace
        ]

    def clear_reasoning_trace(self) -> None:
        """clear accumulated reasoning trace."""
        self._reasoning_trace.clear()

    def get_privacy_status(self) -> dict[str, Any]:
        """return privacy budget status."""
        return self._accountant.budget.to_dict()

    def get_stats(self) -> dict[str, Any]:
        """return agent statistics."""
        return {
            "total_interactions": self._total_interactions,
            "belief_count": self._beliefs.count(),
            "goal_count": len(self._goals._goals),
            "episode_count": self._episodes.count(),
            "contradiction_count": len(self._tracer.detect_contradictions()),
            "history_length": len(self._history),
            "privacy_spent": self._accountant.budget.spent_epsilon,
            "privacy_remaining": self._accountant.budget.remaining_epsilon(),
            "llm_connected": self._llm.is_connected(),
        }
