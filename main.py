# Author: Bradley R. Kinnard
# sentinel-os-core main entry point

import argparse
from pathlib import Path

from utils.helpers import load_system_config, load_security_rules, get_logger
from core import BeliefEcology, ContradictionTracer, GoalCollapse, MetaCognition
from memory import PersistentMemory, EpisodicReplay
from security import Sandbox, AuditLogger
from interfaces import LocalLLM, InputLayer, OutputLayer, FederatedSync
from graphs import IntrospectionGraph


logger = get_logger(__name__)


class SentinelOS:
    """
    sentinel-os-core: offline-first cognitive operating system.
    manages beliefs, goals, memory, and introspection.
    """

    def __init__(self, config_path: str = "config/system_config.yaml"):
        self._config = load_system_config(config_path)
        self._security = load_security_rules()

        # core components
        self._beliefs = BeliefEcology()
        self._contradictions = ContradictionTracer()
        self._goals = GoalCollapse()
        self._meta = MetaCognition(seed=self._config["llm"]["seed"])

        # memory
        self._memory = PersistentMemory(
            enable_he=self._security.get("use_homomorphic_enc", False)
        )
        self._episodes = EpisodicReplay(
            max_episodes=self._config["performance"]["max_episodes"]
        )

        # security
        self._sandbox = Sandbox(config=self._security)
        self._audit = AuditLogger(
            master_seed=self._security["hmac_key_seed"],
            pq_crypto=self._security.get("pq_crypto", False)
        )

        # interfaces
        self._llm = LocalLLM(
            model_path=self._config["llm"]["model_path"],
            backend=self._config["llm"]["backend"],
            seed=self._config["llm"]["seed"]
        )
        self._input = InputLayer()
        self._output = OutputLayer()
        self._sync = FederatedSync(
            enabled=self._security.get("enable_federated_sync", False)
        )

        # visualization
        self._graph = IntrospectionGraph(
            neuromorphic_mode=self._config["features"].get("neuromorphic_mode", False)
        )

        self._running = False
        logger.info("sentinel-os initialized")
        self._audit.log("system_init", "INFO", {"config": config_path})

    def start(self) -> None:
        """start the system."""
        self._running = True
        self._audit.log("system_start", "INFO")
        logger.info("sentinel-os started")

    def stop(self) -> None:
        """stop the system and save state."""
        self._running = False
        self._audit.log("system_stop", "INFO")
        self._save_state()
        logger.info("sentinel-os stopped")

    def _save_state(self) -> None:
        """save current state to disk."""
        data_dir = Path("data")

        # save beliefs
        beliefs = self._beliefs._beliefs
        self._memory._beliefs = beliefs
        self._memory.save_to_disk(data_dir / "beliefs" / "state.json")

        # save episodes
        self._episodes.save_to_disk(data_dir / "episodes" / "replay.json")

        # save audit log
        self._audit.save_to_disk()

        # save introspection graph
        self._update_graph()
        self._graph.save_to_disk(data_dir / "graphs" / "introspection.json")

        logger.info("state saved to disk")

    def _update_graph(self) -> None:
        """update introspection graph from current state."""
        self._graph = IntrospectionGraph()
        self._graph.from_beliefs(list(self._beliefs._beliefs.values()))
        self._graph.from_goals(list(self._goals._goals.values()))

    def process_input(self, input_data: dict) -> dict:
        """process input through the system."""
        # validate and sanitize
        result = self._input.process(input_data)
        if not result["valid"] or not result["safe"]:
            self._audit.log("input_rejected", "WARNING", {"errors": result["errors"]})
            return self._output.format_error(result["errors"][0], "INVALID_INPUT")

        data = result["data"]
        self._audit.log("input_accepted", "INFO", {"type": data["type"]})

        # route by type
        if data["type"] == "belief":
            return self._handle_belief(data)
        elif data["type"] == "goal":
            return self._handle_goal(data)
        elif data["type"] == "query":
            return self._handle_query(data)
        else:
            return self._output.format_error("unknown input type", "UNKNOWN_TYPE")

    def _handle_belief(self, data: dict) -> dict:
        """handle belief input."""
        belief = self._beliefs.create_belief(
            belief_id=data.get("metadata", {}).get("id", f"b_{len(self._beliefs._beliefs)}"),
            content=data["content"],
            confidence=data.get("priority", 0.5),
            source="input"
        )
        self._audit.log("belief_created", "INFO", {"id": belief["id"]})
        return self._output.format_belief(belief)

    def _handle_goal(self, data: dict) -> dict:
        """handle goal input."""
        goal = self._goals.create_goal(
            goal_id=data.get("metadata", {}).get("id", f"g_{len(self._goals._goals)}"),
            description=data["content"],
            priority=data.get("priority", 0.5)
        )
        self._audit.log("goal_created", "INFO", {"id": goal["id"]})
        return self._output.format_goal(goal)

    def _handle_query(self, data: dict) -> dict:
        """handle query input."""
        if not self._llm.is_model_loaded():
            return self._output.format_response(
                "llm not available for queries",
                response_type="text"
            )

        response = self._llm.generate_sync(data["content"], max_tokens=256)
        self._audit.log("query_processed", "INFO")
        return self._output.format_response(response, response_type="text")

    def get_status(self) -> dict:
        """get system status."""
        return {
            "running": self._running,
            "beliefs": self._beliefs.count(),
            "goals": len(self._goals._goals),
            "episodes": self._episodes.count(),
            "llm_loaded": self._llm.is_model_loaded()
        }


def main():
    parser = argparse.ArgumentParser(description="Sentinel OS Core")
    parser.add_argument("--config", default="config/system_config.yaml", help="config path")
    args = parser.parse_args()

    os_instance = SentinelOS(config_path=args.config)
    os_instance.start()

    logger.info("sentinel-os running. press Ctrl+C to stop.")

    try:
        # simple event loop
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("shutdown requested")
    finally:
        os_instance.stop()


if __name__ == "__main__":
    main()
