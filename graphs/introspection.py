# Author: Bradley R. Kinnard
# introspection graph - belief/goal network visualization

import json
from pathlib import Path
from typing import Any

import aiofiles

from utils.helpers import get_logger


logger = get_logger(__name__)


class IntrospectionGraph:
    """
    graph representation of beliefs and goals for visualization.
    supports async export and optional neuromorphic mode.
    """

    def __init__(self, neuromorphic_mode: bool = False):
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._neuromorphic = neuromorphic_mode
        self._snn = None

        if neuromorphic_mode:
            self._init_snn()

    def _init_snn(self) -> None:
        """initialize brian2 SNN for neuromorphic processing."""
        try:
            from brian2 import NeuronGroup, Synapses, ms  # noqa: F401
            self._snn = {"NeuronGroup": NeuronGroup, "Synapses": Synapses, "ms": ms}
            logger.info("brian2 SNN initialized")
        except ImportError:
            logger.warning("brian2 not available, neuromorphic mode disabled")
            self._neuromorphic = False

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        properties: dict[str, Any] | None = None
    ) -> None:
        """add a node to the graph."""
        self._nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "label": label,
            "properties": properties or {}
        }
        logger.debug(f"added node {node_id}")

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        weight: float = 1.0
    ) -> None:
        """add an edge between nodes."""
        if source not in self._nodes:
            raise KeyError(f"source node not found: {source}")
        if target not in self._nodes:
            raise KeyError(f"target node not found: {target}")

        self._edges.append({
            "source": source,
            "target": target,
            "type": edge_type,
            "weight": weight
        })
        logger.debug(f"added edge {source} -> {target}")

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """retrieve a node by id."""
        return self._nodes.get(node_id)

    def get_edges(self, node_id: str) -> list[dict[str, Any]]:
        """get all edges connected to a node."""
        return [
            e for e in self._edges
            if e["source"] == node_id or e["target"] == node_id
        ]

    def node_count(self) -> int:
        """return number of nodes."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """return number of edges."""
        return len(self._edges)

    def to_dict(self) -> dict[str, Any]:
        """export graph as dictionary."""
        return {
            "nodes": list(self._nodes.values()),
            "edges": self._edges,
            "metadata": {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "neuromorphic": self._neuromorphic
            }
        }

    def to_json(self) -> str:
        """export graph as JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_to_disk(self, path: Path | str) -> None:
        """save graph to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"saved graph to {path}")

    async def async_save_to_disk(self, path: Path | str) -> None:
        """async save graph to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = self.to_json()
        async with aiofiles.open(path, "w") as f:
            await f.write(content)

        logger.info(f"async saved graph to {path}")

    def load_from_disk(self, path: Path | str) -> None:
        """load graph from disk."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        self._nodes = {n["id"]: n for n in data.get("nodes", [])}
        self._edges = data.get("edges", [])

        logger.info(f"loaded graph from {path}")

    def from_beliefs(self, beliefs: list[dict[str, Any]]) -> None:
        """build graph from belief list."""
        for belief in beliefs:
            self.add_node(
                node_id=belief["id"],
                node_type="belief",
                label=belief.get("content", "")[:50],
                properties={"confidence": belief.get("confidence", 0.5)}
            )

    def from_goals(self, goals: list[dict[str, Any]]) -> None:
        """build graph from goal list."""
        for goal in goals:
            self.add_node(
                node_id=goal["id"],
                node_type="goal",
                label=goal.get("description", "")[:50],
                properties={"priority": goal.get("priority", 0.5)}
            )

            # add parent edge if exists
            if goal.get("parent"):
                try:
                    self.add_edge(goal["parent"], goal["id"], "parent-child")
                except KeyError:
                    pass  # parent not added yet

    def compute_energy(self) -> float:
        """
        compute energy consumption estimate (mock).
        for neuromorphic mode, returns estimated mJ per cycle.
        """
        if not self._neuromorphic:
            return 0.0

        # mock energy estimate based on graph size
        # real SNN would compute this from spike rates
        base_energy = 0.001  # 1 uJ base
        node_energy = len(self._nodes) * 0.0001  # 0.1 uJ per node
        edge_energy = len(self._edges) * 0.00005  # 0.05 uJ per edge

        total_mj = (base_energy + node_energy + edge_energy) * 1000
        logger.debug(f"estimated energy: {total_mj:.4f} mJ")
        return total_mj
