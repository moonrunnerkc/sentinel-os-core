# Author: Bradley R. Kinnard
# chatbot interface - REPL-style interaction with reasoning agent

import sys
import time
import json
import signal
from typing import Any, Callable
from pathlib import Path
from dataclasses import dataclass, field

from core.reasoning_agent import ReasoningAgent, AgentConfig
from utils.helpers import get_logger


logger = get_logger(__name__)


@dataclass
class ChatConfig:
    """configuration for chatbot interface."""
    model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 512
    context_size: int = 4096
    data_dir: Path = field(default_factory=lambda: Path("data"))
    auto_save: bool = True
    save_interval: int = 5  # save every N interactions
    show_stats: bool = False
    show_reasoning: bool = False
    privacy_epsilon: float = 1.0


class Chatbot:
    """
    REPL-style chatbot interface wrapping the reasoning agent.
    handles user interaction, state persistence, and session management.
    """

    def __init__(self, config: ChatConfig | None = None):
        self._config = config or ChatConfig()

        # build agent config from chat config
        agent_config = AgentConfig(
            model=self._config.model,
            ollama_url=self._config.ollama_url,
            temperature=self._config.temperature,
            seed=self._config.seed,
            max_tokens=self._config.max_tokens,
            context_size=self._config.context_size,
            data_dir=self._config.data_dir,
            privacy_epsilon=self._config.privacy_epsilon,
        )

        self._agent = ReasoningAgent(agent_config)
        self._session_start = time.time()
        self._interaction_count = 0
        self._running = False

        # commands registry
        self._commands: dict[str, Callable[[list[str]], str | None]] = {
            "/help": self._cmd_help,
            "/stats": self._cmd_stats,
            "/beliefs": self._cmd_beliefs,
            "/goals": self._cmd_goals,
            "/history": self._cmd_history,
            "/clear": self._cmd_clear,
            "/save": self._cmd_save,
            "/load": self._cmd_load,
            "/privacy": self._cmd_privacy,
            "/reasoning": self._cmd_reasoning,
            "/model": self._cmd_model,
            "/episodes": self._cmd_episodes,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
        }

        # try to load existing state
        self._agent.load_state()

    def _cmd_help(self, args: list[str]) -> str:
        """show available commands."""
        lines = [
            "Available commands:",
            "  /help      - show this help message",
            "  /stats     - show agent statistics",
            "  /beliefs   - list current beliefs",
            "  /goals     - list active goals",
            "  /history   - show conversation history",
            "  /clear     - clear conversation history",
            "  /save      - save agent state",
            "  /load      - load agent state",
            "  /privacy   - show privacy budget status",
            "  /reasoning - toggle reasoning trace display",
            "  /model [m] - show or set current model",
            "  /episodes  - show recent episodes",
            "  /quit      - exit the chatbot",
        ]
        return "\n".join(lines)

    def _cmd_stats(self, args: list[str]) -> str:
        """show agent statistics."""
        stats = self._agent.get_stats()
        lines = [
            f"Session duration: {(time.time() - self._session_start) / 60:.1f} minutes",
            f"Interactions: {stats['total_interactions']}",
            f"Beliefs: {stats['belief_count']}",
            f"Goals: {stats['goal_count']}",
            f"Episodes: {stats['episode_count']}",
            f"Contradictions: {stats['contradiction_count']}",
            f"History length: {stats['history_length']} messages",
            f"Privacy spent: {stats['privacy_spent']:.4f} / {self._config.privacy_epsilon}",
            f"LLM connected: {stats['llm_connected']}",
        ]
        return "\n".join(lines)

    def _cmd_beliefs(self, args: list[str]) -> str:
        """list current beliefs."""
        beliefs = self._agent._beliefs._beliefs
        if not beliefs:
            return "(no beliefs recorded)"

        lines = [f"Beliefs ({len(beliefs)} total):"]
        for bid, belief in list(beliefs.items())[:15]:
            conf = belief["confidence"]
            content = belief["content"][:60]
            source = belief["source"]
            lines.append(f"  [{bid}] ({conf:.2f}, {source}): {content}")

        if len(beliefs) > 15:
            lines.append(f"  ... and {len(beliefs) - 15} more")

        return "\n".join(lines)

    def _cmd_goals(self, args: list[str]) -> str:
        """list active goals."""
        goals = self._agent._goals._goals
        if not goals:
            return "(no goals)"

        lines = ["Active goals:"]
        for gid, goal in goals.items():
            pri = goal["priority"]
            desc = goal["description"][:50]
            lines.append(f"  [{gid}] (priority {pri:.2f}): {desc}")

        return "\n".join(lines)

    def _cmd_history(self, args: list[str]) -> str:
        """show conversation history."""
        history = self._agent._history
        if not history:
            return "(no conversation history)"

        lines = ["Recent conversation:"]
        for msg in history[-10:]:
            role = msg["role"].capitalize()
            content = msg["content"][:100]
            if len(msg["content"]) > 100:
                content += "..."
            lines.append(f"  {role}: {content}")

        return "\n".join(lines)

    def _cmd_clear(self, args: list[str]) -> str:
        """clear conversation history."""
        self._agent._history.clear()
        self._agent.clear_reasoning_trace()
        return "Conversation history cleared."

    def _cmd_save(self, args: list[str]) -> str:
        """save agent state."""
        path = args[0] if args else None
        self._agent.save_state(path)
        return f"State saved to {path or 'default location'}."

    def _cmd_load(self, args: list[str]) -> str:
        """load agent state."""
        path = args[0] if args else None
        success = self._agent.load_state(path)
        if success:
            return f"State loaded from {path or 'default location'}."
        return "Failed to load state."

    def _cmd_privacy(self, args: list[str]) -> str:
        """show privacy budget status."""
        status = self._agent.get_privacy_status()
        lines = [
            "Privacy budget status:",
            f"  Total epsilon: {status['total_epsilon']}",
            f"  Spent epsilon: {status['spent_epsilon']:.4f}",
            f"  Remaining: {status['remaining_epsilon']:.4f}",
            f"  Query count: {status['query_count']}",
            f"  Exhausted: {status['exhausted']}",
        ]
        return "\n".join(lines)

    def _cmd_reasoning(self, args: list[str]) -> str:
        """toggle or show reasoning trace."""
        self._config.show_reasoning = not self._config.show_reasoning
        status = "enabled" if self._config.show_reasoning else "disabled"
        return f"Reasoning trace display {status}."

    def _cmd_model(self, args: list[str]) -> str:
        """show or set current model."""
        if args:
            model = args[0]
            if self._agent._llm.set_model(model):
                return f"Model switched to: {model}"
            return f"Model not found: {model}"

        available = self._agent._llm.list_models()
        current = self._agent._llm._config.model
        lines = [f"Current model: {current}", "Available models:"]
        for m in available:
            lines.append(f"  - {m}")
        return "\n".join(lines)

    def _cmd_episodes(self, args: list[str]) -> str:
        """show recent episodes."""
        n = int(args[0]) if args else 5
        episodes = self._agent._episodes.replay(n_episodes=n, seed=None)

        if not episodes:
            return "(no episodes recorded)"

        lines = [f"Recent episodes ({len(episodes)} shown):"]
        for ep in episodes[:n]:
            eid = ep.get("id", "?")[:12]
            user = ep.get("user_input", "")[:40]
            resp = ep.get("response", "")[:40]
            lines.append(f"  [{eid}] User: {user}...")
            lines.append(f"           Bot: {resp}...")

        return "\n".join(lines)

    def _cmd_quit(self, args: list[str]) -> str | None:
        """exit the chatbot."""
        self._running = False
        return None

    def _process_command(self, input_text: str) -> tuple[bool, str | None]:
        """process a command input. returns (is_command, response)."""
        parts = input_text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].split() if len(parts) > 1 else []

        if cmd in self._commands:
            result = self._commands[cmd](args)
            return True, result

        return False, None

    def _format_reasoning_trace(self) -> str:
        """format the reasoning trace for display."""
        trace = self._agent.get_reasoning_trace()
        if not trace:
            return ""

        lines = ["\n--- Reasoning Trace ---"]
        for step in trace[-5:]:  # last 5 steps
            lines.append(f"  [{step['step_type']}]")
            if step['step_type'] == 'belief_extraction':
                beliefs = step['output'].get('beliefs', [])
                for b in beliefs[:3]:
                    lines.append(f"    - {b.get('content', '')[:50]}...")
            elif step['step_type'] == 'chain_of_thought':
                conclusion = step['output'].get('conclusion', '')
                if conclusion:
                    lines.append(f"    Conclusion: {conclusion[:80]}...")
        lines.append("--- End Trace ---\n")

        return "\n".join(lines)

    def run(self) -> None:
        """run the REPL interaction loop."""
        self._running = True
        self._session_start = time.time()

        # handle graceful shutdown
        def signal_handler(sig, frame):
            print("\n\nInterrupted. Saving state...")
            if self._config.auto_save:
                self._agent.save_state()
            self._running = False
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # print welcome
        print("\n" + "=" * 50)
        print("  Sentinel OS Chatbot")
        print("  Type /help for commands, /quit to exit")
        print("=" * 50 + "\n")

        if not self._agent.is_ready():
            print("[WARNING] LLM not connected. Check Ollama status.")
            print("  Run: ollama serve")
            print(f"  Model: {self._config.model}\n")

        while self._running:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # check for commands
                is_command, response = self._process_command(user_input)
                if is_command:
                    if response:
                        print(f"\n{response}\n")
                    continue

                # process through agent
                start = time.perf_counter()
                response = self._agent.process(user_input)
                elapsed = (time.perf_counter() - start) * 1000

                # display response
                print(f"\nBot: {response}")

                if self._config.show_stats:
                    print(f"  [{elapsed:.0f}ms]")

                if self._config.show_reasoning:
                    print(self._format_reasoning_trace())

                print()

                # auto-save periodically
                self._interaction_count += 1
                if self._config.auto_save and self._interaction_count % self._config.save_interval == 0:
                    self._agent.save_state()
                    logger.debug("auto-saved state")

            except EOFError:
                print("\n")
                break
            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                logger.error(f"error processing input: {e}")
                print(f"\n[ERROR] {e}\n")

        # final save on exit
        if self._config.auto_save:
            print("Saving state...")
            self._agent.save_state()

        print("Goodbye!")


def run_chatbot(
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
    **kwargs
) -> None:
    """convenience function to run chatbot with custom settings."""
    config = ChatConfig(
        model=model,
        ollama_url=ollama_url,
        **{k: v for k, v in kwargs.items() if hasattr(ChatConfig, k)}
    )
    chatbot = Chatbot(config)
    chatbot.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentinel OS Chatbot")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per response")
    parser.add_argument("--show-stats", action="store_true", help="Show timing stats")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning trace")

    args = parser.parse_args()

    config = ChatConfig(
        model=args.model,
        ollama_url=args.url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        show_stats=args.show_stats,
        show_reasoning=args.show_reasoning,
    )

    chatbot = Chatbot(config)
    chatbot.run()
