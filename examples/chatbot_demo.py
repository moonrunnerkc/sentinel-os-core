#!/usr/bin/env python3
# Author: Bradley R. Kinnard
# chatbot demo - interactive demonstration of sentinel reasoning agent

"""
Sentinel OS Chatbot Demo

This script provides an interactive demonstration of the reasoning agent
with belief extraction, contradiction detection, and goal-directed behavior.

Prerequisites:
    - Ollama running locally (ollama serve)
    - A model installed (e.g., ollama pull llama3.2)

Usage:
    python examples/chatbot_demo.py [--model MODEL] [--verbose]

Example:
    python examples/chatbot_demo.py --model llama3.2 --verbose
"""

import sys
import argparse
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from interfaces.chatbot import Chatbot, ChatConfig
from interfaces.ollama_llm import OllamaLLM, OllamaConfig
from utils.helpers import get_logger


logger = get_logger(__name__)


def check_ollama_status(url: str = "http://localhost:11434") -> bool:
    """verify ollama is running and accessible."""
    config = OllamaConfig(base_url=url)
    llm = OllamaLLM(config)
    return llm.is_connected()


def list_available_models(url: str = "http://localhost:11434") -> list[str]:
    """list models available in ollama."""
    config = OllamaConfig(base_url=url)
    llm = OllamaLLM(config)
    return llm.list_models()


def run_demo(
    model: str = "llama3.2",
    verbose: bool = False,
    show_reasoning: bool = False,
) -> None:
    """run the chatbot demo."""

    print("\n" + "=" * 60)
    print("  Sentinel OS Chatbot Demo")
    print("=" * 60)

    # check ollama status
    print("\nChecking Ollama status...")
    if not check_ollama_status():
        print("\n[ERROR] Ollama is not running!")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull a model: ollama pull llama3.2")
        print("  3. Run this demo again")
        sys.exit(1)

    print("  ✓ Ollama is running")

    # list available models
    models = list_available_models()
    print(f"  ✓ {len(models)} model(s) available")

    if model not in " ".join(models):
        print(f"\n[WARNING] Model '{model}' may not be installed")
        print(f"  Available models: {', '.join(models[:5])}")
        print(f"  To install: ollama pull {model}")

    # create chatbot config
    config = ChatConfig(
        model=model,
        ollama_url="http://localhost:11434",
        temperature=0.0,
        seed=42,
        max_tokens=512,
        context_size=4096,
        data_dir=Path("data"),
        auto_save=True,
        save_interval=3,
        show_stats=verbose,
        show_reasoning=show_reasoning,
        privacy_epsilon=1.0,
    )

    # create and run chatbot
    print(f"\nInitializing chatbot with model: {model}")
    chatbot = Chatbot(config)

    print("\nChatbot Features:")
    print("  • Belief extraction from user messages")
    print("  • Contradiction detection and resolution")
    print("  • Goal-directed response generation")
    print("  • Chain-of-thought reasoning")
    print("  • Differential privacy on goal updates")
    print("  • Episodic memory for context")

    print("\nCommands:")
    print("  /help     - show all commands")
    print("  /beliefs  - show extracted beliefs")
    print("  /goals    - show active goals")
    print("  /stats    - show agent statistics")
    print("  /quit     - exit the chatbot")

    print("\n" + "-" * 60)
    print("  Start chatting! (Type /quit to exit)")
    print("-" * 60 + "\n")

    # run the interaction loop
    chatbot.run()


def run_quick_test() -> None:
    """run a quick non-interactive test of the agent."""
    print("\n" + "=" * 60)
    print("  Quick Test Mode")
    print("=" * 60)

    from core.reasoning_agent import ReasoningAgent, AgentConfig

    # check ollama
    if not check_ollama_status():
        print("\n[SKIP] Ollama not running, testing offline components only")

        config = AgentConfig(
            model="llama3.2",
            ollama_url="http://localhost:11434",
            enable_verification=True,
            enable_chain_of_thought=False,
        )

        print("\nCreating agent...")
        from unittest.mock import patch
        with patch.object(OllamaLLM, "_check_connection"):
            agent = ReasoningAgent(config)

        print("  ✓ Agent created")
        print(f"  ✓ {len(agent._goals._goals)} goals seeded")

        # test belief creation
        agent._beliefs.create_belief("test_b1", "The sky is blue", 0.9, "test")
        agent._tracer.register_belief("test_b1", "The sky is blue", 0.9, "test")
        print("  ✓ Belief creation works")

        # test goal evolution
        agent._evolve_goals()
        print("  ✓ Goal evolution works")

        # test verification
        state = agent._build_verification_state()
        print(f"  ✓ Verification state built: {len(state.beliefs)} beliefs")

        # test stats
        stats = agent.get_stats()
        print(f"  ✓ Stats: {stats['goal_count']} goals, {stats['belief_count']} beliefs")

        print("\n[OK] Offline components working correctly")
        return

    # full test with ollama
    print("\nRunning full test with Ollama...")

    config = AgentConfig(
        model="llama3.2",
        ollama_url="http://localhost:11434",
        temperature=0.0,
        seed=42,
        max_tokens=256,
        enable_verification=True,
        enable_chain_of_thought=True,
    )

    print("\nCreating agent...")
    agent = ReasoningAgent(config)
    print(f"  ✓ Agent ready: {agent.is_ready()}")

    # test a simple interaction
    print("\nTesting interaction...")
    test_input = "I believe the Earth is round and orbits the Sun."
    print(f"  Input: {test_input}")

    response = agent.process(test_input)
    print(f"  Response: {response[:100]}...")

    # show stats
    stats = agent.get_stats()
    print(f"\n  Stats after interaction:")
    print(f"    Beliefs: {stats['belief_count']}")
    print(f"    Episodes: {stats['episode_count']}")
    print(f"    Privacy spent: {stats['privacy_spent']:.4f}")

    print("\n[OK] Full test completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentinel OS Chatbot Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chatbot_demo.py                    # Run with defaults
  python chatbot_demo.py --model mistral    # Use mistral model
  python chatbot_demo.py --verbose          # Show timing stats
  python chatbot_demo.py --reasoning        # Show reasoning trace
  python chatbot_demo.py --test             # Run quick test mode
        """
    )

    parser.add_argument(
        "--model", "-m",
        default="llama3.2",
        help="Ollama model name (default: llama3.2)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show timing and stats after each response"
    )
    parser.add_argument(
        "--reasoning", "-r",
        action="store_true",
        help="Show reasoning trace after each response"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Run quick test mode instead of interactive chat"
    )

    args = parser.parse_args()

    if args.test:
        run_quick_test()
    else:
        run_demo(
            model=args.model,
            verbose=args.verbose,
            show_reasoning=args.reasoning,
        )
