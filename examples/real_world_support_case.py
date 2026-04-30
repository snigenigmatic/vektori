"""Real-world support handoff demo for the native Vektori harness.

This example simulates a small customer-support / account-management case,
then shows how `VektoriAgent` recalls preferences, deadlines, and open blockers.

Run with OpenAI:
    OPENAI_API_KEY=... python examples/real_world_support_case.py

Run locally with Ollama via LiteLLM:
    VEKTORI_CHAT_MODEL=litellm:ollama/your-local-model \
    VEKTORI_EMBED_MODEL=ollama:nomic-embed-text \
    VEKTORI_EXTRACT_MODEL=ollama:qwen2.5:1.5b \
    python examples/real_world_support_case.py
"""

from __future__ import annotations

import argparse
import asyncio
import os

from vektori import AgentConfig, Vektori, VektoriAgent
from vektori.models.factory import create_chat_model


SUPPORT_CASE_MESSAGES = [
    {
        "role": "user",
        "content": (
            "I'm Amina Khan from Northstar Logistics. Please email me by default; "
            "WhatsApp is only for urgent outages."
        ),
    },
    {
        "role": "assistant",
        "content": "Understood — email first, WhatsApp only for urgent issues.",
    },
    {
        "role": "user",
        "content": (
            "We need the revised invoice sent before Friday and the PO number "
            "must be included. Legal still has to approve the updated MSA."
        ),
    },
    {
        "role": "assistant",
        "content": "Got it — invoice before Friday, PO included, MSA still blocked on legal.",
    },
    {
        "role": "user",
        "content": "I'm in Dubai, so afternoons are best for meetings.",
    },
]


def _print_memories(result) -> None:
    facts = result.memories_used.get("facts", [])
    episodes = result.memories_used.get("episodes", [])

    if facts:
        print("\nRetrieved facts:")
        for fact in facts[:4]:
            score = fact.get("score")
            prefix = f"  [{score:.2f}] " if isinstance(score, (int, float)) else "  "
            print(f"{prefix}{fact.get('text', '')}")

    if episodes:
        print("\nRetrieved episodes:")
        for episode in episodes[:3]:
            print(f"  {episode.get('text', '')}")

    print(f"\nRetrieval debug: {result.retrieval_debug}")

    if result.profile_updates:
        print("\nProfile patches learned:")
        for patch in result.profile_updates:
            print(f"  - {patch.key} = {patch.value} ({patch.reason})")


async def _run_turn(agent: VektoriAgent, prompt: str) -> None:
    print(f"\nYou: {prompt}")
    result = await agent.chat(prompt)
    print(f"Assistant: {result.content}")
    _print_memories(result)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a realistic VektoriAgent support case demo.")
    parser.add_argument(
        "--chat-model",
        default=os.getenv("VEKTORI_CHAT_MODEL", "litellm:gpt-4o-mini"),
        help="Chat model for the harness (default: litellm:gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("VEKTORI_EMBED_MODEL", "openai:text-embedding-3-small"),
        help="Embedding model used by Vektori memory",
    )
    parser.add_argument(
        "--extraction-model",
        default=os.getenv("VEKTORI_EXTRACT_MODEL", "litellm:gpt-4o-mini"),
        help="Extraction model used by Vektori memory",
    )
    parser.add_argument("--user-id", default=os.getenv("VEKTORI_USER_ID", "support-demo-user"))
    parser.add_argument("--agent-id", default=os.getenv("VEKTORI_AGENT_ID", "support-demo-agent"))
    parser.add_argument(
        "--session-id",
        default=os.getenv("VEKTORI_SESSION_ID", "support-case-001"),
    )
    args = parser.parse_args()

    async with Vektori(
        embedding_model=args.embedding_model,
        extraction_model=args.extraction_model,
        async_extraction=False,
    ) as memory:
        agent = VektoriAgent(
            memory=memory,
            model=create_chat_model(args.chat_model),
            user_id=args.user_id,
            agent_id=args.agent_id,
            session_id=args.session_id,
            config=AgentConfig(
                retrieve_on_every_turn=True,
                background_add=False,
            ),
        )

        print("Seeding the support case transcript...")
        await agent.add_messages(SUPPORT_CASE_MESSAGES)

        await _run_turn(agent, "Call me Amina.")
        await _run_turn(agent, "Keep your answers short.")
        await _run_turn(
            agent,
            "What is the customer's preferred contact channel, and what is still blocked?",
        )
        await _run_turn(agent, "What should I tell them next?")

        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())