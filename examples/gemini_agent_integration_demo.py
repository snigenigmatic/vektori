"""
Gemini-powered VektoriAgent demo for integration showcases.

This demo keeps embeddings local and uses Gemini for extraction + chat:
  - extraction_model: gemini:gemini-2.5-flash-lite (direct Gemini provider)
  - chat model:       litellm:gemini/gemini-2.5-flash (agent responses)

Setup:
    export GOOGLE_API_KEY="your-gemini-api-key"
    pip install "vektori[litellm,sentence-transformers]" google-generativeai
    python examples/gemini_agent_integration_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running this file directly from a source checkout:
#   python3 examples/gemini_agent_integration_demo.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vektori import AgentConfig, Vektori, VektoriAgent
from vektori.models.factory import create_chat_model

USER_ID = os.getenv("VEKTORI_DEMO_USER_ID", "demo-gemini-user")
CHAT_MODEL = os.getenv("VEKTORI_CHAT_MODEL", "litellm:gemini/gemini-2.5-flash")
EXTRACTION_MODEL = os.getenv("VEKTORI_EXTRACT_MODEL", "gemini:gemini-2.5-flash-lite")
EMBEDDING_MODEL = os.getenv(
    "VEKTORI_EMBED_MODEL",
    "sentence-transformers:all-MiniLM-L6-v2",
)


async def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit(
            "GOOGLE_API_KEY is not set. Export it first, then run this demo again."
        )

    memory = Vektori(
        embedding_model=EMBEDDING_MODEL,
        extraction_model=EXTRACTION_MODEL,
        async_extraction=False,
    )
    agent = VektoriAgent(
        memory=memory,
        model=create_chat_model(CHAT_MODEL),
        user_id=USER_ID,
        agent_id="gemini-demo-agent",
        session_id="gemini-demo-session-001",
        config=AgentConfig(
            retrieve_on_every_turn=True,
            background_add=False,
        ),
    )

    print("=== Gemini Agent Integration Demo ===")
    print(f"user_id: {USER_ID}")
    print(f"chat_model: {CHAT_MODEL}")
    print(f"extraction_model: {EXTRACTION_MODEL}")
    print(f"embedding_model: {EMBEDDING_MODEL}\n")

    try:
        # Seed two prior sessions so retrieval has meaningful history.
        await memory.add(
            messages=[
                {"role": "user", "content": "I prefer concise updates with bullet points."},
                {"role": "assistant", "content": "Got it, concise bullet updates."},
                {
                    "role": "user",
                    "content": "Our demo target is support teams handling many customer follow-ups.",
                },
            ],
            session_id="seed-session-001",
            user_id=USER_ID,
            agent_id="gemini-demo-agent",
        )
        await memory.add(
            messages=[
                {
                    "role": "user",
                    "content": "Please avoid sending me long narratives in responses.",
                },
                {
                    "role": "assistant",
                    "content": "Understood. I'll keep replies short and action-oriented.",
                },
            ],
            session_id="seed-session-002",
            user_id=USER_ID,
            agent_id="gemini-demo-agent",
        )

        prompts = [
            "What do you remember about my response style?",
            "What's the product use case we are focusing on for this demo?",
            "Now give me a launch checklist in my preferred style.",
        ]

        for idx, prompt in enumerate(prompts, start=1):
            print(f"Turn {idx}")
            print(f"You: {prompt}")
            result = await agent.chat(prompt)
            print(f"Assistant: {result.content}")

            counts = result.retrieval_debug["counts"]
            print(
                "Retrieval:",
                f"reason={result.retrieval_debug['reason']}, "
                f"facts={counts['facts']}, episodes={counts['episodes']}, "
                f"sentences={counts['sentences']}",
            )

            facts = result.memories_used.get("facts", [])
            if facts:
                print("Top memory fact:", facts[0].get("text", ""))
            print()
    finally:
        await agent.close()
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
