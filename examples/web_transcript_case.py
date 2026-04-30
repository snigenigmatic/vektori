import asyncio
import urllib.request
import json
import os

from vektori import AgentConfig, Vektori, VektoriAgent
from vektori.models.factory import create_chat_model

def get_hacker_news_item(item_id):
    url = f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def fetch_hn_discussion():
    # Fetch Ask HN: What is your favorite productivity tool?
    item_id = 31089169
    data = get_hacker_news_item(item_id)
    messages = []
    
    title = data.get('title', '')
    text = data.get('text', '')
    messages.append({
        "role": "user",
        "content": f"Hey, I was reading this discussion: '{title}'. The post said: {text}"
    })
    
    # fetch top level kids
    kids = data.get('kids', [])[:5]
    for kid_id in kids:
        kid_data = get_hacker_news_item(kid_id)
        if kid_data and 'text' in kid_data:
            messages.append({
                "role": "assistant",
                "content": kid_data['text'][:500] 
            })
            
    return messages

async def main():
    print("Fetching real Hacker News discussion...")
    messages = fetch_hn_discussion()
    print(f"Fetched {len(messages)} messages from HN.")
    
    chat_model = os.getenv("VEKTORI_CHAT_MODEL", "litellm:gpt-4o-mini")
    embed_model = os.getenv("VEKTORI_EMBED_MODEL", "openai:text-embedding-3-small")
    extract_model = os.getenv("VEKTORI_EXTRACT_MODEL", "litellm:gpt-4o-mini")

    print(f"Using models:\nChat: {chat_model}\nEmbed: {embed_model}\nExtract: {extract_model}")

    async with Vektori(
        embedding_model=embed_model,
        extraction_model=extract_model,
        async_extraction=False
    ) as memory:
        agent = VektoriAgent(
            memory=memory,
            model=create_chat_model(chat_model),
            user_id="hn-user",
            agent_id="hn-agent",
            config=AgentConfig(
                retrieve_on_every_turn=True,
                background_add=False,
            )
        )

        print("Ingesting discussion into Vektori Agent...")
        await agent.add_messages(messages)
        
        print("\nNow asking the agent about the discussion...")
        question = "What productivity tools were mentioned in the discussion? Just give me a short list."
        print(f"\nYou: {question}")
        result = await agent.chat(question)
        print(f"\nAssistant: {result.content}")
        
        print("\nRetrieval Memory Used:")
        for fact in result.memories_used.get("facts", []):
            score = fact.get("score", 0.0)
            print(f"- [Fact {score:.2f}] {fact['text']}")
        for ep in result.memories_used.get("episodes", []):
            print(f"- [Episode] {ep['text']}")

if __name__ == "__main__":
    asyncio.run(main())
