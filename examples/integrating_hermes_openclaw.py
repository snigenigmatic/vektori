"""
Example showing how to configure Vektori memory backends inside
Hermes Agent and OpenClaw loops.
"""

from vektori import Vektori
from vektori.integrations.hermes import VektoriHermesMemory
from vektori.integrations.openclaw import VektoriOpenClawPlugin

async def configure_hermes():
    print("Initializing Hermes Vektori Memory Provider...")
    v = Vektori()
    
    # 1. Create the Memory Provider referencing our Vektori DB
    memory_provider = VektoriHermesMemory(v, user_id="user-123")
    
    # 2. Add it to a standard Hermes Agent
    # agent = HermesAgent(
    #     model="litellm:gemini-2.5-flash-lite",
    #     memory_provider=memory_provider
    # )
    
    print("Hermes configuration successful.")

async def configure_openclaw():
    print("Initializing OpenClaw Vektori Plugin...")
    v = Vektori()
    
    # 1. Create the active memory plugin replacement
    plugin = VektoriOpenClawPlugin(v)
    
    # 2. Add it to an OpenClaw registry
    # app = OpenClawApp()
    # app.register_plugin(plugin)
    
    print("OpenClaw configuration successful.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(configure_hermes())
    asyncio.run(configure_openclaw())
