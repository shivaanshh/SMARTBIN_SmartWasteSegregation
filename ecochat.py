# ecochat.py — Powered by OpenAI GPT-4o-mini
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load from .env file if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are EcoChat 🌿, an expert AI assistant for waste management and recycling built into the SmartBin Dashboard. "
    "Your job is to help users understand how to properly dispose of, recycle, or repurpose items detected by the SmartBin camera.\n\n"
    "For every item or question:\n"
    "1. Identify the correct bin: Wet (green), Dry (blue), or Metal (grey).\n"
    "2. State whether it is Biodegradable or Non-Biodegradable.\n"
    "3. Give 2-3 practical recycling or repurposing tips.\n"
    "4. If compostable, say so and explain how.\n"
    "5. If hazardous (e-waste, batteries, chemicals), give safe disposal instructions.\n\n"
    "Keep responses concise, friendly, and actionable. Use emojis to make it engaging."
)

# Persistent chat history (list of role/content dicts)
_history = [{"role": "system", "content": SYSTEM_PROMPT}]


def eco_chat_response(user_input: str) -> str:
    """OpenAI-powered recycling assistant with persistent chat history."""
    global _history

    _history.append({"role": "user", "content": user_input.strip()})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=_history,
            temperature=0.4,
            max_tokens=512,
        )
        reply = response.choices[0].message.content.strip()
        _history.append({"role": "assistant", "content": reply})

        # Keep history manageable (system + last 20 turns)
        if len(_history) > 22:
            _history = [_history[0]] + _history[-20:]

        return reply
    except Exception as e:
        # Remove the failed user message from history
        _history.pop()
        return f"⚠️ EcoChat error: {e}"


def eco_chat_recycle_tip(item_name: str, bin_type: str, degradability: str) -> str:
    """Quick recycling tip for a detected item."""
    prompt = (
        f"I just detected a **{item_name}** in my SmartBin camera. "
        f"It was classified as **{bin_type} Bin** ({degradability}). "
        f"Give me quick recycling/disposal tips and any creative reuse ideas."
    )
    return eco_chat_response(prompt)


# Test
if __name__ == "__main__":
    print("🌿 EcoChat (OpenAI GPT-4o-mini) — type 'exit' to quit\n")
    while True:
        q = input("♻️ You: ")
        if q.lower() == "exit":
            break
        print(f"\n🤖 EcoChat: {eco_chat_response(q)}\n")
