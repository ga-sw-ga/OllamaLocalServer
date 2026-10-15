import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: No GROQ_API_KEY found in .env file!")

try:
    client = Groq(api_key=GROQ_API_KEY)
    print("Successfully connected to Groq.")
except Exception as e:
    print(f"Connection Error: {e}")

MODEL_ID = "qwen/qwen3.8-27b"

app = Flask(__name__)

# ---------------------------------------------------------
# SYSTEM PROMPT (Detection - Method 2)
# ---------------------------------------------------------
DETECTION_PROMPT = """You are an expert semantic text analysis model. Your task is to determine if the core meaning of a "Fact" is confirmed by the given "Dialogue".

Evaluation Rules:
1. Pronoun Context: Assume "I", "me", or "my" in the dialogue directly refers to Adrian Gale.
2. Semantic Matching: Focus on the meaning, not exact word-for-word matches.

First, write a single brief sentence evaluating if the dialogue confirms the fact.
Then, on a new line, output exactly "RESULT: true" or "RESULT: false"."""

# ---------------------------------------------------------
# THE ENDPOINTS
# ---------------------------------------------------------

# --- ENDPOINT: THE LOGIC DETECTOR ---
@app.route('/chat', methods=['POST'])
def chat_detect():
    try:
        user_text = request.get_json().get("prompt", "").strip()

        # Stateless: We send the system prompt and the user's text dynamically
        messages_payload = [
            {"role": "system", "content": DETECTION_PROMPT},
            {"role": "user", "content": user_text}
        ]

        # Low temperature for strict logic (0.1)
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages_payload,
            temperature=0.1,
            max_tokens=150
        )

        ai_reply = response.choices[0].message.content.strip()

        # The Parser
        ai_reply_lower = ai_reply.lower()
        final_boolean_answer = "true" if "result: true" in ai_reply_lower else "false"

        print(f"\n[Evaluating Fact]:\n{user_text}")
        print(f"[LLM Logic]: {ai_reply}")
        print(f"[Engine Result]: {final_boolean_answer}")

        return jsonify({"response": final_boolean_answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "false"}), 500

# --- ENDPOINT: RESET (Just returns success since detection is stateless) ---
@app.route('/reset', methods=['POST'])
def reset():
    print("\n[System]: Detection pinged for reset (Stateless, no action needed).")
    return jsonify({"status": "success"})


if __name__ == "__main__":
    print("Starting Groq Detection Server on Port 5007...")
    app.run(host="127.0.0.1", port=5007)