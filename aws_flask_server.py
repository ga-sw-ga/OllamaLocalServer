import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# 1. Find the exact folder this python script is located in
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join that folder path with '.env'
env_path = os.path.join(script_dir, '.env')

# 3. Explicitly load that exact file
load_dotenv(dotenv_path=env_path)

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: No GROQ_API_KEY found in .env file!")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------
app = Flask(__name__)

# The specific Llama 3.3 70B model hosted by Groq
MODEL_ID = "qwen/qwen3.8-27b"
SYSTEM_PROMPT = "You are a helpful NPC in a video game. Keep your answers concise and immersive."

# ---------------------------------------------------------
# MEMORY CLASS (Adapted for Groq/OpenAI Format)
# ---------------------------------------------------------
class ChatSession:
    def __init__(self):
        self.history = []

    def add_user_message(self, content):
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        self.history.append({"role": "assistant", "content": content})

    def clear(self):
        self.history = []

# Create a global session to hold memory while the server runs
current_session = ChatSession()

# ---------------------------------------------------------
# ROUTES (Matching ULlamaRequester.cpp)
# ---------------------------------------------------------

# 1. THE CHAT ENDPOINT
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_text = data.get("prompt", "").strip()

        if not user_text:
            return jsonify({"error": "No prompt provided"}), 400

        # Add the detective's new line to the history
        current_session.add_user_message(user_text)

        # Build the exact message format the Groq API expects
        # Insert the system prompt at the beginning of the list
        messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + current_session.history

        # Send to Groq
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages_payload,
            temperature=0.7,
            max_tokens=300
        )

        ai_reply = response.choices[0].message.content

        # Save Adrian's reply to the history
        current_session.add_assistant_message(ai_reply)

        print(f"\n[Detective]: {user_text}")
        print(f"[Suspect]: {ai_reply}")

        return jsonify({"response": ai_reply})

    except Exception as e:
        print(f"Error: {e}")
        # If the API crashes, remove the last user message so it doesn't corrupt the history sequence
        if current_session.history and current_session.history[-1]["role"] == "user":
            current_session.history.pop()

        return jsonify({"response": "I cannot speak right now. (Server Error)"}), 500


# 2. THE RESET ENDPOINT (Matches C++ "ResetContext")
@app.route('/reset', methods=['POST'])
def reset():
    current_session.clear()
    print("\n[System]: Context Memory Cleared.")
    return jsonify({"status": "success"})


# ---------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        port_input = input("Enter port number (default 5000): ")
        port = int(port_input) if port_input else 5000
    except ValueError:
        print("Invalid number. Defaulting to 5000.")
        port = 5000

    print(f"Starting Groq Chat Server on Port {port}...")
    app.run(host="127.0.0.1", port=port)