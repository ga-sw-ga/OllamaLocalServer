import boto3
import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# 1. Find the exact folder this python script is located in
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Join that folder path with '.env'
env_path = os.path.join(script_dir, '.env')

# 3. Explicitly load that exact file
load_dotenv(dotenv_path=env_path)

ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION = os.getenv("AWS_REGION", "us-west-2")

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------
app = Flask(__name__)

# AWS Setup - Uses 'aws configure' credentials automatically
# Ensure region matches your model access (e.g., us-west-2 or us-east-1)
try:
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    print("Successfully connected to AWS with hardcoded keys.")
except Exception as e:
    print(f"Connection Error: {e}")

MODEL_ID = "meta.llama3-1-70b-instruct-v1:0"  # Fast model for games
SYSTEM_PROMPT = "You are a helpful NPC in a video game. Keep your answers concise and immersive."


# ---------------------------------------------------------
# MEMORY CLASS (Because C++ doesn't send history)
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

    def get_bedrock_messages(self):
        # Bedrock expects a specific list format
        return self.history


# Create a global session to hold memory while the server runs
current_session = ChatSession()


# ---------------------------------------------------------
# ROUTES (Matching ULlamaRequester.cpp)
# ---------------------------------------------------------

# ---------------------------------------------------------
# LLAMA 3 SCRIPT FORMATTER
# ---------------------------------------------------------
def format_llama_prompt(system_info, history):
    # Cue the actor with the structured system information first
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_info}<|eot_id|>"

    # Run through the conversational history
    for msg in history:
        prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"

    # Cue the assistant to deliver their next line
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


# ---------------------------------------------------------
# 1. THE CHAT ENDPOINT
# ---------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_text = data.get("prompt", "").strip()

        if not user_text:
            return jsonify({"error": "No prompt provided"}), 400

        current_session.add_user_message(user_text)

        # Build the exact string format Llama 3 expects
        compiled_prompt = format_llama_prompt(SYSTEM_PROMPT, current_session.history)

        # Llama 3 Bedrock Payload
        payload = {
            "prompt": compiled_prompt,
            "max_gen_len": 300,
            "temperature": 0.7
        }

        # Send to AWS
        response = bedrock.invoke_model(
            body=json.dumps(payload),
            modelId=MODEL_ID,
            accept='application/json',
            contentType='application/json'
        )

        # Parse Llama's specific response format
        response_body = json.loads(response.get('body').read())
        ai_reply = response_body.get('generation', '')

        current_session.add_assistant_message(ai_reply)

        print(f"\n[Detective]: {user_text}")
        print(f"[Suspect]: {ai_reply}")

        return jsonify({"response": ai_reply})

    except Exception as e:
        print(f"Error: {e}")
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
    # Ask for port so you can match the Game's configuration
    try:
        port_input = input("Enter port number (default 5000): ")
        port = int(port_input) if port_input else 5000
    except ValueError:
        print("Invalid number. Defaulting to 5000.")
        port = 5000

    print(f"Starting AWS Bedrock Server on Port {port}...")
    app.run(host="127.0.0.1", port=port)