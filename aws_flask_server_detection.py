import boto3
import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path=env_path)

ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION = os.getenv("AWS_REGION", "us-west-2")

app = Flask(__name__)

# ---------------------------------------------------------
# 1. AWS CONNECTION
# ---------------------------------------------------------
try:
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    print("Successfully connected to AWS Bedrock.")
except Exception as e:
    print(f"Connection Error: {e}")

MODEL_ID = "meta.llama3-1-70b-instruct-v1:0"

# ---------------------------------------------------------
# 2. SYSTEM PROMPT (Detection - Method 2)
# ---------------------------------------------------------
DETECTION_PROMPT = """You are an expert semantic text analysis model. Your task is to determine if the core meaning of a "Fact" is confirmed by the given "Dialogue".

Evaluation Rules:
1. Pronoun Context: Assume "I", "me", or "my" in the dialogue directly refers to Adrian Gale.
2. Semantic Matching: Focus on the meaning, not exact word-for-word matches.

First, write a single brief sentence evaluating if the dialogue confirms the fact.
Then, on a new line, output exactly "RESULT: true" or "RESULT: false"."""

# ---------------------------------------------------------
# 3. FORMATTER
# ---------------------------------------------------------
def format_llama_prompt(system_info, history):
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_info}<|eot_id|>"
    for msg in history:
        prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

# ---------------------------------------------------------
# 4. THE ENDPOINTS
# ---------------------------------------------------------

# --- ENDPOINT: THE LOGIC DETECTOR ---
# Notice this is back to '/chat' so Unreal Engine doesn't complain!
@app.route('/chat', methods=['POST'])
def chat_detect():
    try:
        user_text = request.get_json().get("prompt", "").strip()

        # Stateless: We create a temporary history list just for this one request
        temp_history = [{"role": "user", "content": user_text}]
        compiled_prompt = format_llama_prompt(DETECTION_PROMPT, temp_history)

        # Low temperature for strict logic (0.1)
        payload = {"prompt": compiled_prompt, "max_gen_len": 150, "temperature": 0.1}

        response = bedrock.invoke_model(body=json.dumps(payload), modelId=MODEL_ID, accept='application/json',
                                        contentType='application/json')
        ai_reply = json.loads(response.get('body').read()).get('generation', '').strip()

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
    print("Starting Detection Server on Port 5007...")
    # This specifically targets Port 5007
    app.run(host="127.0.0.1", port=5007)