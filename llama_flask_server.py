from flask import Flask, request, jsonify
import requests
from transformers import AutoTokenizer

# Setup
app = Flask(__name__)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

# Load tokenizer (used if needed for advanced control, not required for core logic)
TOKENIZER_PATH = r"C:\Users\parsa.rahmaty\.cache\llamatokenizer"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)


# 🧠 ChatSession stores the running conversation
class ChatSession:
    def __init__(self, model="llama3.2"):
        self.history = []
        self.model = model
        self.system_prompt = "You are a helpful assistant."  # You can customize this

    def add_user_message(self, message):
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.history.append({"role": "assistant", "content": message})

    def build_prompt(self):
        prompt = f"<|system|>\n{self.system_prompt}\n"
        for msg in self.history:
            tag = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
            prompt += f"{tag}\n{msg['content'].strip()}\n"
        return prompt

    def send_prompt(self):
        prompt_text = self.build_prompt()
        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "stream": False
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload)
            result = response.json()
            assistant_reply = result.get("response", "").strip()
            self.add_assistant_message(assistant_reply)
            return assistant_reply
        except Exception as e:
            return f"Error from Ollama: {e}"


# 🧠 Global session (shared across requests)
chat_session = ChatSession(model=MODEL_NAME)


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_prompt = data.get("prompt", "").strip()

        if not user_prompt:
            return jsonify({"error": "No prompt provided."}), 400

        chat_session.add_user_message(user_prompt)
        assistant_reply = chat_session.send_prompt()

        return jsonify({
            "response": assistant_reply,
            "history": chat_session.history
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    global chat_session
    chat_session = ChatSession(model=MODEL_NAME)
    return jsonify({"message": "Chat history reset."})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005)
