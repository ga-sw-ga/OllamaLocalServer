import requests
import json
from transformers import AutoTokenizer


# TOKENIZER SETUP
TOKENIZER_PATH = r"C:\Users\parsa.rahmaty\.cache\llamatokenizer"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)


# 🧠 ChatSession stores the running conversation
class ChatSession:
    def __init__(self, model="llama3.2"):
        self.history = []
        self.model = model
        self.system_prompt = "You are "  # Optional: can change

    def add_user_message(self, message):
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.history.append({"role": "assistant", "content": message})

    def build_prompt(self):
        # Start with system prompt
        prompt = f"<|system|>\n{self.system_prompt}\n"

        for msg in self.history:
            role_tag = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
            prompt += f"{role_tag}\n{msg['content'].strip()}\n"

        return prompt

    def send_prompt(self):
        url = "http://localhost:11434/api/generate"
        prompt_text = self.build_prompt()

        payload = {
            "model": self.model,
            "prompt": prompt_text,
            "stream": False
        }

        response = requests.post(url, json=payload)

        try:
            result = response.json()
            assistant_reply = result.get("response", "").strip()
            self.add_assistant_message(assistant_reply)
            return result
        except Exception as e:
            return f"Error: {e}"


def query_llama3(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)

    try:
        result = response.json()
        # print("Full Response:", json.dumps(result, indent=2))  # Debugging line

        return result

        # TO ONLY RETURN THE RESPONSE
        # if "response" in result:
        #     return result["response"]
        # else:
        #     return f"Error: Unexpected response format: {result}"

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except json.JSONDecodeError:
        return f"Error: Response is not in JSON format. Raw response: {response.text}"


def decode_tokens(token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def ollama_token_decode(token_ids):
    url = "http://localhost:11434/api/tokenize"


def reset_llama3_context():
    url = "http://localhost:11434/api/reset"
    payload = { "model": "llama3.2" }
    requests.post(url, json=payload)


# Example Usage

chat = ChatSession()

# Round 1
chat.add_user_message("Hi, what's your name?")
print("Assistant:", decode_tokens(chat.send_prompt()["context"]))

# Round 2
chat.add_user_message("Can you remember my name is Parsa?")
print("Assistant:", decode_tokens(chat.send_prompt()["context"]))

# Round 3
chat.add_user_message("What's my name?")
print("Assistant:", decode_tokens(chat.send_prompt()["context"]))
