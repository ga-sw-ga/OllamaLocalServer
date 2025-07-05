import requests

# Server URLs
chat_url = "http://127.0.0.1:5005/chat"
reset_url = "http://127.0.0.1:5005/reset"


# 🔁 Step 1: Reset context
def reset_context():
    try:
        response = requests.post(reset_url)
        response.raise_for_status()
        print("✅ Context reset:", response.json())
    except requests.exceptions.RequestException as e:
        print("❌ Reset failed:", e)


# 📤 Step 2: Send a test prompt
def send_prompt(prompt):
    data = {"prompt": prompt}
    try:
        response = requests.post(chat_url, json=data)
        response.raise_for_status()
        result = response.json()
        print("✅ Response:")
        print(result)
    except requests.exceptions.RequestException as e:
        print("❌ Prompt request failed:", e)
    except ValueError:
        print("❌ Invalid JSON in response.")


# 🧪 Run Test
# reset_context()
send_prompt("Hello! What was my last prompt to you exactly?")
