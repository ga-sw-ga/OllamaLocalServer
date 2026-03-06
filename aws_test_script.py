import requests
import time
import json


def run_test_client():
    print("-------------------------------------------------")
    print("      AWS BEDROCK FLASK SERVER TESTER")
    print("-------------------------------------------------")

    # 1. Configuration
    host = "127.0.0.1"

    # Get the port to match whatever you typed in the Server window
    try:
        port_input = input("Enter Server Port (default 5000): ")
        port = int(port_input) if port_input else 5000
    except ValueError:
        port = 5000

    url = f"http://{host}:{port}/ask"

    # 2. Setup the NPC Personality (System Prompt)
    print("\n[Configuration]")
    personality = input("Enter NPC Personality (Press Enter for 'Helpful Assistant'): ")
    if not personality:
        personality = "You are a helpful AI assistant."

    print(f"\nConnecting to: {url}")
    print(f"NPC Persona: {personality}")
    print("Type 'quit' or 'exit' to stop.\n")

    # 3. Main Chat Loop
    while True:
        user_prompt = input("\n[You]: ")

        if user_prompt.lower() in ["quit", "exit"]:
            break

        if not user_prompt.strip():
            continue

        # Prepare the data (Payload)
        payload = {
            "prompt": user_prompt,
            "personality": personality
        }

        try:
            # Start Timer
            start_time = time.time()

            # Send Request to your Flask Server
            response = requests.post(url, json=payload)

            # Stop Timer
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms

            # Check if successful
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "No response field found.")

                print(f"[AWS]: {answer}")
                print(f"      (Latency: {latency:.0f}ms)")
            else:
                print(f"[Error] Server returned status code: {response.status_code}")
                print(response.text)

        except requests.exceptions.ConnectionError:
            print(f"[Error] Could not connect to {url}.")
            print("Make sure your Flask server is running in a separate window!")
            break
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    # Ensure 'requests' library is installed
    try:
        run_test_client()
    except ImportError:
        print("Missing library. Please run: pip install requests")