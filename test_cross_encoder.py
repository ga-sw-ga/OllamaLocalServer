import requests
import json
import numpy as np


def run_tests():
    print("--- Detective's Dilemma Test Client ---")
    try:
        port = int(input("Enter server port number (e.g. 5000): "))
    except ValueError:
        port = 5000

    url = f"http://127.0.0.1:{port}/chat"

    # These are the same tricky cases we discussed earlier
    test_cases = [
        {
            "id": "Lexical Trap",
            "question": "What are the names of the 3 girls who were kidnapped by a man?",
            "facts": [
                "The girls report being kidnapped by a man.",  # Trap (high overlap)
                "The girls had no idea why they were taken.",
                "The girls are Rachel, Chloe, and Laura."  # Correct
            ],
            "expected_index": 2
        },
        {
            "id": "Entity Swap",
            "question": "Who killed the Butler?",
            "facts": [
                "The Butler killed the Gardener.",  # Trap (Subject vs Object)
                "The Cook was killed by the Butler.",
                "The Gardener killed the Butler."  # Correct
            ],
            "expected_index": 2
        },
        {
            "id": "Negation",
            "question": "Which room is safe?",
            "facts": [
                "The Kitchen is NOT safe.",  # Trap (contains keywords)
                "The Library is dangerous.",
                "The Bedroom is free of danger."  # Correct (synonym)
            ],
            "expected_index": 2
        },
        {
            "id": "Temporal Logic",
            "question": "Where is the suspect hiding NOW?",
            "facts": [
                "The suspect was hiding in the attic yesterday.",
                "The suspect is currently in the basement.",  # Correct
                "The suspect plans to go to the roof."
            ],
            "expected_index": 1
        }
    ]

    print(f"\nConnecting to {url}...\n")

    passes = 0
    fails = 0

    for case in test_cases:
        # 1. Construct the prompt string exactly like Unreal would
        # We format it with "Player Question:" and "Story Nodes:" headers
        nodes_text = ""
        for i, fact in enumerate(case["facts"]):
            nodes_text += f"{i + 1}: {fact}\n"

        # Refactored to avoid triple-quote issues:
        unreal_prompt = (
            f"Player Question: {case['question']}\n"
            f"Story Nodes:\n"
            f"{nodes_text}"
            f"Return the relevance scores as a JSON array."
        )

        payload = {"prompt": unreal_prompt}

        try:
            # 2. Send Request
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            # 3. Parse Response
            # The server returns a string "[0.1, 0.5, ...]" inside the JSON
            score_string = data.get("response", "[]")
            scores = json.loads(score_string)

            if not scores:
                print(f"⚠️  SKIPPED: {case['id']} (Empty response)")
                continue

            # 4. Check Results
            best_idx = int(np.argmax(scores))
            expected_idx = case["expected_index"]
            is_correct = (best_idx == expected_idx)

            # 5. Print Output
            if is_correct:
                passes += 1
                print(f"✅ PASS: {case['id']}")
            else:
                fails += 1
                print(f"❌ FAIL: {case['id']}")
                print(f"   Question: {case['question']}")
                print(f"   Expected: ({expected_idx}) {case['facts'][expected_idx]}")
                print(f"   Got:      ({best_idx}) {case['facts'][best_idx]}")
                print(f"   Scores:   {scores}")

        except requests.exceptions.ConnectionError:
            print("🚨 Error: Could not connect to server. Is it running?")
            return
        except Exception as e:
            print(f"🚨 Error on {case['id']}: {e}")
            return

    print("-" * 30)
    print(f"Test Complete. Pass: {passes} | Fail: {fails}")
    if fails == 0:
        print("Result: 🏆 PERFECT SCORE")
    else:
        print("Result: 🔧 NEEDS TUNING")


if __name__ == "__main__":
    run_tests()