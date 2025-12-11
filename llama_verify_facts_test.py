import requests
import json
import re


def run_tests():
    print("--- Story Verification Test Suite (LLM Version) ---")
    try:
        port = int(input("Enter LLM server port number (e.g. 5000): "))
    except ValueError:
        port = 5000

    url = f"http://127.0.0.1:{port}/chat"
    reset_url = f"http://127.0.0.1:{port}/reset"

    # Reset chat history before starting to ensure clean context for every request if needed,
    # though we are treating each request as a standalone prompt in this loop.
    try:
        requests.post(reset_url)
    except:
        pass

    # Your specific Story Nodes (Same test cases as before)
    test_cases = [
        {
            "fact": "The girls are Rachel Myers, Chloe Bennett, and Laura Hayes",
            "scenarios": [
                {"type": "✅ Reveal",
                 "text": "I remember them clearly now... it was Rachel Myers, Chloe Bennett, and Laura Hayes.",
                 "should_reveal": True},
                {"type": "⚠️ Vague", "text": "I know who the girls are, I've seen their names in the paper.",
                 "should_reveal": False},
                {"type": "❌ Lie", "text": "I don't know who those girls were.", "should_reveal": False}
            ]
        },
        {
            "fact": "The girls report being kidnapped by a man",
            "scenarios": [
                {"type": "✅ Reveal", "text": "The girls said a guy snatched them.", "should_reveal": True},
                {"type": "⚠️ Vague", "text": "Someone took them, that's for sure.", "should_reveal": False},
                {"type": "❌ Lie", "text": "They weren't kidnapped, they ran away.", "should_reveal": False}
            ]
        },
        {
            "fact": "The girls were drugged and abducted",
            "scenarios": [
                {"type": "✅ Reveal",
                 "text": "He used chloroform or something... drugged them before dragging them off.",
                 "should_reveal": True},
                {"type": "⚠️ Vague", "text": "They were taken against their will.", "should_reveal": False},
                {"type": "❌ Lie", "text": "They went with him willingly.", "should_reveal": False}
            ]
        },
        {
            "fact": "The kidnapper threatened to kill the girls",
            "scenarios": [
                {"type": "✅ Reveal", "text": "He screamed at them... said he'd end their lives if they moved.",
                 "should_reveal": True},
                {"type": "⚠️ Vague", "text": "He was very aggressive verbally.", "should_reveal": False},
                {"type": "❌ Lie", "text": "He was actually quite polite to them.", "should_reveal": False}
            ]
        },
        {
            "fact": "The kidnapper wore a mask",
            "scenarios": [
                {"type": "✅ Reveal", "text": "We couldn't identify him because he had this creepy mask on.",
                 "should_reveal": True},
                {"type": "⚠️ Vague", "text": "I didn't get a good look at his face.", "should_reveal": False},
                {"type": "❌ Lie", "text": "He wasn't hiding his face at all.", "should_reveal": False}
            ]
        },
        {
            "fact": "A young man was also kidnapped there",
            "scenarios": [
                {"type": "✅ Reveal", "text": "It wasn't just the girls, a teenage boy was taken too.",
                 "should_reveal": True},
                {"type": "⚠️ Vague", "text": "There was someone else there besides the girls.", "should_reveal": False},
                {"type": "❌ Lie", "text": "Only the three girls were taken.", "should_reveal": False}
            ]
        },
        {
            "fact": "The young man comforted the girls",
            "scenarios": [
                {"type": "✅ Reveal", "text": "The boy tried to keep them calm, told them it would be okay.",
                 "should_reveal": True},
                {"type": "⚠️ Vague", "text": "The boy talked to them a lot.", "should_reveal": False},
                {"type": "❌ Lie", "text": "The boy ignored them completely.", "should_reveal": False}
            ]
        },
        {
            "fact": "The girls had no idea why they were kidnapped",
            "scenarios": [
                {"type": "✅ Reveal", "text": "They kept asking 'why us?', they were totally clueless about the motive.",
                 "should_reveal": True},
                {"type": "⚠️ Vague", "text": "They were very confused.", "should_reveal": False},
                {"type": "❌ Lie", "text": "They knew exactly why they were there.", "should_reveal": False}
            ]
        },
        {
            "fact": "The girls were found in the sewers",
            "scenarios": [
                {"type": "✅ Reveal", "text": "Police located them deep inside the city sewer tunnels.",
                 "should_reveal": True},
                {"type": "⚠️ Vague", "text": "They were found underground.", "should_reveal": False},
                {"type": "❌ Lie", "text": "They were found in an abandoned warehouse.", "should_reveal": False}
            ]
        },
        {
            "fact": "The girls found a way out and escaped through the sewers",
            "scenarios": [
                {"type": "✅ Reveal", "text": "They managed to break loose and crawled out via the sewer lines.",
                 "should_reveal": True},
                {"type": "⚠️ Vague", "text": "They managed to escape on their own.", "should_reveal": False},
                {"type": "❌ Lie", "text": "The police had to break in to rescue them.", "should_reveal": False}
            ]
        }
    ]

    print(f"\nRunning {len(test_cases) * 3} scenarios via LLM...\n")

    passes = 0
    fails = 0

    for case in test_cases:
        print(f"🔹 Fact: {case['fact']}")
        for scenario in case["scenarios"]:

            # Construct a prompt that forces the LLM to act as a logic verifier
            prompt = (
                f"I need to verify if a suspect's dialogue reveals a specific fact.\n\n"
                f"Fact: \"{case['fact']}\"\n"
                f"Suspect Dialogue: \"{scenario['text']}\"\n\n"
                f"Does the dialogue explicitly reveal the fact provided? "
                f"Respond with exactly one word: TRUE or FALSE."
            )

            payload = {
                "prompt": prompt
            }

            try:
                # We reset the history for every question so the LLM doesn't get confused by previous facts
                requests.post(reset_url)

                response = requests.post(url, json=payload)
                result = response.json()
                assistant_reply = result.get("response", "").strip().upper()

                # Parse the LLM response
                # We look for explicit TRUE or FALSE keywords
                is_true = "TRUE" in assistant_reply
                is_false = "FALSE" in assistant_reply

                if is_true and not is_false:
                    revealed = True
                elif is_false and not is_true:
                    revealed = False
                else:
                    # Ambiguous response handling
                    revealed = False
                    assistant_reply = f"Ambiguous ({assistant_reply})"

                # Check if result matches expectation
                is_pass = (revealed == scenario["should_reveal"])

                status_icon = "✅" if is_pass else "❌"

                print(f"   {status_icon} [{scenario['type']}] Dial: \"{scenario['text']}\"")
                print(f"      -> LLM says: {revealed} (Raw: {assistant_reply})")

                if is_pass:
                    passes += 1
                else:
                    fails += 1
                    print(f"      ⚠️ FAILURE: Expected {scenario['should_reveal']} but got {revealed}")

            except Exception as e:
                print(f"🚨 Connection Error: {e}")
                return
        print("-" * 40)

    print(f"\nFinal Results: {passes} Pass | {fails} Fail")
    print(f"Accuracy: {(passes / (passes + fails)) * 100:.1f}%")


if __name__ == "__main__":
    run_tests()
