from flask import Flask, request, jsonify
from sentence_transformers import CrossEncoder
import numpy as np
import re
import logging

# Setup Flask
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# 1. Load the Model
MODEL_NAME = "cross-encoder/qnli-electra-base"
print(f"Loading model: {MODEL_NAME} ...")
model = CrossEncoder(MODEL_NAME)

# 2. Advanced Calibration
print("Calibrating model logic...")
dummy_pairs = [
    ("Q", "Wrong Answer"),  # Expect low score
    ("Q", "Q")  # Expect high score (Identity match)
]
scores = model.predict(dummy_pairs)

# Detect Output Type
SINGLE_SCORE_MODE = False
SKIP_ACTIVATION = False  # New flag for models that return 0/1 or 0.0-1.0 natively

# Check dimensions
if scores.ndim == 1 or (scores.ndim == 2 and scores.shape[1] == 1):
    SINGLE_SCORE_MODE = True
    print("Detected: Single-score output.")

    # Check if the model is outputting Hard Labels (0/1) or Probabilities
    flat_scores = scores.flatten() if scores.ndim > 1 else scores
    max_score = np.max(flat_scores)
    min_score = np.min(flat_scores)

    print(f"Calibration Range: Min={min_score}, Max={max_score}")

    # If scores are exactly 0 and 1, or already bounded [0,1], don't Sigmoid them again!
    if (0 <= min_score and max_score <= 1.0):
        print("Detected: Model returns Probabilities/Labels (0-1). Skipping Sigmoid.")
        SKIP_ACTIVATION = True
    else:
        print("Detected: Model returns Logits. Using Sigmoid.")

else:
    print("Detected: Multi-class logits output. Using Softmax.")
    # Determine entailment index (usually 1 or 0)
    # We assume the second pair ("Q", "Q") should have the higher score in the entailment column
    logits_wrong = scores[0]
    logits_right = scores[1]

    # Find which column increased for the "Right" answer
    if logits_right[1] > logits_right[0]:
        ENTAILMENT_IDX = 1
    else:
        ENTAILMENT_IDX = 0
    print(f"Entailment matches Index: {ENTAILMENT_IDX}")


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        # Parse Input
        if "question" in data and "facts" in data:
            question = data["question"]
            nodes = data["facts"]
        else:
            user_prompt = data.get("prompt", "").strip()
            if not user_prompt: return jsonify({"error": "No prompt."}), 400

            q_match = re.search(r"Player Question:\s*(.*?)(?:\n|Story Nodes:|$)", user_prompt, re.IGNORECASE)
            question = q_match.group(1).strip() if q_match else ""

            nodes_block_match = re.search(r"Story Nodes:\s*(.*?)($|Return the relevance)", user_prompt,
                                          re.DOTALL | re.IGNORECASE)
            nodes = []
            if nodes_block_match and question:
                nodes_text = nodes_block_match.group(1).strip()
                lines = nodes_text.split('\n')
                for line in lines:
                    line = re.sub(r'^\d+[:.]\s*', '', line.strip())
                    if line: nodes.append(line)

        if not nodes or not question:
            return jsonify({"response": "[]", "history": []})

        # Predict
        pairs = [(question, fact) for fact in nodes]
        raw_output = model.predict(pairs)

        # Apply correct activation
        if SINGLE_SCORE_MODE:
            if SKIP_ACTIVATION:
                # Model already gave us 0.0 to 1.0 (or 0/1)
                final_probs = raw_output
                # Flatten if necessary
                if final_probs.ndim > 1: final_probs = final_probs.flatten()
            else:
                final_probs = sigmoid(raw_output)

            relevant_scores = final_probs
        else:
            probs = softmax(raw_output)
            relevant_scores = probs[:, ENTAILMENT_IDX]

        # Guarantee 0.0 - 1.0 range
        relevant_scores = np.clip(relevant_scores, 0.0, 1.0)

        # Formatting: Increase precision to 5 decimals to catch subtle ranking differences
        final_scores = [round(float(s), 5) for s in relevant_scores]
        response_string = str(final_scores).replace("'", "")

        # Tie Detection Logger
        sorted_scores = sorted(final_scores, reverse=True)
        is_tie = False
        if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1] and sorted_scores[0] > 0.5:
            is_tie = True

        print(f"\n[Scoring Request] Q: {question}")
        print(f"Nodes: {len(nodes)} | Top Score: {sorted_scores[0]}")
        if is_tie:
            print(f"⚠️  WARNING: Tie detected at score {sorted_scores[0]}. Ranking might be ambiguous.")

        return jsonify({"response": response_string, "history": []})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    try:
        port = int(input("Enter port number: "))
    except:
        port = 5000
    print(f"Serving Cross-Encoder on port {port}...")
    app.run(host="127.0.0.1", port=port)