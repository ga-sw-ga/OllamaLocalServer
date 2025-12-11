from flask import Flask, request, jsonify
from sentence_transformers import CrossEncoder
import numpy as np
import re
import logging

# Setup Flask
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ==========================================
# LOAD VERIFIER MODEL (Fact Checking)
# ==========================================
# Switched to a model trained on FEVER (Fact Verification) & ANLI (Adversarial Logic)
VERIFIER_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
print(f"Loading Verifier Model: {VERIFIER_MODEL_NAME} ...")
verifier_model = CrossEncoder(VERIFIER_MODEL_NAME)

# --- Calibration for Verifier (NLI) ---
# We auto-detect which output index means "True/Entailment"
print("Calibrating Verifier logic...")
# Pair that is definitely TRUE
test_entailment = verifier_model.predict([("I murdered him.", "The speaker killed someone.")])
# Pair that is definitely FALSE (Contradiction)
test_contradiction = verifier_model.predict([("I did not do it.", "The speaker killed someone.")])

VERIFIER_ENTAIL_IDX = int(np.argmax(test_entailment[0]))
print(f"Verifier: Entailment (True) matches Index {VERIFIER_ENTAIL_IDX}")


# Helper Functions
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


# ==========================================
# ENDPOINT: VERIFY
# ==========================================
@app.route("/verify", methods=["POST"])
def verify():
    try:
        data = request.get_json()

        # Expects: {"dialogue": "...", "fact": "..."}
        dialogue = data.get("dialogue", "").strip()
        fact = data.get("fact", "").strip()

        if not dialogue or not fact:
            return jsonify({"error": "Missing dialogue or fact"}), 400

        # NLI Logic: Does Dialogue (Premise) Entail Fact (Hypothesis)?
        scores = verifier_model.predict([(dialogue, fact)])
        probs = softmax(scores)[0]  # Get probabilities

        entailment_score = probs[VERIFIER_ENTAIL_IDX]

        # DECISION LOGIC
        # We lower the threshold slightly to 0.4 because Verification models
        # can be cautious with pronouns ("them" vs "girls").
        is_revealed = False
        if entailment_score > 0.4:
            is_revealed = True

        # Debug Logging
        status_icon = "✅" if is_revealed else "❌"
        print(f"\n[Verify] {status_icon} Score: {entailment_score:.4f}")
        print(f"   Fact: {fact}")
        print(f"   Dial: {dialogue}")

        return jsonify({
            "revealed": is_revealed,
            "score": float(entailment_score)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    try:
        port = int(input("Enter port number: "))
    except:
        port = 5000
    print(f"Serving Fact Verifier on port {port}...")
    app.run(host="127.0.0.1", port=port)


# from flask import Flask, request, jsonify
# from sentence_transformers import CrossEncoder
# import numpy as np
# import re
# import logging
#
# # Setup Flask
# app = Flask(__name__)
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)
#
# # ==========================================
# # LOAD VERIFIER MODEL (For checking reveals)
# # ==========================================
# VERIFIER_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# print(f"Loading Verifier Model: {VERIFIER_MODEL_NAME} ...")
# verifier_model = CrossEncoder(VERIFIER_MODEL_NAME)
#
# # --- Calibration for Verifier (NLI) ---
# # We need to find which index corresponds to "Entailment" (True)
# # We test a pair that is DEFINITELY an entailment.
# print("Calibrating Verifier logic...")
# # Premise: "I murdered him with a knife." -> Hypothesis: "The victim was stabbed."
# test_entailment = verifier_model.predict([("I murdered him with a knife.", "The victim was stabbed.")])
# VERIFIER_ENTAIL_IDX = int(np.argmax(test_entailment[0]))
# print(f"Verifier: Entailment (True) is at Index {VERIFIER_ENTAIL_IDX}")
#
#
# # Helper Functions
# def softmax(x):
#     e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#     return e_x / e_x.sum(axis=1, keepdims=True)
#
#
# # ==========================================
# # ENDPOINT: VERIFY (New Logic)
# # ==========================================
# @app.route("/verify", methods=["POST"])
# def verify():
#     try:
#         data = request.get_json()
#
#         # Expects: {"dialogue": "...", "fact": "..."}
#         dialogue = data.get("dialogue", "").strip()
#         fact = data.get("fact", "").strip()
#
#         if not dialogue or not fact:
#             return jsonify({"error": "Missing dialogue or fact"}), 400
#
#         # NLI Logic: Does Dialogue (Premise) Entail Fact (Hypothesis)?
#         # Note the order: (Dialogue, Fact)
#         scores = verifier_model.predict([(dialogue, fact)])
#         probs = softmax(scores)[0]  # Get probabilities [Contradiction, Entailment, Neutral]
#
#         entailment_score = probs[VERIFIER_ENTAIL_IDX]
#
#         # DECISION LOGIC
#         # We consider it revealed if Entailment is the highest score
#         # AND it has a decent confidence (> 0.5)
#         is_revealed = False
#         if entailment_score > 0.5 and np.argmax(probs) == VERIFIER_ENTAIL_IDX:
#             is_revealed = True
#
#         print(f"\n[Verify] Revealed? {is_revealed}")
#         print(f"   Fact: {fact}")
#         print(f"   Dial: {dialogue}")
#         print(f"   Score: {entailment_score:.4f}")
#
#         return jsonify({
#             "revealed": is_revealed,
#             "score": float(entailment_score)
#         })
#
#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({"error": str(e)}), 500
#
#
# if __name__ == "__main__":
#     try:
#         port = int(input("Enter port number: "))
#     except:
#         port = 5000
#     print(f"Serving Verifier Model on port {port}...")
#     app.run(host="127.0.0.1", port=port)


# from sentence_transformers import CrossEncoder
#
# model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
# scores = model.predict([
#     # ("A man is eating pizza", "A man eats something"),
#     # ("A black race car starts up in front of a crowd of people.", "A man is driving down a lonely road."),
#     ("I don't know the names of the three girls", "The girls are Rachel Myers, Chloe Bennett, and Laura Hayes"),
# ])
#
# # Convert scores to labels
# label_mapping = ["contradiction", "entailment", "neutral"]
# labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
# print(labels)
# print(scores)
# # => ['entailment', 'contradiction']
