from transformers import AutoModelForCausalLM, AutoTokenizer

# Define model path
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Download model (without loading it into memory)
AutoModelForCausalLM.from_pretrained(model_id)

print("Model downloaded successfully!")
