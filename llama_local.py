import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

start_time = time.time()  # Start tracking time

def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_length=max_length)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model, tokenizer, device = load_model(model_name)
    prompt = "Explain black holes in simple terms."
    output_text = generate_text(model, tokenizer, device, prompt)
    print("Generated Text:", output_text)
    end_time = time.time()  # End tracking time
    print(f"Generation time: {end_time - start_time:.2f} seconds")  # Print the time taken
