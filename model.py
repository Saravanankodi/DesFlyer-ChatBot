from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "google/gemma-2-2b-it"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("Gemma 2B Loaded Successfully!")