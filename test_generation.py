import time
import torch

from model import tokenizer, model


print("\n====================================")
print("Gemma Generation Speed Test")
print("====================================")


prompt = """
You are the official DesFlyer FAQ assistant.

Answer using only the information below.

Context:
DesFlyer offers customized, high-quality, secure, and scalable software solutions.

Question:
What service does DesFlyer offer?

Answer:
"""


# ============================================================
# TOKENIZATION
# ============================================================

start = time.time()

inputs = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": prompt
        }
    ],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}

tokenization_time = time.time() - start

input_tokens = inputs["input_ids"].shape[-1]

print(f"📥 Input tokens: {input_tokens}")
print(f"⏱️ Tokenization time: {tokenization_time:.2f} seconds")


# ============================================================
# GENERATION
# ============================================================

print("\n🤖 Generating...")

start = time.time()

with torch.inference_mode():

    outputs = model.generate(

        **inputs,

        max_new_tokens=20,

        do_sample=False,

        use_cache=True,

        pad_token_id=tokenizer.eos_token_id

    )

generation_time = time.time() - start


# ============================================================
# DECODE
# ============================================================

input_length = inputs["input_ids"].shape[-1]

generated_tokens = outputs[0][input_length:]

answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()


output_tokens = len(generated_tokens)

# ============================================================
# RESULTS
# ============================================================

print("\n====================================")
print("Generation Result")
print("====================================")

print("Answer:")
print(answer)

print("\nInput tokens:", input_tokens)
print("Output tokens:", output_tokens)

print(
    f"Generation time: {generation_time:.2f} seconds"
)

if generation_time > 0:

    speed = output_tokens / generation_time

    print(
        f"Generation speed: {speed:.2f} tokens/sec"
    )

print("====================================")