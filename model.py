from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "google/gemma-2-2b-it"

print("\n====================================")
print("Loading Gemma 2B")
print("====================================")

# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

    print("✅ CUDA GPU detected")
    print("GPU:", torch.cuda.get_device_name(0))

else:

    DEVICE = "cpu"

    print("⚠️ CUDA GPU not available")
    print("Using CPU")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading model...")

if DEVICE == "cuda":

    model = AutoModelForCausalLM.from_pretrained(

        MODEL_ID,

        torch_dtype=torch.float16,

        device_map="cuda",

        low_cpu_mem_usage=True

    )

else:

    model = AutoModelForCausalLM.from_pretrained(

        MODEL_ID,

        torch_dtype=torch.float32

    )

    model = model.to("cpu")


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n====================================")
print("Gemma 2B Loaded Successfully!")
print("====================================")

print("Device:", DEVICE)

print(
    "Model device:",
    next(model.parameters()).device
)

print(
    "Data type:",
    next(model.parameters()).dtype
)

if torch.cuda.is_available():

    print(
        "GPU memory allocated:",
        round(
            torch.cuda.memory_allocated() / 1024**3,
            2
        ),
        "GB"
    )

    print(
        "GPU memory reserved:",
        round(
            torch.cuda.memory_reserved() / 1024**3,
            2
        ),
        "GB"
    )

print("====================================\n")