from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# ============================================================
# MODEL
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

DEVICE = "cpu"

# ============================================================
# CPU SETTINGS
# ============================================================

CPU_CORES = os.cpu_count() or 1

# Keep CPU usage controlled
CPU_THREADS = min(4, CPU_CORES)

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# MODEL LOADING
# ============================================================

print("\n====================================")
print("Loading Qwen2.5-1.5B-Instruct")
print("====================================")

print("Model:", MODEL_ID)
print("Device:", DEVICE)
print("CPU cores:", CPU_CORES)
print("CPU threads:", CPU_THREADS)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

print("✅ Tokenizer loaded")


# ============================================================
# MODEL
# ============================================================

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    low_cpu_mem_usage=True
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# INFERENCE SETTINGS
# ============================================================

model.config.use_cache = True

torch.set_grad_enabled(False)


# ============================================================
# INFORMATION
# ============================================================

print("\n====================================")
print("Qwen2.5-1.5B-Instruct Loaded Successfully!")
print("====================================")

print(
    "Model device:",
    next(model.parameters()).device
)

print(
    "Data type:",
    next(model.parameters()).dtype
)

print(
    "CPU threads:",
    torch.get_num_threads()
)

try:

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count / 1e6:.0f} Million"
    )

except Exception:

    pass

print("====================================\n")