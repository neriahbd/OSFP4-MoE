# OSFP4-MoE

OSFP4 quantizes MoE models with optimized per-channel smoothing and FP4 scales. Each expert shares a smoothing vector across its gate/up projections and uses a separate vector for its down projection. LLM Compressor calibrates and exports the model; compressed-tensors stores the diagonal transforms; vLLM applies them during inference.

Supports NVFP4 (W4A4) and NVFP4A16 (W4A16), with RTN or SIC optimization.

## Forks

| Fork | Branch | Role |
|---|---|---|
| [LLM Compressor](https://github.com/orimeirgit/llm-compressor-for-osfp4/tree/add-moe-support) | `add-moe-support` | OSFP4 calibration and export |
| [compressed-tensors](https://github.com/orimeirgit/compressed-tensors-for-osfp4/tree/osfp4-diagonal-transform) | `osfp4-diagonal-transform` | Diagonal transform representation |
| [vLLM](https://github.com/neriahbd/vllm-for-osfp4/tree/add-moe-support) | `add-moe-support` | Native MoE inference |

## 1. Install for calibration

Use `uv` and a CUDA-enabled PyTorch environment with enough memory for the BF16 model and calibration workspace. Run from a new working directory:

```bash
git clone -b add-moe-support https://github.com/orimeirgit/llm-compressor-for-osfp4.git llm-compressor
git clone -b osfp4-diagonal-transform https://github.com/orimeirgit/compressed-tensors-for-osfp4.git compressed-tensors
git clone -b add-moe-support https://github.com/neriahbd/vllm-for-osfp4.git vllm

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ./compressed-tensors -e ./llm-compressor --torch-backend=auto
uv pip check --python .venv/bin/python
```

## 2. Quantize and export

Save this as `quantize.py` and run `.venv/bin/python quantize.py`. This example quantizes [Qwen/Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B) (14.3B total parameters, 2.7B active per token) using 128 passages from the [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) training split, with up to 512 tokens per passage. This small calibration subset is for trying the workflow; use representative data and check expert coverage when evaluating quality.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.osfp4 import OSFP4Modifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
SAVE_DIR = "moe-osfp4"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ds = ds.filter(lambda row: len(row["text"].split()) >= 64)
ds = ds.shuffle(seed=42).select(range(128))
ds = ds.map(
    lambda row: tokenizer(
        row["text"], truncation=True, max_length=512,
        padding=False, add_special_tokens=False,
    ),
    remove_columns=ds.column_names,
)

oneshot(
    model=model,
    dataset=ds,
    recipe=OSFP4Modifier(
        scheme="NVFP4",           # Or "NVFP4A16" for W4A16.
        optimization_mode="sic",  # Or "rtn".
        targets=["Linear"],
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        activation_subsample_size="auto",
    ),
    num_calibration_samples=128,
    max_seq_length=512,
    propagate_error=True,
    moe_calibrate_all_experts=False,
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

`load_context()` prepares supported MoE experts for calibration. Keep router layers in full precision. The example uses `moe_calibrate_all_experts=False` so each expert is calibrated on its routed tokens. Check expert coverage with your calibration data. Set it to `True` to expose every expert to all calibration tokens.

## 3. Serve with vLLM

From the same working directory, create a separate serving environment:

```bash
cd vllm
uv venv --python 3.12 .venv
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python \
  -e . -e ../compressed-tensors --torch-backend=auto
uv pip check --python .venv/bin/python

.venv/bin/vllm serve /absolute/path/to/moe-osfp4 \
  --tensor-parallel-size 1 --enforce-eager
```

Use a GPU supported by the selected backend: CUTLASS for NVFP4 or Humming for NVFP4A16. Follow the [vLLM installation guide](https://github.com/neriahbd/vllm-for-osfp4/tree/add-moe-support/docs/getting_started/installation) for backend dependencies or a source build if compatible precompiled artifacts are unavailable. No OSFP4 plugin is needed.

Start with single-GPU execution. Expert parallelism, expert load balancing, and fused shared-expert execution are not supported by this transformed-expert path. The commands are a development workflow, not a GPU-validated release recipe.
