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
BUILD_TYPE=nightly uv pip install --python .venv/bin/python ./compressed-tensors -e ./llm-compressor --torch-backend=auto
uv pip check --python .venv/bin/python
```

`BUILD_TYPE=nightly` avoids a development version such as `0.1.devN` that
fails LLM Compressor's `compressed-tensors>=0.20.2a2` requirement. Use it whenever
building this fork in either environment; the nightly version depends on the
checkout, so do not assume a fixed version string.

## 2. Quantize and export

Save this as `quantize.py` and run `.venv/bin/python quantize.py`. This example quantizes [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B total parameters, about 3B active per token) using 128 passages from the [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) training split, with up to 512 tokens per passage. This small calibration subset is for trying the workflow; use representative data and check expert coverage when evaluating quality.

```python
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.osfp4 import OSFP4Modifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
SAVE_DIR = "moe-osfp4"

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
with load_context(Qwen3_5MoeForConditionalGeneration):
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
    )

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
    processor=processor,
    dataset=ds,
    recipe=OSFP4Modifier(
        scheme="NVFP4",           # Or "NVFP4A16" for W4A16.
        optimization_mode="sic",  # Or "rtn".
        targets=["Linear"],
        ignore=[
            "re:.*lm_head",
            "re:visual.*",
            "re:model.visual.*",
            "re:.*mlp.gate$",
            "re:.*embed_tokens$",
            "re:.*shared_expert_gate$",
            "re:.*linear_attn.*",
        ],
        activation_subsample_size="auto",
    ),
    num_calibration_samples=128,
    max_seq_length=512,
    propagate_error=True,
    moe_calibrate_all_experts=False,
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
```

`load_context(Qwen3_5MoeForConditionalGeneration)` prepares the Qwen3.6 experts for calibration. This example uses text-only calibration for the multimodal checkpoint and leaves the vision tower, linear-attention modules, and router layers in full precision, following the fork's Qwen3.6 example. The model uses the Qwen3.5 MoE architecture, hence the loader class name. Budget memory for all 35B parameters (roughly 70 GB of BF16 weights before workspace), not just the 3B active per token; use adequate aggregate memory or supported offloading. The example uses `moe_calibrate_all_experts=False` so each expert is calibrated on its routed tokens. Check expert coverage with your calibration data. Set it to `True` to expose every expert to all calibration tokens.

## 3. Serve with vLLM

Create a separate serving environment. Use a direct-checkout override so vLLM
installs the companion compressed-tensors fork instead of the PyPI package.

```bash
cd vllm
uv venv --python 3.12 .venv
.venv/bin/python - <<'PYTHON'
from pathlib import Path
checkout = Path("../compressed-tensors").resolve(strict=True)
Path("osfp4-overrides.txt").write_text(
    f"compressed-tensors @ {checkout.as_uri()}\n"
)
PYTHON
BUILD_TYPE=nightly VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python \
  --override osfp4-overrides.txt -e . --torch-backend=auto
uv pip check --python .venv/bin/python

.venv/bin/vllm serve /absolute/path/to/moe-osfp4 \
  --tensor-parallel-size 1 --enforce-eager
```

`uv pip check` reports the known conflict with vLLM's stale `compressed-tensors==0.17.0`
pin; the override leaves that metadata unchanged. Investigate any additional conflicts.
The compressed-tensors fork is installed as a regular build from the checkout.

Use a supported GPU backend: CUTLASS for NVFP4 or Humming for NVFP4A16. See the
[vLLM installation guide](https://github.com/neriahbd/vllm-for-osfp4/tree/add-moe-support/docs/getting_started/installation)
if compatible precompiled artifacts are unavailable. No OSFP4 plugin is needed.
Start with one GPU; expert parallelism, expert load balancing, and fused
shared-expert execution are unsupported. This is a development workflow, not a
fully GPU-validated release recipe.
