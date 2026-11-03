# Setup, calibration, and serving

This is a development workflow derived from the companion branches. The commands have not been exercised end to end on a GPU as part of this documentation. Check [support](support.md), the model's expert layout, and the selected backend's hardware requirements first.

## 1. Get the matching sources

From the root of this documentation repository:

```bash
bash scripts/checkout-forks.sh
```

This creates `sources/llm-compressor`, `sources/compressed-tensors`, and `sources/vllm` at the commits in [revisions.tsv](../revisions.tsv). To follow later work, use the named branches in [the README](../README.md) and record the new commits.

## 2. Prepare the calibration environment

Use a machine with sufficient CPU/GPU memory for the original BF16 model and calibration workspace. Quantized checkpoint size is not the memory required to calibrate it.

From this repository's root, with `uv` installed:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  -e ./sources/compressed-tensors -e ./sources/llm-compressor
uv pip check --python .venv/bin/python
.venv/bin/python -c 'import compressed_tensors, llmcompressor; print(compressed_tensors.__file__); print(llmcompressor.__file__)'
```

Install the appropriate CUDA-enabled PyTorch build following the fork's setup instructions if your environment does not already provide it. The local forks must satisfy the selected package constraints together. If resolution fails, inspect the fork requirements and build versions; do not silently substitute stock compressed-tensors or bypass dependencies with `--no-deps`.

The printed import paths should point to your local sources. Environment resolution, CUDA setup, and model-specific dependencies remain part of validating a deployment; the source revision list is not a Python dependency lock.

## 3. Prepare and calibrate a model

The following template uses Qwen-style MoE names. Set `MODEL_ID` to the exact model path or revision you intend to evaluate and provide `calibration.jsonl` with representative `{"text": "..."}` records. This is an adaptation template, not a claim that every model accepted by Transformers is supported.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.osfp4 import OSFP4Modifier
from llmcompressor.utils import load_context

MODEL_ID = "/path/to/your/bf16-moe-model"
SAVE_DIR = "checkpoints/moe-osfp4-nvfp4"
NUM_SAMPLES = 1024
MAX_LENGTH = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    )

ds = load_dataset("json", data_files="calibration.jsonl", split="train")
ds = ds.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(ds))))

def tokenize(sample):
    return tokenizer(
        sample["text"], padding=False, truncation=True,
        max_length=MAX_LENGTH, add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

oneshot(
    model=model,
    dataset=ds,
    recipe=OSFP4Modifier(
        scheme="NVFP4",          # Or NVFP4A16 for W4A16.
        optimization_mode="sic", # Or rtn.
        targets=["Linear"],
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        activation_subsample_size="auto",
    ),
    num_calibration_samples=len(ds),
    max_seq_length=MAX_LENGTH,
    propagate_error=True,
    moe_calibrate_all_experts=True,
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

Save the template as a local Python file, fill in your inputs, and run it with `.venv/bin/python`. Use nonempty calibration data. Apply the model's chat template and BOS/tokenization conventions before tokenization when appropriate; the plain-text example cannot select those conventions for every model.

`load_context()` prepares the MoE representation for calibration. Confirm that the selected architecture exposes supported linearized expert projections and that your router exclusions match its actual names. Do not quantize the router merely because it is a Linear module. Do not exclude `gate_proj` by confusing it with the router.

Use `moe_calibrate_all_experts=False` only when intentionally evaluating routed-token calibration and recording expert coverage. Keep model revision, calibration rows, sequence length, tokenization, optimization settings and sampling records alongside the exported checkpoint.

## 4. Inspect the export

The output should contain model configuration, packed weights, tokenizer files, and transform metadata. For expert mappings that need runtime scaling, inspect both:

- The `smooth_quant` diagonal group inside `quantization_config.transform_config`.
- Per-projection keys such as `model.layers.0.mlp.experts.0.down_proj.smooth_quant_input.weight` in the checkpoint tensors.

Preserve the original tensor names and dtypes. Folded dense mappings do not require a runtime vector. `input_global_scale` and the per-channel vector have different roles; neither can replace the other.

The [design note](design.md) explains explicit transform-aware CT reconstruction. Do not assume stock `AutoModelForCausalLM.from_pretrained` can reload every exported OSFP4 checkpoint.

## 5. Prepare vLLM and generate

Use a separate serving environment. From the documentation repository root:

```bash
cd sources/vllm
uv venv --python 3.12 .venv
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python \
  -e . -e ../compressed-tensors --torch-backend=auto
uv pip check --python .venv/bin/python
```

Follow the [vLLM fork's installation documentation](https://github.com/neriahbd/vllm-for-osfp4/tree/add-moe-support/docs/getting_started/installation) for your GPU and CUDA environment. The precompiled editable path depends on compatible build artifacts; if they are unavailable or incompatible with the checkout, use the fork's documented source build workflow. Humming W4A16 also needs the Humming backend dependencies described by vLLM; installing the Python package alone is not evidence of backend availability.

Run an initial single-GPU generation from this serving environment, using an absolute checkpoint path:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/absolute/path/to/checkpoints/moe-osfp4-nvfp4",
    tensor_parallel_size=1,
    enforce_eager=True,
)
outputs = llm.generate(
    ["Explain what a mixture-of-experts model does."],
    SamplingParams(temperature=0, max_tokens=64),
)
print(outputs[0].outputs[0].text)
```

Automatic backend selection chooses the supported transformed-expert path. Check startup logs for the actual backend. Do not force a different backend merely to suppress a compatibility error. Start with eager, single-GPU execution before evaluating CUDA graphs or tensor parallelism; expert parallelism is not supported for this path.

For an HTTP server, after generation is validated:

```bash
.venv/bin/vllm serve /absolute/path/to/checkpoints/moe-osfp4-nvfp4 \
  --tensor-parallel-size 1 --enforce-eager
```

No separate OSFP4 plugin is required. A plausible sample response is only a smoke check; run the [validation and model evaluation](support.md) before making deployment or accuracy claims.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Unknown OSFP4 modifier or diagonal transform | Import paths, installed fork, branch/commit, and accidental stock-package replacement |
| Missing vectors or shape mismatch | Export metadata, projection names, saved tensor keys, model architecture, and TP slicing |
| Gate/up vector mismatch | Export must use one shared input vector for fused gate/up projections |
| Incompatible backend | Format, installed backend dependencies, GPU support, and explicit backend overrides |
| Unobserved expert during calibration | Router coverage, dataset, and all-expert versus routed calibration |
| Calibration runs out of memory | Original model residency, sequence count/length, sampling policy, SIC statistics and workspace |
| Generic Transformers reload fails | Use the documented explicit CT reconstruction or native vLLM path; generic reload is not established |

For implementation errors, report them in the responsible fork with exact revisions and the failing command.
