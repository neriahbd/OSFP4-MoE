# Support and validation status

This page describes the source snapshot in [revisions.tsv](../revisions.tsv). An implemented path is not a claim of completed GPU validation.

| Area | Status |
|---|---|
| OSFP4 formats | NVFP4 W4A4 and NVFP4A16 W4A16 |
| Optimization | RTN and SIC |
| Calibration | Single process; sequential pipeline or independent dispatcher selecting sequential |
| MoE calibration | Linearized expert mappings; all-expert or routed-token calibration |
| Expert vectors | Joint gate/up and separate down; stored with original projection names |
| MoE serving | CUTLASS NVFP4 and Humming NVFP4A16 paths implemented |
| Tensor parallelism | Vector sharding implemented; native GPU validation remains required |
| Expert parallelism / expert load balancing | Not implemented for this transformed-expert path |
| Fused shared experts | Not implemented; ordinary Linear shared experts use the dense path |
| Distributed OSFP4 calibration | Rejected |
| Other or mixed quantization schemes / output activation quantization | Rejected by the modifier |
| Composition with other modifiers | Rejected; retry with a fresh model after partial initialization or deployment failure |
| Existing nonempty transforms / layerwise compression-decompression | Rejected |
| Non-Linear targets / input widths not divisible by 16 | Rejected |
| Explicit compressed-tensors reconstruction | Covered by fork tests; not a generic Transformers reload promise |
| Public Transformers `from_pretrained` reload | Not established by this integration |
| GPU architecture / memory requirements | Backend- and model-dependent; no universal GPU or VRAM claim |
| End-to-end MoE accuracy / throughput / calibration peak memory | No matched benchmark results supplied here |

The LLMC tests include small Qwen3-MoE configurations. These exercise behavior; they do not certify every Qwen model, a pretrained 30B checkpoint, or unrelated MoE architectures.

## Validation commands

Run in the appropriate fork's configured environment after installing its test dependencies. These commands are directions for validation, not a record of tests executed when preparing this documentation.

LLM Compressor:

```bash
.venv/bin/python -m pytest -v tests/llmcompressor/modifiers/osfp4/test_moe.py
```

compressed-tensors:

```bash
.venv/bin/python -m pytest -v tests/test_transform/factory/test_correctness.py tests/test_transform/factory/test_serialization.py tests/test_transform/factory/test_memory.py
```

vLLM, with two NVIDIA GPUs supported by the selected backend:

```bash
.venv/bin/python -m pytest -v tests/quantization/test_compressed_tensors_moe_transforms.py tests/quantization/test_moe_diagonal_loading.py
```

The vLLM suite covers packed checkpoints, TP=1/2, prefill/decode, eager/CUDA graph execution, and CPU loader checks. Review skips and actual device coverage rather than treating an aggregate pass as proof that every GPU path ran.

## Recording a model evaluation

Record the three code commits; model and dataset revisions; exact calibration samples and tokenization; scheme and optimizer; expert calibration mode; activation sampling policy and resolved counts; GPU, CUDA, driver, PyTorch and package versions; backend and parallelism; accuracy before and after export; and measured calibration time, peak memory and serving performance.

The dense Llama reference table in [the design note](design.md) is external reference evidence. It is not a MoE benchmark or a measurement of these three revisions.
