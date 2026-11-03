# OSFP4-MoE: integration design

## Repositories

| Repository | Branch |
|---|---|
| [LLMC fork](https://github.com/orimeirgit/llm-compressor-for-osfp4/tree/add-moe-support) | `add-moe-support` |
| [compressed-tensors fork](https://github.com/orimeirgit/compressed-tensors-for-osfp4/tree/osfp4-diagonal-transform) | `osfp4-diagonal-transform` |
| [vLLM fork](https://github.com/neriahbd/vllm-for-osfp4/tree/add-moe-support) | `add-moe-support` |

## MoE branch scope

This note is adapted from `osfp4-design-note-2.md`. The repository links above
select the MoE branches for LLMC and vLLM and the diagonal branch for CT.
The component tables retain the original integration design; they are not a
claim that every listed follow-up has been completed.

The MoE extension resolves joint gate/up and separate down mappings inside
linearized experts. It supports all-expert and routed-token calibration;
routed W4A4 capture uses bounded reservoir sampling. vLLM loads the per-expert
vectors and implements CUTLASS NVFP4 and Humming NVFP4A16 execution paths.
Expert parallelism, expert load balancing, and fused shared experts remain
outside this path. Native GPU validation and matched MoE benchmarks remain
required. See [MoE behavior](moe.md), [setup](getting-started.md), and
[support status](support.md) for the branch-specific workflow.

## 1. Component mapping

| Existing source/component | Upstream API or proposed destination | Required adaptation |
|---|---|---|
| `modifiers/base.py`: modifier orchestration | `modifiers/osfp4/base.py`, using `Modifier` + `QuantizationMixin` | Adapt lifecycle methods and scheme validation to current LLMC; set `requires_calibration_data=True`. Keep mapping resolution, capture lifetime, optimization dispatch, and deployment in the modifier. |
| Quantization lifecycle | `modifiers/quantization/quantization/mixin.py`: `initialize_quantization`, `start_calibration`, `end_calibration` | Reuse setup, observer fusion, and freezing; configure the OSFP4 weight observer and install selected weight parameters explicitly. |
| Sequential calibration | `pipelines/sequential/pipeline.py`, `pipelines/cache.py::IntermediatesCache` | Optimize in `on_sequential_epoch_end`. Reuse subgraph propagation and cache; retain layer-level capture for OSFP4 statistics and optimize each observed mapping once. |
| Smooth/balance layer resolution | SmoothQuant `dynamic_mappings.get_layer_mappings_from_model` and compressed-tensors `match_modules_set` | Reuse existing smooth/balance mappings and matching utilities. Attach explicit smoothing to remaining supported targets. Mapping lookup alone does not validate an architecture; no separate AWQ resolver is added. |
| `modifiers/calibration_cache.py::OSFP4CalibrationCache` | `modifiers/osfp4/calibration_cache.py`, using `HooksMixin.register_hook` | Adapt capture and cleanup to sequential events. Accumulate statistics over all calibration rows; retain selected activation rows where bounded capture is supported, otherwise cache all rows until sampling before optimization. |
| `modifiers/activation_subsampling.py::subsample_activations` | `modifiers/osfp4/activation_subsampling.py` | Preserve seeded sampling and provenance. Count calibration dataloader tokens when possible and select indices during capture; fall back to sampling cached activations when the count or observer does not support bounded capture. |
| `modifiers/optimization/{rtn,sic,_activation_groups}.py` | `modifiers/osfp4/optimization/` | Adapt `optimize_rtn`, `optimize_sic`, and activation grouping without changing their numerical objectives. Call the custom observer's scale-optimization and qparam-selection methods explicitly. |
| `observers/scale_optimization.py`, `loss.py`, `lut.py`, `fp4.py` | `observers/osfp4/{observer,scale_optimization,loss,lut}.py` and compressed-tensors FP4 definitions | Retain `OSFP4Observer` and its numerical utilities in the observer package. Reuse joint optimization, reconstruction loss, and FP4 lookup behavior; the current package has no separate `fp4.py`. |
| `observers/scale_selection.py::_search_e4m3_gamma_star` | `observers/osfp4/scale_selection.py`, exposed through `OSFP4Observer.select_weight_qparams` | Reuse E4M3 selection for RTN and for each error-adjusted target inside SIC's rounding loop. |
| Hessian helpers in `modifiers/optimization/sic.py` | `modifiers/osfp4/optimization/sic.py` | Keep OSFP4-local allocation and FP32 `XᵀX` accumulation; normalize by captured token count before SIC. No GPTQ helper adapter or factor-of-two correction is used. |
| `modifiers/osfp4_quantize.py`, `modifiers/smoothing.py` | `modifiers/osfp4/` with `update_offload_parameter` and `update_qparams(layers, "input")` | Preserve rounded-scale balancing; deploy explicit input scaling through the CT diagonal transform. Reset and replay input observers after smoothing, then install optimized weight parameters. |
| Runtime scale storage | Compressed-tensors `diagonal` factory, using `TransformFactory` and `apply_transform_config` | Attach a permanent input multiply and one validated BF16 vector per runtime target. Export the `smooth_quant` group and `<layer>.smooth_quant_input.weight`; folded mappings need no runtime vector. |
| Reference `tests/modifiers/optimization/`, `tests/observers/` | `tests/llmcompressor/modifiers/osfp4/optimization/` and `tests/llmcompressor/observers/osfp4/` | Adapt algorithm and custom-observer tests to their current locations. Add modifier/input-observer integration checks while retaining numerical reference comparisons. |
| Reference `tests/golden/fixtures/`, `tests/golden/generate_golden.py`, `tests/test_golden.py` | `tests/llmcompressor/modifiers/osfp4/` | Reuse RTN/SIC × NVFP4/NVFP4A16 numerical fixtures. Adapt checkpoint names/metadata separately and test normal saving plus explicit transform-aware CT reload. |

### Modifier versus observer

The modifier orchestrates joint smoothing/scale optimization and discrete
weight-scale selection. Reuse `OSFP4Observer` for weight-scale optimization and
selection, and retain existing input observers for activation calibration.
No additional observer implementation is needed. The weight observer provides
the numerical optimization and selection methods; the modifier owns capture,
RTN/SIC dispatch, deployment, and cleanup. SIC selects scales using each
block's error-adjusted target inside its rounding loop, rather than using
GPTQ's single up-front `get_qparams()` pattern.

The per-mapping sequence is capture, joint optimization, E4M3 selection,
smoothing deployment, input recalibration, weight-parameter installation, and
cache release. SIC installs rounded weights; RTN installs scales and uses the
quantization forward/save path for weight rounding. Round the
smoothing vector to its deployment dtype before final scale selection and
balancing. Runtime scales use BF16. Deploy `weight_global_scale=1` with the
selected E4M3 block scales; preserve that common global value for fused groups.

`QuantizationMixin` creates and fuses the configured observers, including the
OSFP4 weight observer. The optimization routines call its scale-optimization
and qparam-selection methods explicitly. OSFP4 installs the resulting weight
parameters and avoids generic weight-qparam recomputation. Reset input
observers and replay smoothed activations or sufficient full-calibration
channel maxima before `update_qparams(layers, "input")`. Completion must be
idempotent because `end_calibration` can trigger another epoch-end callback.

## 2. Required changes

LLMC paths are relative to `src/llmcompressor/` unless stated otherwise. CT paths
are relative to `src/compressed_tensors/`. vLLM transform paths are relative to
`vllm/model_executor/layers/quantization/compressed_tensors/`.

| First-version file/module | Change |
|---|---|
| LLMC `modifiers/osfp4/{__init__,base}.py` | Add the modifier/public import, current lifecycle integration, scheme validation, mappings, and guards. |
| LLMC `modifiers/osfp4/{calibration_cache,activation_subsampling,osfp4_quantize,smoothing}.py` | Adapt reference capture, sampling, orchestration, and deployment. Accumulate all-token statistics, select activation rows for optimization, balance weights using the rounded scale, and recalibrate input global scales. |
| LLMC `modifiers/osfp4/optimization/` | Adapt RTN/SIC and activation grouping. Keep OSFP4-local Hessian allocation and accumulation; no GPTQ helper adapter is used. |
| LLMC `observers/osfp4/`, `observers/__init__.py` | Add/register the custom weight observer and retain scale optimization, E4M3 selection, loss, and lookup utilities in the observer package. |
| LLMC `modifiers/osfp4/transforms.py` | Construct the input-only diagonal transform config and attach permanent runtime transforms through compressed-tensors. |
| LLMC `modifiers/quantization/calibration.py`, `modifiers/quantization/quantization/mixin.py` | Rename `calibrate_input_hook` to `observe_input_activations_hook` and update its caller. This readability change is present in the fork but is not required for OSFP4 behavior. |
| CT `transform/factory/diagonal.py`, `transform/__init__.py` | Add/register the per-layer diagonal input transform and validate scale shape, stored precision, values, and target association. |
| LLMC `transformers/compression/compressed_tensors_utils.py` | **Not modified.** Reuse `modify_save_pretrained` and `ModelCompressor`; test the normal save path rather than adding an OSFP4-specific serializer. |
| vLLM `transform/module.py` | Add `DiagonalTransform` to load the saved channel vector and apply multiplication or inverse division, using the existing parameter-loading machinery. |
| vLLM `transform/linear.py` | Construct the diagonal module through the existing transform wrapper, validate supported configurations, and apply input scaling before activation quantization. |
| Repository `setup.py` | Current pins remain `compressed-tensors==0.20.1` for releases and `>=0.20.2a2` for nightly builds. Before upstream release, require a version containing the diagonal factory; use the companion fork in the current integration. |
| LLMC modifier/observer tests and fixtures | Adapt reference numerical tests and fixtures; add sequential propagation, finalization, input recalibration, save/reload, and rejected-configuration coverage. |
| CT `tests/test_transform/factory/{test_correctness,test_serialization,test_memory}.py` | Extend the existing transform suites with diagonal correctness, serialization, and memory coverage. |
| vLLM `tests/quantization/test_compressed_tensors.py` | Add a dense diagonal checkpoint generation test alongside existing transform tests, covering normal/inverse scaling and nonzero bias. |
| `docs/`, modifier docstrings | Document the recipe, checkpoint representation, restrictions, and serving dependency. Complete user-facing reload instructions before advertising a public reload route. |

Three upstream constraints need explicit handling:

- **Gradients:** sequential calibration runs under `torch.no_grad()` via
  `calibration_forward_context`. Scale optimization uses `torch.enable_grad()`;
  model weights and captured data remain detached. No shared lifecycle change
  is needed.
- **Cache timing:** count tokens from the calibration dataloader when possible
  and initialize sampling indices during capture. Fall back to full activation
  capture when the count or input observer does not support bounded caching;
  apply the configured subsampling before optimization and test both paths.
- **Hessian normalization:** OSFP4 directly accumulates FP32 `XᵀX` and divides
  by captured token count before SIC. It does not use GPTQ's `√2` pre-scaling
  or require a factor-of-two correction.

The integration spans three upstream contributions: compressed-tensors diagonal
transforms, LLMC calibration and export, and native vLLM loading and execution.
LLMC's runtime-transform support depends on the CT changes. Public Transformers
reload, DDP, and modifier composition remain follow-ups.

## 3. Usage and checkpoint design

### Calibration recipe

Use a BF16 model and the companion compressed-tensors fork for the save/reload
flow below. Assume `ds` contains the calibration dataset and that the model was loaded
inside LLMC `load_context()` for MoE calibration. The ignore patterns below
assume Qwen-style router names; adapt them for the actual architecture.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.osfp4 import OSFP4Modifier

oneshot(
    model=model,
    dataset=ds,
    recipe=OSFP4Modifier(
        scheme="NVFP4",            # or "NVFP4A16"
        optimization_mode="sic",   # or "rtn"
        targets=["Linear"],
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        activation_subsample_size="auto",
    ),
    num_calibration_samples=1024,
    max_seq_length=2048,
    propagate_error=True,
    moe_calibrate_all_experts=True,
)
model.save_pretrained(save_dir, save_compressed=True)
```

The default `independent` dispatcher chooses sequential calibration through
`requires_calibration_data=True`. The propagation guarantees below assume
`propagate_error=True`, which is also upstream's default. With `False`, upstream
passes pre-optimization outputs to subsequent subgraphs; the note does not
claim quantized-output propagation for that mode.

Preserve the reference `"auto"` sampling policy:
`min(total_rows, 2 × input_width)` rows per mapping for W4A4 optimization.
Second moments and input-global-scale statistics still use all calibration
rows. W4A16 requires those moments but no activation-row cache.
For W4A4 with subsampling enabled, when the token count is known and the input
observer supports it, selected rows are retained during capture; otherwise, supported routed MoE experts use bounded reservoir sampling.
Other unsupported capture paths cache activations and subsample before optimization.

### Restrictions and enforcement

| Configuration | Current behavior |
|---|---|
| Validated dense decoder; NVFP4/NVFP4A16; RTN/SIC; single process | Implemented. Architecture validation remains separate from accepting a quantization scheme. |
| Other or mixed schemes, output activation quantization | Reject. |
| Another modifier in the active recipe | Reject when the resolved recipe includes this modifier. |
| Pipeline other than sequential or the independent dispatcher | Reject. |
| Distributed calibration, existing nonempty transforms, layerwise compression/decompression | Reject. |
| Non-Linear target or input width not divisible by 16 | Reject. |
| Remaining target without a folded smooth-layer mapping | Use explicit scaling where the supported mapping and target rules allow it. Mapping resolution alone does not validate an architecture. |
| `kv_cache_scheme` | Not categorically rejected; completion includes optional KV-cache qparam updates. Dedicated validation is needed before declaring broad support. |
| XPU | Outside declared initial support; the modifier does not implement an explicit XPU rejection. |

Run local checks in `on_initialize` before OSFP4 modifies the model. Reuse the
existing SmoothQuant mapping and module-matching utilities; fold scales only
where the selected smooth/balance mapping permits it. No shared Gemma mapping
change or new normalization context is introduced by this integration.

For “OSFP4 only,” inspect the resolved recipe in the active session. LLMC
initializes modifiers in order, so this guard prevents OSFP4's changes but cannot
undo an earlier modifier's changes. A rejected mixed recipe requires a fresh
model. A lifecycle preflight would be additional work if maintainers require
rejection before any modifier initializes. Deployment failures also prevent
reuse of the modifier because model parameters may already be partially changed.

### Recommended checkpoint contract

Use an input-only CT `diagonal` transform with
`quant_method="compressed-tensors"`. This follows the standard transform
configuration and saved-weight structure used by Hadamard transforms and requires
the companion CT diagonal implementation.

The reference `smooth_quant_scale` remains the same activation multiplier `s`:
`y = (W / s) (x · s)`. Store it as `<layer>.smooth_quant_input.weight`, a positive,
finite BF16 vector of length `in_features`, associated with the exact target.
Weight balancing and input calibration use this rounded vector. Packed weights
already include inverse balancing, so reload applies only `x · s`. Folded
mappings update the predecessor and consuming weights and need no runtime vector.
NVFP4's `input_global_scale` is saved separately and controls activation
quantization; it does not replace the per-channel multiplier.

Example `transform_config` inside `quantization_config`:

```json
{
  "config_groups": {
    "smooth_quant": {
      "type": "diagonal",
      "precision": "torch.bfloat16",
      "head_dim": null,
      "randomize": false,
      "requires_grad": false,
      "apply": [{
        "targets": ["model.layers.0.mlp.down_proj"],
        "location": "input",
        "inverse": false,
        "ignore": []
      }]
    }
  }
}
```

Attach identity transforms before calibration, then deploy the learned vector
and balanced weights before sequential propagation. Runtime transforms are
registered by CT separately from temporary modifier capture hooks, so capture
cleanup does not remove them. Smoothing remains active during propagation and
immediately after `oneshot`. W4A4 scales before activation quantization; W4A16
scales before the weight-only Linear operation.

### Save, reload, and serving flow

**Save:** existing `modify_save_pretrained` uses
`ModelCompressor.from_pretrained_model`, `compress_model`, the underlying save,
and `compressor.update_config`. Reuse that flow to save packed parameters,
transform config, and each target's vector through normal
`save_pretrained(save_compressed=True)`. No OSFP4-specific serializer is added.

**Reload:** current checkpoint tests compose existing APIs explicitly:

1. Read the saved quantization metadata and extract `transform_config`. Validate
   the two configurations with `QuantizationConfig.model_validate` and
   `TransformConfig.model_validate`.
2. Construct the same model architecture under `accelerate.init_empty_weights`,
   using the intended compute dtype.
3. Call `apply_quantization_config` and `apply_transform_config`, then
   `ModelCompressor(quantization_config=..., transform_config=...).compress_model`
   to create the packed parameter layout on the meta model.
4. Load the saved safetensors state with
   `load_state_dict(assign=True, strict=True)`, evaluate, and compare outputs
   with the model after calibration and before saving.

Preserve checkpoint tensor dtypes, including BF16 smoothing vectors and E4M3
scales. The input-only configuration never reapplies inverse weight balancing.
The tests cover explicit CT reconstruction with BF16 compute for both schemes
and optimizers. FP32 tests compare against golden fixtures without packed reload.
There is no implemented `load_osfp4_model()` helper, and these tests do not
establish generic public `AutoModelForCausalLM.from_pretrained` support.
User-facing reload instructions and sharded/device-dispatched coverage remain
follow-ups.

**vLLM:** native support is implemented in the MoE fork. It reads the
standard transform configuration and vectors, creates `DiagonalTransform`, and
applies input scaling through the existing Linear wrapper before activation
quantization. No separate OSFP4 plugin is needed. The parameter-loading machinery
supports row-parallel scale slicing; this code alone does not establish TP=2+
GPU validation. Multi-partition fused Linear diagonal transforms are rejected by
this branch. Serving claims should identify the tested backend, hardware, format,
and execution mode.

## 4. Key results

The original design note reports the following external implementation results
from the [paper, Table 1 and Appendix D](https://arxiv.org/html/2610.08231v1).
They are retained here as reference results, not measurements of the current
LLMC integration or newly revalidated paper figures. Scores are reported mean
accuracy across WinoGrande, HellaSwag, GSM8K, and MMLU-CoT for
Llama-3.1-8B-Instruct; the reported BF16 score is 79.22.

| Setting | NVFP4 RTN baseline | GPTQ | NVIDIA checkpoint | OSFP4 RTN | OSFP4 SIC |
|---|---|---|---|---|---|
| W4A4 | 75.65 (FP-Quant) | 76.34 (FP-Quant) | 76.11 | 76.36 | 77.03 |
| W4A16 | 77.67 (authors' implementation) | — | — | 78.15 | 78.42 |

| Requested setup/measurement | Evidence |
|---|---|
| Reference calibration dataset | The original note reports FineWeb-Edu; 1,024 sequences × 2,048 tokens (2,097,152 tokens). Current integration runs need their own dataset revision, exact windows, BOS/tokenization, and padding settings recorded. |
| Optimizer | Current defaults are 80 steps, learning rate 0.12, and SIC damping fraction 0.01, matching the settings reported in the original note. |
| Exact `activation_subsample_size` for published accuracy runs | **Not established by the reviewed run evidence.** The current auto policy does not prove what was used for every published result. |
| Accuracy versus LLMC's NVFP4 baseline | **A matched benchmark table for these fork revisions is not supplied.** |
| RTN/SIC calibration time | **Comparable measurements and hardware/software details are still required.** |
| RTN/SIC peak GPU and CPU memory | **Comparable peak measurements are still required.** Cache size is not whole-process peak memory, and projected growth is not a measured result. |

For W4A4, `activation_subsample_size="auto"` caps reconstruction-loss sampling
at **twice each mapping’s input width**: a 4,096-channel input uses up to 8,192
token rows, or all rows if fewer are available. RTN/SIC statistics still use every
calibration token. W4A16 disables activation subsampling. For each benchmark,
record the policy, resolved cap, sampled and total row counts, and selection
provenance from `activation_subsampling_records`. These defaults do not establish
the sampling settings used for the published results.

Existing tests cover RTN/SIC numerics, smoothing, sequential calibration, and
explicit transform-aware checkpoint reconstruction in LLMC, plus CT transform
coverage. The dense vLLM regression checks equivalent generation from original
and balanced BF16 checkpoints in eager mode, including inverse scaling and bias.
Six additional CPU byte checks passed for saved vectors and transformed
activations across FP32, BF16, and FP16 with normal and inverse scaling. This is
correctness evidence; full-model NVFP4 GPU accuracy and performance remain to be
validated.

Before upstream submission, compare these companion branches against LLMC’s NVFP4
baseline under identical evaluation settings, both after `oneshot` and after
saving and reloading for serving. Publish accuracy, calibration time, peak memory,
and raw logs, with exact calibration samples, sequence lengths, sampling settings,
code revisions, and environment versions.

## 5. Integration support matrix

| Dimension | First version / follow-up |
|---|---|
| Dense | Architecture-specific validation remains required. |
| MoE | Linearized expert calibration and per-expert serving transforms implemented; see [support status](support.md). |
| NVFP4 / NVFP4A16 | Both implemented. |
| RTN / SIC | Both implemented. |
| CUDA | Primary calibration target; GPU-specific validation required. |
| CPU | Correctness tests and explicit CT reload/evaluation tests. |
| XPU | Outside initial support; not validated or explicitly rejected. |
| Single-process / DDP | Single process / DDP rejected. |
| Explicit CT reload | Tested through existing CT APIs; no reload helper. Sharded/device-dispatched loading remains a follow-up. |
| Public Transformers reload | Follow-up. |
| vLLM / tensor-parallel serving | Native support implemented. TP shards scales for row-parallel inputs and replicates them for column-parallel layers; TP=2+ GPU validation remains outstanding. |

Maintainer decisions: agree on the input-only transform checkpoint contract and
compressed-tensors dependency, the modifier–observer boundary, and where
configuration restrictions should be validated. These decisions define the first
upstream PR’s scope.
