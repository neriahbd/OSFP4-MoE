# How OSFP4 works for MoE

## Smoothing and quantization

For a linear projection with weight matrix `W` and input `x`, a channel vector `s` permits the equivalent floating-point expression:

```text
y = W x = (W / s) (x * s)
```

Division operates on weight columns, and multiplication operates on input channels. OSFP4 optimizes smoothing and quantization scales to reduce reconstruction error. Quantization introduces approximation, so the identity alone is not an accuracy guarantee.

The deployed runtime vector is rounded to BF16 before final balancing and scale selection. The checkpoint already contains the balanced, quantized weights. Inference applies the input transform; it must not divide the weights a second time.

## Why experts need separate vectors

A router sends tokens to different experts. Those experts have different weights and activation distributions. The MoE branch resolves OSFP4 mappings within each linearized expert:

- `gate_proj` and `up_proj` form a joint mapping with a shared input vector.
- `down_proj` forms its own mapping with a separate vector.
- Ungated up/down experts can also be represented by the serving loader.

The model's router is distinct from an expert's `gate_proj`. Keep router layers outside the quantization targets. The example ignore expressions in this guide are for Qwen-style names; other architectures require their own selection rules.

## Calibration choices

LLM Compressor's `moe_calibrate_all_experts=True` makes every expert observe all calibration tokens. With `False`, experts observe their routed tokens. Routed calibration can leave experts with insufficient or no observations; inspect coverage and use representative data.

For W4A4, OSFP4 keeps activation samples for reconstruction-loss optimization. The `"auto"` policy caps selected rows at twice the mapping's input width, or the number of available rows if smaller. Routed experts have unknown token counts ahead of time, so the MoE branch uses bounded reservoir sampling. Statistics still accumulate over all observed calibration rows. For W4A16, the algorithm uses statistics without retaining an activation-row cache.

RTN and SIC both optimize OSFP4 parameters. SIC additionally uses second-order information and error-adjusted weight targets during rounding. Its Hessian statistics and optimization workspace still consume memory: bounded activation sampling is not a bound on total process memory.

## Checkpoint to serving

A runtime vector is stored under a key such as:

```text
model.layers.0.mlp.experts.0.down_proj.smooth_quant_input.weight
```

The checkpoint's `quantization_config` includes an input-only `diagonal` transform in `transform_config`. A vector is positive, finite, BF16, and associated with the exact projection input width. NVFP4 also saves `input_global_scale` separately; it is not a substitute for the channel vector.

vLLM preserves the per-expert association while loading and routing. Gate/up vectors must agree for the fused gate/up input. For NVFP4, the vector is applied before activation quantization. For NVFP4A16, activations remain in the model's 16-bit dtype.

The initial transformed-expert paths are native CUTLASS for NVFP4 and grouped Humming experts for NVFP4A16. Automatic selection chooses the applicable path; an explicitly incompatible backend raises an error. Shared experts running as ordinary linear layers use the dense transform path. Fused shared-expert execution is outside the current MoE implementation.

See [support](support.md) for parallelism limits and [the design](design.md) for the exact representation and explicit CT reconstruction flow.
