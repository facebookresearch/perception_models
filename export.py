"""
Export PE (Perception Encoder) encode_image / encode_text to separate TensorRT
engines with dynamic batch size, verify numerical parity vs PyTorch, and
benchmark both.

Usage:
    python export.py <checkpoint_path> \
        --config PE-Core-L14-336 \
        --precision fp16             # fp16 | fp32 | both \
        --min-batch 1 --opt-batch 4 --max-batch 8 \
        --output-dir trt_out \
        --benchmark-iters 50 --warmup-iters 10

Requirements (in your PE/CUDA environment, not this sandbox):
    pip install tensorrt onnx onnxsim

Notes / assumptions you should sanity-check against your PE version:
  1. model.encode_image(image) / model.encode_text(text) each return a single
     tensor. If your version returns a tuple, the wrappers below already take
     element [0] defensively -- check that's actually the feature tensor you want.
  2. Text token ids are exported as int32 (cast to long inside the wrapper).
     TensorRT's ONNX parser is more reliable with int32 embedding indices than
     int64 on some TRT versions. If your tokenizer needs int64 semantics that
     matter (ids > 2^31), switch DTYPE_TEXT below.
  3. Some PE builds use fused/flash attention kernels that don't trace to ONNX
     cleanly. If torch.onnx.export fails inside attention, look for a flag like
     model.set_attn_implementation("sdpa") / attn_implementation="eager" in the
     pe package and set it before export (eager/sdpa attention exports fine;
     flash-attn custom kernels typically do not).
  4. FP16 is handled by TensorRT's builder flag (BuilderFlag.FP16), not by
     exporting an FP16 ONNX graph. This is the standard/robust approach:
     one FP32 ONNX graph, TRT decides per-layer precision when FP16 is enabled.
"""

import argparse
import copy
import os
import time
import types

import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# --------------------------------------------------------------------------
# Export-friendly patch for AttentionPooling
# --------------------------------------------------------------------------
# AttentionPooling (used for vision pool_type="attn", and optionally text
# pool_type="attn"/"attn_eos") wraps an nn.MultiheadAttention. That module's
# internal fast-path eligibility logic runs a data-dependent check that
# torch.export/ONNX export can't trace (DataDependentOutputException on
# aten.equal.default). Disabling torch.backends.mha's fast path doesn't
# reliably avoid this since the check can run before the fast/slow branch
# decision either way.
#
# The fix: replace AttentionPooling.forward on an export-only copy of the
# model with a manual recomputation using the exact same fused
# in_proj_weight/in_proj_bias/out_proj parameters and F.scaled_dot_product_
# attention (same op VisionTransformer's own SelfAttention class already
# uses elsewhere in this file). Numerically identical to nn.MultiheadAttention
# with batch_first=True, need_weights=False, no masks, no bias_k/bias_v,
# add_zero_attn=False -- all true for how AttentionPooling calls it here.
def _manual_mha(
    attn: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    embed_dim = attn.embed_dim
    num_heads = attn.num_heads
    head_dim = embed_dim // num_heads

    w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
    if attn.in_proj_bias is not None:
        b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)
    else:
        b_q = b_k = b_v = None

    q = F.linear(query, w_q, b_q)
    k = F.linear(key, w_k, b_k)
    v = F.linear(value, w_v, b_v)

    B, Tq, _ = q.shape
    Tk = k.shape[1]
    q = q.reshape(B, Tq, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(B, Tk, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(B, Tk, num_heads, head_dim).transpose(1, 2)

    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    attn_out = attn_out.transpose(1, 2).reshape(B, Tq, embed_dim)
    return F.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)


def _export_attention_pooling_forward(self, x: torch.Tensor) -> torch.Tensor:
    batch = x.shape[0]
    q = self.probe.repeat((batch, 1, 1)).to(x.dtype)
    x_out = _manual_mha(self.attn, q, x, x)
    x_out = x_out + self.mlp(self.layernorm(x_out))
    return x_out


def _export_call_attn(self, q_x: torch.Tensor, attn_mask=None):
    # Mirrors ResidualAttentionBlock._call_attn's SelfAttention branch, but
    # for the nn.MultiheadAttention branch (used whenever rope=None, e.g. all
    # of TextTransformer's blocks). attn_mask here is the additive/bool causal
    # mask; F.scaled_dot_product_attention accepts either directly.
    if isinstance(self.attn, nn.MultiheadAttention):
        embed_dim = self.attn.embed_dim
        num_heads = self.attn.num_heads
        head_dim = embed_dim // num_heads

        w_q, w_k, w_v = self.attn.in_proj_weight.chunk(3, dim=0)
        if self.attn.in_proj_bias is not None:
            b_q, b_k, b_v = self.attn.in_proj_bias.chunk(3, dim=0)
        else:
            b_q = b_k = b_v = None

        q = F.linear(q_x, w_q, b_q)
        k = F.linear(q_x, w_k, b_k)
        v = F.linear(q_x, w_v, b_v)

        B, T, _ = q.shape
        q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2)

        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = (
                attn_mask if attn_mask.dtype == torch.bool else attn_mask.to(q.dtype)
            )

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, T, embed_dim)
        return F.linear(attn_out, self.attn.out_proj.weight, self.attn.out_proj.bias)
    else:
        return self.attn(q_x, attn_mask=attn_mask)


def make_export_friendly(model):
    """Deep-copy the model and patch AttentionPooling and any
    ResidualAttentionBlock using nn.MultiheadAttention (i.e. built without
    rope -- true for all of TextTransformer's blocks here) to use a traceable
    manual attention computation instead of nn.MultiheadAttention's internal
    fast path. The original `model` is left untouched so it can still be used
    as the PyTorch reference for verification/benchmarking -- if either patch
    were numerically wrong, the verify step's cos_sim/max_abs_diff check would
    catch it."""
    from core.vision_encoder.pe import AttentionPooling, ResidualAttentionBlock

    export_model = copy.deepcopy(model)
    n_pool, n_block = 0, 0
    for module in export_model.modules():
        if isinstance(module, AttentionPooling):
            module.forward = types.MethodType(_export_attention_pooling_forward, module)
            n_pool += 1
        elif isinstance(module, ResidualAttentionBlock) and isinstance(
            module.attn, nn.MultiheadAttention
        ):
            module._call_attn = types.MethodType(_export_call_attn, module)
            n_block += 1
    print(
        f"[export] patched {n_pool} AttentionPooling and {n_block} ResidualAttentionBlock "
        f"module(s) (nn.MultiheadAttention -> traceable manual attention)"
    )
    return export_model


TRT_DTYPE_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


# --------------------------------------------------------------------------
# Model wrappers -- one nn.Module per exported graph, single tensor in/out
# --------------------------------------------------------------------------
class ImageEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(image)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats.contiguous()


class TextEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_text(text_ids.long())
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats.contiguous()


# --------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------
def export_onnx(
    wrapper,
    dummy_input,
    onnx_path,
    input_name,
    output_name,
    min_batch,
    max_batch,
    opset=18,
):
    wrapper.eval()

    # Modern torch defaults torch.onnx.export to the dynamo-based exporter,
    # which wants 'dynamic_shapes' (torch.export.Dim), not the legacy
    # 'dynamic_axes' dict. Passing dynamic_axes under dynamo=True relies on an
    # internal best-effort conversion that can silently bake dims as
    # constants -- this is the likely cause of the earlier baked-batch
    # Reshape bug. Use dynamic_shapes directly instead.
    batch_dim = torch.export.Dim("batch", min=min_batch, max=max_batch)
    dynamic_shapes = ({0: batch_dim},)  # matches args=(dummy_input,) pytree structure

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_input,),
                onnx_path,
                input_names=[input_name],
                output_names=[output_name],
                dynamic_shapes=dynamic_shapes,
                opset_version=opset,
                dynamo=True,
                do_constant_folding=True,
            )
        print(f"[onnx] exported {onnx_path} (dynamo exporter, dynamic_shapes)")
    except Exception as e:
        print(
            f"[onnx] dynamo export failed ({e!r}), falling back to legacy TorchScript exporter"
        )
        dynamic_axes = {input_name: {0: "batch"}, output_name: {0: "batch"}}
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_input,
                onnx_path,
                input_names=[input_name],
                output_names=[output_name],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                dynamo=False,
                do_constant_folding=True,
            )
        print(
            f"[onnx] exported {onnx_path} (legacy exporter, dynamic_axes) -- "
            f"double check dynamic batch with find_static_reshapes.py, this path "
            f"is more prone to baking constants"
        )

    # Optional simplification pass -- helps TensorRT's parser on transformer
    # graphs full of Reshape/Slice/Constant chains. Non-fatal if unavailable.
    try:
        import onnx
        from onnxsim import simplify

        model_onnx = onnx.load(onnx_path)
        model_simplified, ok = simplify(model_onnx)
        if ok:
            onnx.save(model_simplified, onnx_path)
            print(f"[onnx] simplified {onnx_path}")
        else:
            print(f"[onnx] simplification check failed, keeping original graph")
    except ImportError:
        print(
            "[onnx] onnxsim not installed, skipping simplification (pip install onnxsim)"
        )


# --------------------------------------------------------------------------
# TensorRT engine build
# --------------------------------------------------------------------------
def build_engine(
    onnx_path,
    engine_path,
    precision,
    input_name,
    min_shape,
    opt_shape,
    max_shape,
    workspace_gb=4,
    force=False,
):
    if os.path.exists(engine_path) and not force:
        print(f"[trt] reusing cached engine {engine_path}")
        return engine_path

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Use parse_from_file rather than parse(f.read()): large PE checkpoints
    # export with external-data weights (a companion <name>.onnx.data file),
    # and parse_from_file resolves that file relative to onnx_path's directory.
    # parser.parse(bytes) has no path context and fails to find it.
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[trt] warning: platform does not report fast fp16 support")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision}")

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(
        f"[trt] building {precision} engine: {engine_path} "
        f"(min={min_shape}, opt={opt_shape}, max={max_shape}) -- this can take a few minutes"
    )
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"[trt] wrote {engine_path}")
    return engine_path


# --------------------------------------------------------------------------
# TensorRT runtime wrapper (torch-native, no pycuda dependency)
# --------------------------------------------------------------------------
class TRTRunner:
    def __init__(self, engine_path, device="cuda:0"):
        self.device = device
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        assert self.input_name and self.output_name, (
            "engine must have exactly one input and one output"
        )

        self.input_dtype = TRT_DTYPE_TO_TORCH[
            self.engine.get_tensor_dtype(self.input_name)
        ]
        self.output_dtype = TRT_DTYPE_TO_TORCH[
            self.engine.get_tensor_dtype(self.output_name)
        ]
        self.stream = torch.cuda.Stream(device=device)

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(
            device=self.device, dtype=self.input_dtype
        ).contiguous()
        self.context.set_input_shape(self.input_name, tuple(input_tensor.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output_tensor = torch.empty(
            out_shape, dtype=self.output_dtype, device=self.device
        )

        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, output_tensor.data_ptr())

        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return output_tensor


# --------------------------------------------------------------------------
# Verification + benchmarking
# --------------------------------------------------------------------------
def compare_outputs(a: torch.Tensor, b: torch.Tensor):
    a = a.float().cpu()
    b = b.float().cpu()
    max_abs_diff = (a - b).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
    rel_err = ((a - b).abs() / (a.abs() + 1e-6)).mean().item()
    return max_abs_diff, cos_sim, rel_err


def benchmark_fn(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000.0  # ms/iter


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_path")
    p.add_argument("--config", default="PE-Core-L14-336")
    p.add_argument("--precision", choices=["fp16", "fp32", "both"], default="fp16")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=4)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--output-dir", default="trt_out")
    p.add_argument("--benchmark-iters", type=int, default=50)
    p.add_argument("--warmup-iters", type=int, default=10)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--opset", type=int, default=18)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"

    import core.vision_encoder.pe as pe

    print(f"[load] {args.config} from {args.checkpoint_path}")
    model = pe.CLIP.from_config(
        args.config, pretrained=True, checkpoint_path=args.checkpoint_path
    )
    model = model.to(device).eval()

    image_size = model.image_size
    context_length = model.context_length
    vocab_size = 49408  # matches original script's dummy text range

    export_model = make_export_friendly(model)
    image_wrapper = ImageEncoderWrapper(export_model).eval()
    text_wrapper = TextEncoderWrapper(export_model).eval()

    # Reference wrappers on the *original*, unpatched model -- this is what
    # TRT output gets verified against, so any numerical drift introduced by
    # the AttentionPooling patch would show up as a bad verify() result.
    ref_image_wrapper = ImageEncoderWrapper(model).eval()
    ref_text_wrapper = TextEncoderWrapper(model).eval()

    # ---- ONNX export (once, FP32 graph; TRT builder handles fp16 casting) ----
    # With native dynamic_shapes (torch.export.Dim) the export-time batch value
    # itself shouldn't matter for correctness -- Dim forces the tracer to treat
    # it symbolically rather than folding it into a constant. Still using
    # export_batch=min_batch here for a realistic min-shape trace.
    export_batch = args.min_batch

    image_onnx = os.path.join(args.output_dir, "image_encoder.onnx")
    dummy_image = torch.randn(
        export_batch, 3, image_size, image_size, dtype=torch.float32, device=device
    )
    export_onnx(
        image_wrapper,
        dummy_image,
        image_onnx,
        "image",
        "image_features",
        min_batch=args.min_batch,
        max_batch=args.max_batch,
        opset=args.opset,
    )

    text_onnx = os.path.join(args.output_dir, "text_encoder.onnx")
    dummy_text = torch.randint(
        0, vocab_size, (export_batch, context_length), dtype=torch.int32, device=device
    )
    export_onnx(
        text_wrapper,
        dummy_text,
        text_onnx,
        "text",
        "text_features",
        min_batch=args.min_batch,
        max_batch=args.max_batch,
        opset=args.opset,
    )

    precisions = ["fp16", "fp32"] if args.precision == "both" else [args.precision]
    batch_sizes_to_test = sorted(set([args.min_batch, args.opt_batch, args.max_batch]))

    rows = []
    for precision in precisions:
        # ---- Build engines ----
        image_engine_path = os.path.join(
            args.output_dir, f"image_encoder_{precision}.engine"
        )
        build_engine(
            image_onnx,
            image_engine_path,
            precision,
            "image",
            min_shape=(args.min_batch, 3, image_size, image_size),
            opt_shape=(args.opt_batch, 3, image_size, image_size),
            max_shape=(args.max_batch, 3, image_size, image_size),
            force=args.force_rebuild,
        )

        text_engine_path = os.path.join(
            args.output_dir, f"text_encoder_{precision}.engine"
        )
        build_engine(
            text_onnx,
            text_engine_path,
            precision,
            "text",
            min_shape=(args.min_batch, context_length),
            opt_shape=(args.opt_batch, context_length),
            max_shape=(args.max_batch, context_length),
            force=args.force_rebuild,
        )

        image_runner = TRTRunner(image_engine_path)
        text_runner = TRTRunner(text_engine_path)

        # PyTorch reference path: fp16 autocast for the fp16 case (matches the
        # original script's warmup), plain fp32 otherwise.
        use_autocast = precision == "fp16"

        for bs in batch_sizes_to_test:
            # ---------------- image ----------------
            img_input = torch.randn(
                bs, 3, image_size, image_size, dtype=torch.float32, device=device
            )

            def torch_image_fn(img_input=img_input):
                with (
                    torch.no_grad(),
                    torch.autocast("cuda", dtype=torch.float16, enabled=use_autocast),
                ):
                    return ref_image_wrapper(img_input)

            pt_out = torch_image_fn()
            trt_out = image_runner.infer(img_input)
            max_abs, cos, rel = compare_outputs(pt_out, trt_out)

            pt_ms = benchmark_fn(
                lambda: torch_image_fn(),
                iters=args.benchmark_iters,
                warmup=args.warmup_iters,
            )
            trt_ms = benchmark_fn(
                lambda: image_runner.infer(img_input),
                iters=args.benchmark_iters,
                warmup=args.warmup_iters,
            )

            rows.append(
                dict(
                    modality="image",
                    precision=precision,
                    batch=bs,
                    pt_ms=pt_ms,
                    trt_ms=trt_ms,
                    speedup=pt_ms / trt_ms,
                    max_abs_diff=max_abs,
                    cos_sim=cos,
                    rel_err=rel,
                )
            )

            # ---------------- text ----------------
            txt_input = torch.randint(
                0, vocab_size, (bs, context_length), dtype=torch.int32, device=device
            )

            def torch_text_fn(txt_input=txt_input):
                with (
                    torch.no_grad(),
                    torch.autocast("cuda", dtype=torch.float16, enabled=use_autocast),
                ):
                    return ref_text_wrapper(txt_input)

            pt_out = torch_text_fn()
            trt_out = text_runner.infer(txt_input)
            max_abs, cos, rel = compare_outputs(pt_out, trt_out)

            pt_ms = benchmark_fn(
                lambda: torch_text_fn(),
                iters=args.benchmark_iters,
                warmup=args.warmup_iters,
            )
            trt_ms = benchmark_fn(
                lambda: text_runner.infer(txt_input),
                iters=args.benchmark_iters,
                warmup=args.warmup_iters,
            )

            rows.append(
                dict(
                    modality="text",
                    precision=precision,
                    batch=bs,
                    pt_ms=pt_ms,
                    trt_ms=trt_ms,
                    speedup=pt_ms / trt_ms,
                    max_abs_diff=max_abs,
                    cos_sim=cos,
                    rel_err=rel,
                )
            )

    print_summary(rows)


def print_summary(rows):
    header = f"{'modality':<8} {'prec':<5} {'batch':<6} {'pt_ms':<9} {'trt_ms':<9} {'speedup':<8} {'max|diff|':<11} {'cos_sim':<9} {'rel_err':<8}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['modality']:<8} {r['precision']:<5} {r['batch']:<6} "
            f"{r['pt_ms']:<9.3f} {r['trt_ms']:<9.3f} {r['speedup']:<8.2f} "
            f"{r['max_abs_diff']:<11.5f} {r['cos_sim']:<9.5f} {r['rel_err']:<8.5f}"
        )
    print()
    print(
        "Guidance: for fp16, cos_sim should be >0.999 and max|diff| small relative to"
    )
    print(
        "feature magnitude -- exact tolerance depends on your model, but cos_sim < 0.99"
    )
    print("or growing rel_err at larger batch usually means a numerics or export bug,")
    print("not just expected fp16 rounding.")


if __name__ == "__main__":
    """
    Results :
    modality prec  batch  pt_ms     trt_ms    speedup  max|diff|   cos_sim   rel_err 
    ---------------------------------------------------------------------------------
    image    fp16  1      94.785    12.934    7.33     0.01953     0.99999   0.03408 
    text     fp16  1      49.927    2.786     17.92    0.09790     0.99983   0.07159 
    image    fp16  4      95.027    40.606    2.34     0.03516     0.99999   0.05386 
    text     fp16  4      51.038    3.112     16.40    0.33594     0.99982   0.12761 
    image    fp16  8      134.446   85.219    1.58     0.02734     0.99999   0.52988 
    text     fp16  8      54.217    5.663     9.57     0.45312     0.99985   0.10726 
    """
    main()
