#!/usr/bin/env python3
"""Compile one experiment kernel through DynamicCV and extract PlanComputeBlock IR."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

KERNEL_FUNCTIONS = {
    "fused_attention": ("_attn_fwd_inner", "_attn_fwd"),
    "flash_attention_npu_v8": ("load_if", "store_if", "mask_fn", "fwd_kernel"),
    "hstu_attention": ("_hstu_attn_fwd_one_block", "_hstu_attn_fwd_compute", "_hstu_attn_fwd"),
    "unified_attention": ("cdiv_fn", "apply_softcap", "kernel_unified_attention_2d"),
}


def kernel_config(name: str):
    if name == "fused_attention":
        stride = (2097152, 65536, 64, 1)
        signature = {
            "Q": "*bf16",
            "K": "*bf16",
            "V": "*bf16",
            "M": "*fp32",
            "Out": "*bf16",
            "acc": "*fp32",
            "sm_scale": "fp32",
        }
        constants = {
            "stride_qz": stride[0],
            "stride_qh": stride[1],
            "stride_qm": stride[2],
            "stride_qk": stride[3],
            "stride_kz": stride[0],
            "stride_kh": stride[1],
            "stride_kn": stride[2],
            "stride_kk": stride[3],
            "stride_vz": stride[0],
            "stride_vh": stride[1],
            "stride_vn": stride[2],
            "stride_vk": stride[3],
            "stride_oz": stride[0],
            "stride_oh": stride[1],
            "stride_om": stride[2],
            "stride_on": stride[3],
            "Z": 4,
            "H": 32,
            "N_CTX": 1024,
            "HEAD_DIM": 64,
            "BLOCK_M": 64,
            "BLOCK_N": 128,
            "STAGE": 1,
        }
        return "_attn_fwd", signature, constants

    if name == "flash_attention_npu_v8":
        signature = {
            "q_ptr": "*bf16",
            "k_ptr": "*bf16",
            "v_ptr": "*bf16",
            "o_ptr": "*bf16",
            "l_ptr": "*fp32",
            "q_attn_arg_ptr": "*i32",
            "k_attn_arg_ptr": "*i32",
            "mask_tensor_ptr": "*i1",
            "cu_seqlens_q": "*i32",
            "cu_seqlens_k": "*i32",
            "q_head": "i32",
            "kv_head": "i32",
            "scale": "fp32",
        }
        constants = {
            "QK_DIM": 64,
            "V_DIM": 64,
            "MASK_FN": 1,
            "SPARSE_OPT": False,
            "DTYPE": 14,
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "AICORE_NUM": 40,
            "MAX_Q_LEN": 1024,
            "MAX_K_LEN": 1024,
            "BATCH_SIZE": 2,
        }
        return "fwd_kernel", signature, constants

    if name == "hstu_attention":
        signature = {
            "Q": "*fp32",
            "K": "*fp32",
            "V": "*fp32",
            "seq_offsets": "*i64",
            "Out": "*fp32",
            "mask": "*fp32",
        }
        constants = {
            "stride_qm": 64,
            "stride_qh": 32,
            "stride_kn": 64,
            "stride_kh": 32,
            "stride_vn": 64,
            "stride_vh": 32,
            "stride_om": 64,
            "stride_oh": 32,
            "alpha": 1,
            "batch": 2,
            "head_num": 2,
            "MAX_SEQ_LEN": 1024,
            "head_dim": 32,
            "CAUSAL": True,
            "HAS_BIAS": False,
            "CORE_NUM": 40,
            "tasks": 20,
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "bias": None,
        }
        return "_hstu_attn_fwd", signature, constants

    if name == "unified_attention":
        signature = {
            "output_ptr": "*fp16",
            "query_ptr": "*fp16",
            "key_cache_ptr": "*fp16",
            "value_cache_ptr": "*fp16",
            "block_tables_ptr": "*i32",
            "seq_lens_ptr": "*i32",
            "scale": "fp32",
            "softcap": "i32",
            "block_table_stride": "i64",
            "query_stride_0": "i64",
            "query_stride_1": "i64",
            "output_stride_0": "i64",
            "output_stride_1": "i64",
            "stride_k_cache_0": "i64",
            "stride_k_cache_1": "i64",
            "stride_k_cache_2": "i64",
            "stride_v_cache_0": "i64",
            "stride_v_cache_1": "i64",
            "stride_v_cache_2": "i64",
            "query_start_len_ptr": "*i32",
            "num_seqs": "i32",
        }
        constants = {
            "alibi_slopes_ptr": None,
            "k_scale": None,
            "v_scale": None,
            "num_queries_per_kv": 4,
            "BLOCK_SIZE": 32,
            "HEAD_SIZE": 128,
            "HEAD_SIZE_PADDED": 128,
            "USE_ALIBI_SLOPES": False,
            "USE_SOFTCAP": False,
            "SLIDING_WINDOW": 0,
            "stride_k_cache_3": 1,
            "stride_v_cache_3": 1,
            "BLOCK_M": 16,
        }
        return "kernel_unified_attention_2d", signature, constants

    raise ValueError(name)


def load_jit_functions(source_path: Path, function_names: tuple[str, ...]):
    import triton
    import triton.language as tl
    import triton.language.extra.cann.extension as extension

    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    wanted = set(function_names)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    found = {node.name for node in selected}
    if found != wanted:
        raise RuntimeError(f"missing JIT functions in {source_path}: {sorted(wanted - found)}")

    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "__file__": str(source_path),
        "__name__": f"plan_dump_{source_path.stem}",
        "triton": triton,
        "tl": tl,
        "extension": extension,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def extract_plan_ir(log_text: str) -> str:
    lines = log_text.splitlines(keepends=True)
    # This build prints IR before every pass. OpClassifier immediately follows
    # PlanComputeBlock, so its input is PlanComputeBlock's output.
    marker = "IR Dump Before OpClassifierPass (op-classifier)"
    starts = [i for i, line in enumerate(lines) if marker in line]
    if not starts:
        raise RuntimeError("the pass boundary after PlanComputeBlock was not present in the MLIR dump")
    start = starts[-1] + 1
    end = len(lines)
    for i in range(start, len(lines)):
        if "IR Dump " in lines[i] and "// -----//" in lines[i]:
            end = i
            break
    result = "\n".join(line.rstrip() for line in lines[start:end]).strip() + "\n"
    if "module " not in result:
        raise RuntimeError("extracted PlanComputeBlock output does not contain an MLIR module")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--operator", choices=sorted(KERNEL_FUNCTIONS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from triton._C.libtriton import ir
    from triton._C.libtriton.ascend import ir as ascend_ir
    from triton.backends.ascend import _apply_ascend_patch
    from triton.backends.ascend.compiler import NPUOptions, make_ttir, min_dot_size, ttir_to_linalg
    from triton.compiler.code_generator import ast_to_ttir
    from triton.compiler.compiler import ASTSource

    _apply_ascend_patch()
    source_path = args.worktree / "experiment_operators" / "candidates" / f"{args.operator}.py"
    namespace = load_jit_functions(source_path, KERNEL_FUNCTIONS[args.operator])
    entry_name, signature, constants = kernel_config(args.operator)
    kernel = namespace[entry_name]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = args.output_dir / "mlir-pass-dump.log"
    os.environ["MLIR_ENABLE_DUMP"] = "1"

    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    option_fields = NPUOptions.__dataclass_fields__
    dynamic_counts = ({"intra_cache_num": 4, "inter_cache_num": 1, "load_cache_num": 1}
                      if "intra_cache_num" in option_fields else
                      {"buf_slot_num_of_veccore": 4, "buf_slot_num_of_crosscore": 1, "buf_slot_num_of_gm": 1})
    option_values = {
        "arch": "Ascend950PR_9579",
        "enable_dynamic_cv_pipeline": True,
        "set_workspace_multibuffer": 0,
        "multibuffer": True,
        "multibuffer_num": 2,
        "vf_merge_level": 0,
        **dynamic_counts,
    }
    if option_fields.get("compile_on_910_95") is not None and option_fields["compile_on_910_95"].init:
        option_values["compile_on_910_95"] = True
    if option_fields.get("use_bytecode") is not None:
        option_values["use_bytecode"] = False
    options = NPUOptions(**option_values)
    source = ASTSource(kernel, signature, constants)
    codegen_fns = {"min_dot_size": min_dot_size(None)}

    saved_stderr = os.dup(2)
    try:
        with raw_log_path.open("w") as raw_log:
            os.dup2(raw_log.fileno(), 2)
            ttir = ast_to_ttir(kernel, source, context, options, codegen_fns, {})
            metadata = {**options.__dict__}
            ttir = make_ttir(ttir, metadata, options)
            (args.output_dir / "optimized.ttir.mlir").write_text(str(ttir))
            final_ir = ttir_to_linalg(ttir, metadata, options, named_ops=True)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)

    log_text = raw_log_path.read_text()
    plan_ir = extract_plan_ir(log_text)
    (args.output_dir / "after-plan-compute-block.mlir").write_text(plan_ir)
    (args.output_dir / "final.ttadapter.mlir").write_text(str(final_ir))
    summary = {
        "operator": args.operator,
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "entry": entry_name,
        "signature": signature,
        "constants": constants,
        "compiler_options": {
            "arch": getattr(options, "target_arch", getattr(options, "arch", "Ascend950PR_9579")),
            "enable_dynamic_cv_pipeline": options.enable_dynamic_cv_pipeline,
            "intra_cache_num": dynamic_counts.get("intra_cache_num", dynamic_counts.get("buf_slot_num_of_veccore")),
            "inter_cache_num": dynamic_counts.get("inter_cache_num", dynamic_counts.get("buf_slot_num_of_crosscore")),
            "load_cache_num": dynamic_counts.get("load_cache_num", dynamic_counts.get("buf_slot_num_of_gm")),
            "multibuffer_num": options.multibuffer_num,
            "vf_merge_level": options.vf_merge_level,
        },
        "plan_ir_sha256": hashlib.sha256(plan_ir.encode()).hexdigest(),
        "plan_ir_boundary": "input to OpClassifierPass immediately after PlanComputeBlockPass",
        "dynamic_cv_result": metadata.get("dynamic_cv_result"),
        "dynamic_cv_errcode": metadata.get("dynamic_cv_errcode"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
