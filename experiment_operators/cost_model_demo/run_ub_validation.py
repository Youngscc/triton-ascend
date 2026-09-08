#!/usr/bin/env python3
"""Recompile the d/m corpus through native PlanMemory without an NPU runtime.

The production backend constructs each compiler command. The validator captures
that command before execution. A host observer builds the same native pipeline
and executes its prefix through PlanMemory. No device binary is fabricated.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_operators.cost_model_demo import (
    DEFAULT_PROFILE, evaluate_configuration, normalized_ir_sha256, prepare_cost_model,
)
from experiment_operators.cost_model_demo.run_shape_validation import (
    _block_rewrites_match, _expected_inner_allocs, _inner_alloc_counter,
    _load_corpus, _materialize_kernel_source,
)

OPERATORS = ("fused_attention", "flash_attention_npu_v8", "hstu_attention", "unified_attention")
TARGET = "Ascend950PR_9579"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def persist(output, rows):
    write_json(output / "measurements.json", rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def cases_for(suite, corpus):
    cases = []
    if suite in ("all", "matrix"):
        cases.extend(dict(case_id="matrix_" + op, operator=op, constants={},
                          source_variant="original", multibuffers=[1, 2, 3, 4])
                     for op in OPERATORS)
    if suite in ("all", "shapes"):
        for item in _load_corpus(corpus)["cases"]:
            cases.append({**item, "case_id": "shape_" + item["case_id"],
                          "operator": "fused_attention", "multibuffers": [1]})
    return cases


def capture_command(adapter, metadata, options, compiler, case_dir):
    import triton.backends.ascend.compiler as backend

    captured = {}
    real_run = subprocess.run

    class Captured(BaseException):
        pass

    def intercept(command, **kwargs):
        if "-o" not in command:
            return real_run(command, **kwargs)
        if Path(command[0]).resolve() != compiler:
            raise RuntimeError(f"unexpected compiler selected: {command[0]}")
        native_input = case_dir / "native-input.mlir"
        native_input.write_bytes(Path(command[1]).read_bytes())
        command = list(command)
        command[1] = str(native_input)
        command[command.index("-o") + 1] = str(case_dir / "after-plan-memory.mlir")
        captured.update(command=command, environment=kwargs.get("env", os.environ.copy()))
        raise Captured()

    # Device availability is irrelevant to host compilation. All option parsing
    # and command construction remain the production backend's implementation.
    proxy = SimpleNamespace(**{name: getattr(subprocess, name) for name in dir(subprocess)})
    proxy.run = intercept
    with patch.object(backend, "subprocess", proxy), patch.object(
        backend, "NPUUtils", lambda: SimpleNamespace(has_device_limit=lambda: False)
    ):
        try:
            backend.linalg_to_bin_enable_npu_compile_910_95(adapter, metadata, options)
        except Captured:
            pass
    if not captured:
        raise RuntimeError("backend did not construct a native compilation command")
    return captured


def native_observation(returncode, stdout):
    observations = list(map(int, re.findall(r"UB\s+size\s*=\s*(\d+)\s*bits", stdout)))
    ub = max(observations) if observations else None
    status = "measured" if returncode == 0 and ub and ub % 8 == 0 else (
        "compile_failed" if returncode else "ub_missing")
    return dict(compiler_status=status, compiler_returncode=returncode,
                ub_observations_bits=observations,
                compiler_ub_bytes=ub // 8 if status == "measured" else None)


def compile_native(frontend_metadata, adapter, d, m, compiler, oracle, case_dir, timeout):
    from triton.backends.ascend.compiler import NPUOptions
    from triton.backends.compiler import GPUTarget

    options = NPUOptions(
        arch=TARGET, enable_dynamic_cv_pipeline=True,
        buf_slot_num_of_veccore=d, buf_slot_num_of_crosscore=1, buf_slot_num_of_gm=1,
        set_workspace_multibuffer=0, multibuffer=True, multibuffer_num=m, vf_merge_level=0,
    )
    metadata = {**options.__dict__, **frontend_metadata["resolved_options"]}
    metadata.update(target=GPUTarget("npu", TARGET, 32), hash=digest(case_dir / "final.ttadapter.mlir"),
                    disable_vf_operand_substitution=frontend_metadata.get("disable_vf_operand_substitution", False))
    captured = capture_command(adapter, metadata, options, compiler, case_dir)
    write_json(case_dir / "production-command.json", captured["command"])
    command = [str(oracle), *captured["command"][1:]] + [
        "--mlir-disable-threading",
        "--mlir-print-ir-before=hivm-mark-multi-buffer,hivm-plan-memory-regbase",
        "--mlir-print-ir-after=hivm-mark-multi-buffer",
    ]
    required = [f"--set-local-multibuffer={m}", "--enable-vf-merge-level=0",
                "--enable-auto-multi-buffer=True", "--limit-auto-multi-buffer-buffer=no-limit"]
    if any(flag not in command for flag in required):
        raise RuntimeError(f"resolved compiler flags do not match requested d/m: {required}")
    write_json(case_dir / "native-command.json", command)
    write_json(case_dir / "effective-backend-options.json", {
        key: value for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None)))
    })
    return run_observer(command, case_dir, timeout, captured["environment"])


def run_observer(command, case_dir, timeout, environment=None):
    started = time.monotonic()
    try:
        result = subprocess.run(command, env=environment, capture_output=True,
                                text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b""; stderr = error.stderr or b""
        if isinstance(stdout, bytes): stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes): stderr = stderr.decode(errors="replace")
        (case_dir / "compiler.log").write_text("COMMAND: " + shlex.join(command) + "\n" + stdout + stderr)
        return dict(compiler_status="timeout", compiler_returncode=124,
                    compile_seconds=time.monotonic()-started, compiler_ub_bytes=None)
    log = "COMMAND: " + shlex.join(command) + "\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr
    (case_dir / "compiler.log").write_text(log)
    observation = native_observation(result.returncode, result.stdout)
    if result.returncode == 0 and "PLAN_MEMORY_ORACLE_OK" not in result.stdout:
        observation.update(compiler_status="oracle_incomplete", compiler_ub_bytes=None)
    return dict(observation,
                compile_seconds=time.monotonic()-started,
                vf_operand_substitution="--enable-vf-operand-substitution=True" in command,
                compiler_error=result.stderr[-3000:] if result.returncode else "")


def effective_counts(plan):
    names = {"buf_slot_num_of_veccore": "intra_buf_count",
             "buf_slot_num_of_crosscore": "inter_core_buf_count",
             "buf_slot_num_of_gm": "load_store_buf_count"}
    result = {}
    for field, attr in names.items():
        match = re.search(r"ssbuffer\." + attr + r"\s*=\s*(\d+)", plan)
        result[field] = int(match.group(1)) if match else None
    return result


def canonical_plan(path, triton_opt):
    """Parse/print only: no optimization passes, discard debug names and locations."""
    result = subprocess.run([str(triton_opt), str(path), "--mlir-print-op-generic",
                             "--mlir-use-nameloc-as-prefix=false"], capture_output=True,
                            text=True, check=True, timeout=30)
    path.with_name("canonical-plan.mlir").write_text(result.stdout)
    return normalized_ir_sha256(result.stdout)


def audit_profile(profile):
    # Reference formulas are evaluated only to quantify the regression. This
    # NEVER changes the compiler's real options or makes its profile supported.
    return replace(profile, enable_vf_operand_substitution=False)


def compare_case(rows):
    by_config = {(r["dynamic_cv"], r["multibuffer_num"]): r for r in rows}
    base = by_config[(1, 1)].get("compiler_ub_bytes")
    identity = "structural_plan_sha256" if any("structural_plan_sha256" in r for r in rows) else "normalized_plan_sha256"
    known = {r.get(identity) for r in rows if r.get(identity)}
    invariant = len(known) == 1 and all(r.get(identity) for r in rows)
    configurations_resolved = all(r.get("requested_dynamic_resolved") and r.get("profile_options_match") for r in rows)
    for row in rows:
        d, m = row["dynamic_cv"], row["multibuffer_num"]
        row["plan_invariant"] = bool(invariant)
        ub = row.get("compiler_ub_bytes")
        ud = by_config[(d, 1)].get("compiler_ub_bytes")
        um = by_config[(1, m)].get("compiler_ub_bytes")
        if all(x is not None for x in (base, ub, ud, um)):
            row.update(compiler_delta_bytes=ub-base, compiler_dynamic_delta_bytes=ud-base,
                       compiler_multibuffer_delta_bytes=um-base,
                       compiler_interaction_bytes=ub-ud-um+base)
            pred = row.get("model_delta_bytes")
            row["error_bytes"] = pred-(ub-base) if pred is not None else None
            pairs = [("model_delta_bytes", "compiler_delta_bytes"),
                     ("model_dynamic_delta_bytes", "compiler_dynamic_delta_bytes"),
                     ("model_multibuffer_delta_bytes", "compiler_multibuffer_delta_bytes"),
                     ("model_interaction_bytes", "compiler_interaction_bytes")]
            row["all_deltas_match"] = all(row.get(a) is not None and row[a] == row[b] for a, b in pairs)
            if not invariant or not configurations_resolved:
                outcome = "unsupported_config"
            elif pred is None:
                outcome = "model_unknown"
            elif row["all_deltas_match"]:
                outcome = "exact_prediction"
            else:
                outcome = "mismatch"
        else:
            outcome = "compiler_incomplete"
        row["validation_outcome"] = outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, help="Defaults to plan-memory-oracle beside the compiler")
    parser.add_argument("--triton-mlir-opt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "matrix", "shapes"), default="all")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--replay-native-artifacts", action="store_true", help="Rerun native UB capture on saved frontend outputs; preserve the prior attempt")
    parser.add_argument("--recheck-artifacts", action="store_true", help="Recheck saved IR identities and comparisons without recompiling")
    args = parser.parse_args()
    compiler, triton_opt, output = args.compiler.resolve(), args.triton_mlir_opt.resolve(), args.output_dir.resolve()
    oracle = args.oracle.resolve() if args.oracle else compiler.parent/"plan-memory-oracle"
    output.mkdir(parents=True, exist_ok=True)
    if args.replay_native_artifacts:
        return replay_native_artifacts(output, compiler, oracle, triton_opt, args.timeout, args.strict)
    if args.recheck_artifacts:
        return recheck_artifacts(output, triton_opt, args.strict)
    if (output / "measurements.json").exists():
        raise RuntimeError("choose a new output directory; existing measurements are never overwritten")
    for tool in (compiler, compiler.parent/"bishengir-opt", triton_opt, oracle):
        if not tool.is_file() or not os.access(tool, os.X_OK): raise RuntimeError(f"missing tool: {tool}")
    if not os.environ.get("REMOTE_MODE") == "dev": raise RuntimeError("set REMOTE_MODE=dev")
    os.environ.update(ENABLE_PRINT_UB_BITS="true", TRITON_BACKENDS_IN_TREE="1",
                      TRITON_NPU_COMPILER_PATH=str(compiler.parent),
                      PATH=os.pathsep.join([str(compiler.parent), str(triton_opt.parent), os.environ["PATH"]]),
                      TRITON_CACHE_DIR=str(output/"cache"))
    provenance = {"top_level_revision": revision(ROOT),
                  "ascend_npu_ir_revision": revision(ROOT/"third_party/ascend/AscendNPU-IR"),
                  "llvm_revision": revision(ROOT/"third_party/ascend/AscendNPU-IR/third-party/llvm-project")}
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    oracle_version = subprocess.check_output([str(oracle), "--version"], text=True)
    if any(provenance["ascend_npu_ir_revision"][:12] not in value for value in (version, oracle_version)):
        raise RuntimeError("bishengir-compile version does not match the checkout")
    profile = replace(DEFAULT_PROFILE, profile_id="a5-main-dm-revalidation-candidate",
                      triton_ascend_revision=provenance["top_level_revision"],
                      ascend_npu_ir_revision=provenance["ascend_npu_ir_revision"],
                      llvm_revision=provenance["llvm_revision"], enable_vf_operand_substitution=True)
    write_json(output/"candidate-profile.json", asdict(profile))
    write_json(output/"reference-rule-profile.json", asdict(audit_profile(profile)))
    corpus = Path(__file__).with_name("large_validation_corpus.json")
    cases = cases_for(args.suite, corpus)
    if args.case:
        unknown = set(args.case)-{c["case_id"] for c in cases}
        if unknown: raise ValueError(f"unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c["case_id"] in args.case]
    import triton
    from triton._C import libtriton
    if Path(triton.__file__).resolve() != ROOT/"python/triton/__init__.py":
        raise RuntimeError("runtime Triton is not from this checkout")
    manifest = dict(provenance, compiler=str(compiler), compiler_sha256=digest(compiler),
                    compiler_version=version,
                    build_patch_sha256={p.name: digest(p) for p in (ROOT/"third_party/ascend/patch").glob("triton-ascend*3.6.0.patch")},
                    oracle_source_sha256=digest(Path(__file__).with_name("plan_memory_oracle.cpp")),
                    oracle_path=str(oracle), oracle_sha256=digest(oracle),
                    oracle_version=oracle_version, triton_mlir_opt=str(triton_opt),
                    triton_mlir_opt_sha256=digest(triton_opt), libtriton=str(libtriton.__file__),
                    libtriton_sha256=digest(libtriton.__file__), python=sys.executable,
                    expected_rows=sum(3*len(c["multibuffers"]) for c in cases),
                    cases=cases, profile_status="unsupported_vf_operand_substitution", model_scope="legacy_rule_audit", target=TARGET,
                    oracle="native HIR + final HIVM prefix through local PlanMemory; first attempt, no fallback or NPU execution")
    write_json(output/"manifest.json", manifest)
    rows=[]
    for index, case in enumerate(cases, 1):
        print(f'CASE {index}/{len(cases)} {case["case_id"]}', flush=True)
        case_root=output/"cases"/case["case_id"]; case_root.mkdir(parents=True)
        if case["operator"] == "fused_attention":
            source, source_hash = _materialize_kernel_source(case, ROOT, case_root)
        else:
            source = case_root/"kernel_variant.py"
            source.write_bytes((ROOT/"experiment_operators/candidates"/(case["operator"]+".py")).read_bytes())
            source_hash=digest(source)
        constants=case_root/"constants.json"; write_json(constants, case["constants"])
        prepared=None; model_error=""; case_rows=[]
        for d in (1,2,3):
            for m in case["multibuffers"]:
                case_dir=case_root/f"d{d}-m{m}"; case_dir.mkdir()
                row=dict(case_id=case["case_id"], operator=case["operator"], dynamic_cv=d, multibuffer_num=m,
                         source_sha256=source_hash, vf_merge_level=0, compiler_status="pending",
                         artifacts=str(case_dir.relative_to(output)), model_scope="legacy_rule_audit", model_profile_supported=False)
                rows.append(row); case_rows.append(row); persist(output, rows)
                command=[sys.executable, str(ROOT/"experiment_operators/plan_compute_block_ir/dump_plan_compute_block.py"),
                         "--worktree", str(ROOT), "--operator",case["operator"], "--source-path",str(source),
                         "--constants-json",str(constants),"--dynamic-cv",str(d),"--multibuffer-num",str(m),
                         "--allow-missing-plan", "--output-dir",str(case_dir)]
                try:
                    result=subprocess.run(command,capture_output=True,text=True,timeout=args.timeout)
                    (case_dir/"frontend.log").write_text(result.stdout+"\n"+result.stderr)
                    if result.returncode: raise RuntimeError(result.stderr[-3000:])
                    front=json.loads((case_dir/"metadata.json").read_text())
                    resolved=front["resolved_options"]
                    row["requested_dynamic_resolved"] = (resolved.get("enable_dynamic_cv_pipeline") is True
                        and resolved.get("buf_slot_num_of_veccore")==d
                        and resolved.get("buf_slot_num_of_crosscore")==1 and resolved.get("buf_slot_num_of_gm")==1)
                    row["dynamic_cv_errcode"]=front.get("dynamic_cv_errcode")
                    plan_path=case_dir/"after-plan-compute-block.mlir"
                    if plan_path.exists():
                        plan=plan_path.read_text(); row["normalized_plan_sha256"]=normalized_ir_sha256(plan)
                        row["structural_plan_sha256"]=canonical_plan(plan_path,triton_opt)
                        row["plan_effective_counts"] = effective_counts(plan)
                        row["requested_dynamic_resolved"] = row["requested_dynamic_resolved"] and (
                            row["plan_effective_counts"] == {"buf_slot_num_of_veccore": d,
                                "buf_slot_num_of_crosscore": 1, "buf_slot_num_of_gm": 1})
                        if d==1 and m==1:
                            try: prepared=prepare_cost_model(plan,profile=audit_profile(profile),observed_provenance=provenance)
                            except Exception as error: model_error=str(error)
                    else:
                        row["requested_dynamic_resolved"] = False
                        model_error=front.get("plan_ir_unavailable_reason") or "PlanComputeBlock not available"
                    if prepared is not None:
                        estimate=evaluate_configuration(prepared,intra_cache_num=d,multibuffer_num=m,baseline=None)
                        row.update(model_delta_bytes=estimate.total_from_11_bytes,
                                   model_dynamic_delta_bytes=estimate.dynamic_from_d1_bytes,
                                   model_multibuffer_delta_bytes=estimate.ordinary_from_m1_bytes,
                                   model_interaction_bytes=estimate.interaction_bytes,
                                   model_blockers=list(prepared.blockers), model_reason=estimate.reason)
                        try:
                            log=(case_dir/"mlir-pass-dump.log").read_text()
                            trace_prepared=prepare_cost_model(plan,profile=audit_profile(profile),observed_provenance=provenance)
                            row["block_rewrite_match"]=_block_rewrites_match(log,trace_prepared.dynamic_buffers)
                            raw=_inner_alloc_counter(log); expected=_expected_inner_allocs(prepared.dynamic_buffers,d)
                            row.update(raw_alloc_expected=dict(expected),raw_alloc_compiler=dict(raw),raw_alloc_match=raw==expected)
                        except Exception as error: row["intermediate_check_error"]=str(error)
                    else: row.update(model_delta_bytes=None,model_blockers=[model_error])
                    row.update(compile_native(front,(case_dir/"final.ttadapter.mlir").read_text(),d,m,
                                              compiler,oracle,case_dir,args.timeout))
                    row["profile_options_match"] = (row.get("vf_operand_substitution")
                        == profile.enable_vf_operand_substitution)
                except subprocess.TimeoutExpired:
                    row.update(compiler_status="frontend_timeout",error="frontend exceeded timeout")
                except Exception as error:
                    row.update(compiler_status="validation_error",error=str(error))
                persist(output,rows)
                print(f'  d={d} m={m} {row["compiler_status"]} UB={row.get("compiler_ub_bytes")} model_delta={row.get("model_delta_bytes")}',flush=True)
        compare_case(case_rows); persist(output,rows)
    summary=summarize(rows,manifest["expected_rows"])
    write_json(output/"summary.json",summary); print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 1 if args.strict and not summary["validation_pass"] else 0


def summarize(rows, expected_rows):
    counts={label:sum(r.get("validation_outcome")==label for r in rows) for label in
            ("exact_prediction","model_unknown","mismatch","unsupported_config","compiler_incomplete")}
    summary=dict(counts,row_count=len(rows),expected_rows=expected_rows,
                 all_rows_measured=all(r.get("compiler_status")=="measured" for r in rows),
                 intermediate_mismatches=sum(r.get("block_rewrite_match") is False or r.get("raw_alloc_match") is False for r in rows),
                 unverified_intermediate_rows=sum("intermediate_check_error" in r for r in rows))
    summary["numerical_validation_pass"]=(len(rows)==expected_rows and summary["all_rows_measured"]
                                           and counts["exact_prediction"] > 0
                                           and not counts["mismatch"] and not counts["unsupported_config"] and not counts["model_unknown"])
    numeric = [r for r in rows if r.get("model_delta_bytes") is not None]
    summary["supported_intermediate_failures"] = sum(
        r.get("block_rewrite_match") is not True or r.get("raw_alloc_match") is not True for r in numeric)
    summary["validation_pass"] = summary["numerical_validation_pass"] and not summary["supported_intermediate_failures"]
    summary["compiler_measured"] = sum(r.get("compiler_status")=="measured" for r in rows)
    summary["compiler_failed_or_missing"] = len(rows)-summary["compiler_measured"]
    summary["model_scope"] = "legacy_rule_audit"
    summary["production_profile_supported"] = all(r.get("model_profile_supported", True) for r in rows)
    summary["validation_pass"] = summary["validation_pass"] and summary["production_profile_supported"]
    return summary



def replay_native_artifacts(output, compiler, oracle, triton_opt, timeout, strict):
    manifest=json.loads((output/"manifest.json").read_text())
    if digest(compiler) != manifest["compiler_sha256"] or digest(triton_opt) != manifest["triton_mlir_opt_sha256"]:
        raise RuntimeError("replay requires the same compiler and frontend parser")
    version=subprocess.check_output([str(oracle),"--version"],text=True)
    if manifest["ascend_npu_ir_revision"][:12] not in version:
        raise RuntimeError("observer revision differs from the saved compiler")
    archive=output/"before-native-replay"
    archive.mkdir()  # never overwrite a prior attempt
    for name in ("manifest.json","measurements.json","results.csv","summary.json"):
        shutil.copy2(output/name,archive/name)
    rows=json.loads((output/"measurements.json").read_text())
    for index,row in enumerate(rows,1):
        directory=output/row["artifacts"]
        saved=archive/row["artifacts"]; saved.mkdir(parents=True)
        for name in ("compiler.log","after-plan-memory.mlir","after-plan-memory.mlir.pipeline.txt","native-command.json"):
            if (directory/name).exists(): shutil.copy2(directory/name,saved/name)
        command=json.loads((directory/"native-command.json").read_text()); command[0]=str(oracle)
        write_json(directory/"native-command.json",command)
        row.update(run_observer(command,directory,timeout))
        persist(output,rows)
        print(f'REPLAY {index}/{len(rows)} {row["case_id"]} d={row["dynamic_cv"]} m={row["multibuffer_num"]} {row["compiler_status"]} UB={row.get("compiler_ub_bytes")}',flush=True)
    manifest.update(oracle_sha256=digest(oracle),oracle_source_sha256=digest(Path(__file__).with_name("plan_memory_oracle.cpp")),
                    oracle="production runPipeline and native HIR/final HIVM builders through local PlanMemory; first attempt, no NPU execution")
    write_json(output/"manifest.json",manifest)
    return recheck_artifacts(output,triton_opt,strict)


def recheck_artifacts(output, triton_opt, strict):
    manifest=json.loads((output/"manifest.json").read_text())
    if digest(triton_opt) != manifest["triton_mlir_opt_sha256"]:
        raise RuntimeError("recheck requires the same native parser used to compile the corpus")
    rows=json.loads((output/"measurements.json").read_text())
    backup=output/"measurements.before-structural-recheck.json"
    if not backup.exists(): write_json(backup,rows)
    from experiment_operators.cost_model_demo.stages.validate_context import CompilerProfile
    profile=CompilerProfile.from_mapping(json.loads((output/"candidate-profile.json").read_text()))
    provenance={key:manifest[key] for key in ("top_level_revision","ascend_npu_ir_revision","llvm_revision")}
    groups={}
    for row in rows:
        groups.setdefault(row["case_id"],[]).append(row)
        row.update(model_scope="legacy_rule_audit",model_profile_supported=False)
        directory=output/row["artifacts"]; plan=directory/"after-plan-compute-block.mlir"
        if not plan.exists(): continue
        row["structural_plan_sha256"]=canonical_plan(plan,triton_opt)
        try:
            trace=prepare_cost_model(plan.read_text(),profile=audit_profile(profile),observed_provenance=provenance)
            row["block_rewrite_match"]=_block_rewrites_match((directory/"mlir-pass-dump.log").read_text(),trace.dynamic_buffers)
        except Exception as error:
            row["intermediate_check_error"]=str(error)
    for group in groups.values(): compare_case(group)
    persist(output,rows)
    manifest.update(profile_status="unsupported_vf_operand_substitution",model_scope="legacy_rule_audit",
                    structural_recheck="same native MLIR parser, generic printing, no optimization passes",
                    validator_source_sha256=digest(__file__))
    write_json(output/"manifest.json",manifest)
    write_json(output/"reference-rule-profile.json",asdict(audit_profile(profile)))
    summary=summarize(rows,manifest["expected_rows"])
    write_json(output/"summary.json",summary); print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 1 if strict and not summary["validation_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
