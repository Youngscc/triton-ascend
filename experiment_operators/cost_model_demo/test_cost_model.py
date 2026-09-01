#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import random
import textwrap
import unittest
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from experiment_operators.cost_model_demo import (
    BaselineCertificate,
    DEFAULT_PROFILE,
    InteractionContribution,
    LifeInterval,
    MemoryEntry,
    UbCostModel,
    UnsupportedModelError,
    prepare_cost_model,
    evaluate_all_configurations,
    evaluate_configuration,
    make_baseline_certificate,
    normalized_ir_sha256,
    plan_lite,
)
from experiment_operators.cost_model_demo.run_shape_validation import (
    _block_rewrites_match,
    _expected_inner_allocs,
    _inner_alloc_counter,
    _materialize_kernel_source,
)
from experiment_operators.plan_compute_block_ir.dump_plan_compute_block import kernel_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MLIR_ROOT = REPO_ROOT / "outputs" / "mac_ub_latest_dev_prev_npuir_20260825" / "mlir"
TRUE_DATA = REPO_ROOT / "outputs" / "mac_ub_latest_dev_prev_npuir_20260825" / "mac_ub_results.csv"


def synthetic_boundary_ir(
    *,
    function_name: str = "kernel",
    loop_kind: str = "scf.for",
    gm_load_hint: Optional[int] = None,
    duplicate_copy: bool = False,
    dynamic_shape: bool = False,
) -> str:
    hint = ""
    if gm_load_hint is not None:
        hint = f"%mark = annotation.mark %buf {{gm_load_multi_buffer = {gm_load_hint} : i32}}\n"
    second_copy = ("      memref.copy %input_view, %buf {ssbuffer.block_id = 1 : i32, "
                   "ssbuffer.core_type = \"VECTOR\"} : memref<64x64xf16> to memref<64x64xf16>\n"
                   if duplicate_copy else "")
    sizes = "%n, 64" if dynamic_shape else "64, 64"
    if loop_kind == "scf.while":
        loop_header = "%loop = scf.while (%arg = %c0) : (index) -> index {"
    else:
        loop_header = f"{loop_kind} %i = %c0 to %c1 step %c1 {{"
    return textwrap.dedent(f"""
        module attributes {{hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.inter_core_buf_count = 1 : i32, ssbuffer.intra_buf_count = 1 : i32, ssbuffer.load_store_buf_count = 1 : i32}} {{
          func.func @{function_name}(%input: memref<?xf16> {{tt.tensor_kind = 0 : i32}}, %output: memref<?xf16> {{tt.tensor_kind = 1 : i32}}) {{
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %n = arith.constant 64 : index
            {loop_header}
              %input_view = memref.reinterpret_cast %input to offset: [0], sizes: [{sizes}], strides: [64, 1] {{ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}} : memref<?xf16> to memref<64x64xf16>
              %output_view = memref.reinterpret_cast %output to offset: [0], sizes: [64, 64], strides: [64, 1] {{ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}} : memref<?xf16> to memref<64x64xf16>
              %buf = memref.alloc() {{ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}} : memref<64x64xf16>
              {hint}memref.copy %input_view, %buf {{ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}} : memref<64x64xf16> to memref<64x64xf16>
        {second_copy}      %value = tensor.empty() {{ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}} : tensor<64x64xf16>
              bufferization.materialize_in_destination %value in writable %output_view {{ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}} : (tensor<64x64xf16>, memref<64x64xf16>) -> ()
            }}
          }}
        }}
    """)


class CostModelRegressionTest(unittest.TestCase):
    EXPECTED = {
        "fused_attention": ([128, 128], (0, 256, 512), 4224),
        "flash_attention_npu_v8": ([128, 128], (0, 256, 512), 4224),
        "hstu_attention": ([], (0, 0, 0), 0),
        "unified_attention": ([32, 32], (0, 64, 128), 2048),
    }

    def ir_text(self, operator: str, dynamic: int = 1) -> str:
        path = MLIR_ROOT / operator / f"dynamic_{dynamic}" / "after-plan-compute-block.mlir"
        return path.read_text(encoding="utf-8")

    def prepare(self, operator: str, dynamic: int = 1):
        return prepare_cost_model(self.ir_text(operator, dynamic), vf_merge_level=0)

    def prepare_verified(self, operator: str, dynamic: int = 1):
        profile = DEFAULT_PROFILE
        provenance = {
            "top_level_revision": profile.triton_ascend_revision,
            "ascend_npu_ir_revision": profile.ascend_npu_ir_revision,
            "llvm_revision": profile.llvm_revision,
        }
        return prepare_cost_model(
            self.ir_text(operator, dynamic),
            vf_merge_level=0,
            profile=profile,
            observed_provenance=provenance,
        )

    def compiler_stage_input(self, operator: str, pass_name: str) -> str:
        path = MLIR_ROOT / operator / "dynamic_1" / "mlir-pass-dump.log"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        marker = next(index for index, line in enumerate(lines)
                      if "IR Dump Before" in line and pass_name in line)
        end = next(index for index in range(marker + 1, len(lines))
                   if "IR Dump " in lines[index] and "// -----//" in lines[index])
        return "\n".join(lines[marker + 1:end])

    def true_tables(self):
        tables = defaultdict(dict)
        with TRUE_DATA.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                dynamic = row["dynamic_cv"]
                multibuffer = row["multibuffer_num"]
                if (dynamic == "off" or multibuffer == "off" or int(dynamic) > 3
                        or row["vf_merge_level"] != "0"
                        or row["status"] != "measured"):
                    continue
                tables[row["operator"]][(int(dynamic), int(multibuffer))] = int(row["ub_bits"]) // 8
        return tables

    def test_dynamic_and_ordinary_families_are_structurally_projected(self):
        for operator, (dynamic_sizes, dynamic_delta, ordinary_step) in self.EXPECTED.items():
            with self.subTest(operator=operator):
                prepared = self.prepare(operator)
                self.assertEqual(sorted(item.aligned_bytes for item in prepared.dynamic_buffers), dynamic_sizes)
                self.assertEqual(prepared.dynamic_delta_table_bytes, dynamic_delta)
                self.assertEqual(prepared.ordinary_step_table_bytes, (ordinary_step, ) * 3)
                self.assertEqual(prepared.blockers, ())

    def test_dynamic_families_are_derived_from_weighted_dependency_cuts(self):
        expected = {
            "fused_attention": {
                "%alpha": ("%m_ij_69", 11, (12, ), 128),
                "%l_ij_81": ("%p", 11, (12, ), 128),
            },
            "flash_attention_npu_v8": {
                "%alpha": ("%m_new_177", 16, (17, ), 128),
                "%p_sum_202": ("%p_179", 16, (17, ), 128),
            },
            "unified_attention": {
                "%alpha": ("%m_j_147", 13, (14, ), 32),
                "%l_j_160": ("%P_149", 13, (14, ), 32),
            },
        }
        for operator, family_expectations in expected.items():
            with self.subTest(operator=operator):
                prepared = self.prepare(operator)
                actual = {
                    family.origin: (
                        family.seed_dependencies[0],
                        family.producer_block,
                        family.consumer_blocks,
                        family.aligned_bytes,
                    )
                    for family in prepared.dynamic_buffers
                }
                self.assertEqual(actual, family_expectations)
                self.assertTrue(all(family.pattern == "ub_usage_min_cut"
                                    for family in prepared.dynamic_buffers))
                self.assertTrue(all(family.selection_rule == "pcb-ub-usage-min-cut-v1"
                                    for family in prepared.dynamic_buffers))
                self.assertEqual(
                    sum(bool(cut.family_values) for cut in prepared.dynamic_cuts),
                    len(family_expectations),
                )

        fused = self.prepare("fused_attention")
        self.assertEqual(fused.dynamic_cv_summary["broad_dependency_count"], 11)
        self.assertEqual(fused.dynamic_cv_summary["intra_dependency_count"], 8)
        self.assertEqual(fused.dynamic_cv_summary["inter_dependency_count"], 3)
        self.assertEqual(fused.dynamic_cv_summary["scope"], "loop-local-weighted-dependency-cut")
        cut_by_family = {
            cut.family_values[0]: cut
            for cut in fused.dynamic_cuts
            if cut.family_values
        }
        self.assertEqual(cut_by_family["%alpha"].origin_frontier_bytes, 512)
        self.assertEqual(cut_by_family["%alpha"].selected_frontier_bytes, 256)
        self.assertEqual(cut_by_family["%alpha"].moved_values, ("%alpha", ))
        self.assertEqual(cut_by_family["%l_ij_81"].origin_frontier_bytes, 33024)
        self.assertEqual(cut_by_family["%l_ij_81"].selected_frontier_bytes, 256)
        self.assertEqual(cut_by_family["%l_ij_81"].moved_values, ("%l_ij_81", ))

    def test_dependency_cut_block_rewrites_match_compiler_intermediate_ir(self):
        for operator in ("fused_attention", "flash_attention_npu_v8", "unified_attention"):
            with self.subTest(operator=operator):
                compiler_ir = self.compiler_stage_input(operator, "BroadcastUBOptPass")
                for family in self.prepare(operator).dynamic_buffers:
                    defining_lines = [line for line in compiler_ir.splitlines()
                                      if line.lstrip().startswith(f"{family.origin} =")]
                    self.assertEqual(len(defining_lines), 1)
                    self.assertIn(
                        f"ssbuffer.block_id = {family.producer_block} : i32",
                        defining_lines[0],
                    )

    def test_dependency_cut_is_independent_of_softmax_labels_and_accumulator_marker(self):
        original = self.prepare("fused_attention")
        relabeled = prepare_cost_model(
            self.ir_text("fused_attention", 1)
            .replace("@attn_fwd", "@arbitrary_kernel")
            .replace("ssbuffer.add_from_matmul", "ssbuffer.unrelated_marker"),
            vf_merge_level=0,
        )
        self.assertEqual(
            [(item.origin, item.seed_dependencies) for item in relabeled.dynamic_buffers],
            [(item.origin, item.seed_dependencies) for item in original.dynamic_buffers],
        )
        self.assertEqual(relabeled.dynamic_delta_table_bytes, original.dynamic_delta_table_bytes)
        self.assertEqual(relabeled.blockers, ())

    def test_count_normalized_hash_is_identical_for_all_three_dynamic_inputs(self):
        for operator in self.EXPECTED:
            with self.subTest(operator=operator):
                digests = {normalized_ir_sha256(self.ir_text(operator, dynamic)) for dynamic in range(1, 4)}
                self.assertEqual(len(digests), 1)

    def test_all_48_total_deltas_match_true_machine_bytes(self):
        tables = self.true_tables()
        self.assertEqual(set(tables), set(self.EXPECTED))
        comparisons = 0
        for operator, values in sorted(tables.items()):
            self.assertEqual(len(values), 12)
            baseline = values[(1, 1)]
            modeled = {(result.intra_cache_num, result.multibuffer_num): result
                       for result in evaluate_all_configurations(self.prepare(operator))}
            for key, actual_ub in sorted(values.items()):
                self.assertEqual(modeled[key].total_from_11_bytes,
                                 actual_ub - baseline,
                                 msg=f"{operator}, d={key[0]}, m={key[1]}")
                comparisons += 1
        self.assertEqual(comparisons, 48)

    def test_all_92_adjacent_and_mixed_differences_match_true_machine_bytes(self):
        comparisons = 0
        for operator, actual in sorted(self.true_tables().items()):
            modeled_results = evaluate_all_configurations(self.prepare(operator))
            modeled = {(item.intra_cache_num, item.multibuffer_num): item.total_from_11_bytes
                       for item in modeled_results}
            for dynamic in range(1, 3):
                for multibuffer in range(1, 5):
                    actual_step = actual[(dynamic + 1, multibuffer)] - actual[(dynamic, multibuffer)]
                    modeled_step = modeled[(dynamic + 1, multibuffer)] - modeled[(dynamic, multibuffer)]
                    self.assertEqual(actual_step,
                                     modeled_step,
                                     msg=f"{operator}, adjacent d at ({dynamic},{multibuffer})")
                    comparisons += 1
            for dynamic in range(1, 4):
                for multibuffer in range(1, 4):
                    actual_step = actual[(dynamic, multibuffer + 1)] - actual[(dynamic, multibuffer)]
                    modeled_step = modeled[(dynamic, multibuffer + 1)] - modeled[(dynamic, multibuffer)]
                    self.assertEqual(actual_step,
                                     modeled_step,
                                     msg=f"{operator}, adjacent m at ({dynamic},{multibuffer})")
                    comparisons += 1
            for dynamic in range(1, 3):
                for multibuffer in range(1, 4):
                    actual_mixed = (actual[(dynamic + 1, multibuffer + 1)] - actual[(dynamic + 1, multibuffer)] -
                                    actual[(dynamic, multibuffer + 1)] + actual[(dynamic, multibuffer)])
                    modeled_mixed = (modeled[(dynamic + 1, multibuffer + 1)] - modeled[(dynamic + 1, multibuffer)] -
                                     modeled[(dynamic, multibuffer + 1)] + modeled[(dynamic, multibuffer)])
                    self.assertEqual(actual_mixed,
                                     modeled_mixed,
                                     msg=f"{operator}, mixed cell ({dynamic},{multibuffer})")
                    comparisons += 1
        self.assertEqual(comparisons, 92)

    def test_identity_matched_baseline_calibrates_all_12_values(self):
        prepared = self.prepare_verified("fused_attention")
        baseline = make_baseline_certificate(
            prepared,
            ub_peak_bytes=63104,
            no_reuse_required_bytes=63104,
            used_reuse=False,
        )
        predicted = {(item.intra_cache_num, item.multibuffer_num): item.predicted_ub_bytes
                     for item in evaluate_all_configurations(prepared, baseline)}
        self.assertEqual(predicted, self.true_tables()["fused_attention"])

    def test_checked_in_baseline_example_matches_fused_input(self):
        path = (REPO_ROOT / "experiment_operators" / "cost_model_demo" / "fused_d1_baseline.example.json")
        baseline = BaselineCertificate.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        results = evaluate_all_configurations(self.prepare_verified("fused_attention"), baseline)
        self.assertEqual(results[-1].predicted_ub_bytes, 76288)
        self.assertEqual(results[-1].confidence, "exact_calibrated_no_reuse")

    def test_unverified_scalar_baseline_never_becomes_a_proof(self):
        prepared = self.prepare("fused_attention")
        result = evaluate_configuration(
            prepared,
            intra_cache_num=3,
            multibuffer_num=4,
            baseline=BaselineCertificate(
                ub_peak_bytes=63104,
                no_reuse_required_bytes=63104,
                used_reuse=False,
            ),
        )
        self.assertEqual(result.predicted_ub_bytes, 76288)
        self.assertEqual(result.verdict, "unknown")
        self.assertEqual(result.confidence, "calibrated_unverified")
        self.assertFalse(result.prune_allowed)

    def test_baseline_must_bind_multibuffer_one(self):
        prepared = self.prepare_verified("fused_attention")
        baseline = make_baseline_certificate(
            prepared,
            ub_peak_bytes=63104,
            no_reuse_required_bytes=63104,
            used_reuse=False,
        )
        with self.assertRaises(UnsupportedModelError):
            evaluate_configuration(
                prepared,
                intra_cache_num=3,
                multibuffer_num=4,
                baseline=replace(baseline, multibuffer_num=2),
            )

    def test_nonzero_interaction_is_computed_from_d_dependent_ordinary_step(self):
        prepared = replace(
            self.prepare("fused_attention"),
            ordinary_step_table_bytes=(4224, 4256, 4224),
            interactions=(InteractionContribution(
                origin="%Out",
                cause="ordinary_physical_size_changed_by_dynamic_projection",
                intra_cache_num=2,
                multibuffer_num=3,
                contribution_bytes=64,
                provenance=("synthetic:d-dependent-ordinary-step", ),
            ), ),
        )
        result = evaluate_configuration(
            prepared,
            intra_cache_num=2,
            multibuffer_num=3,
            baseline=None,
        )
        self.assertEqual(result.dynamic_from_d1_bytes, 256)
        self.assertEqual(result.ordinary_from_m1_bytes, 8448)
        self.assertEqual(result.interaction_bytes, 64)
        self.assertEqual(result.total_from_11_bytes, 8768)
        self.assertEqual(result.interaction_contributions[0].origin, "%Out")

    def test_nonzero_interaction_without_origin_ledger_fails_open(self):
        prepared = replace(
            self.prepare("fused_attention"),
            ordinary_step_table_bytes=(4224, 4256, 4224),
        )
        result = evaluate_configuration(
            prepared,
            intra_cache_num=2,
            multibuffer_num=3,
            baseline=None,
        )
        self.assertIsNone(result.total_from_11_bytes)
        self.assertEqual(result.confidence, "incomplete_interaction_ledger")

    def test_api_reuses_normalized_prepare_across_d_inputs(self):
        model = UbCostModel()
        first = model.prepare(self.ir_text("unified_attention", 1))
        second = model.prepare(self.ir_text("unified_attention", 3))
        self.assertIs(first, second)
        result = model.evaluate(first, intra_cache_num=3, multibuffer_num=4)
        self.assertEqual(result.total_from_11_bytes, 6272)
        self.assertEqual(model.decide(first, intra_cache_num=3, multibuffer_num=4).action, "continue_real_compilation")

    def test_dynamic_four_is_outside_the_model_domain(self):
        prepared = self.prepare("fused_attention")
        with self.assertRaises(UnsupportedModelError):
            evaluate_configuration(
                prepared,
                intra_cache_num=4,
                multibuffer_num=1,
                baseline=None,
            )

    def test_cache_cannot_bypass_fixed_count_guard(self):
        model = UbCostModel()
        text = self.ir_text("unified_attention", 1)
        model.prepare(text)
        bad = text.replace(
            "ssbuffer.inter_core_buf_count = 1 : i32",
            "ssbuffer.inter_core_buf_count = 2 : i32",
            1,
        )
        with self.assertRaises(UnsupportedModelError):
            model.prepare(bad)

    def test_malformed_loop_carried_mapping_fails_open(self):
        text = self.ir_text("fused_attention", 1).replace(
            "%acc_ptr_86, %l_i_83, %m_ij_69, %K_block_ptr_66, %V_block_ptr_57",
            "%acc_ptr_86, %l_i_83, %K_block_ptr_66, %V_block_ptr_57",
            1,
        )
        prepared = prepare_cost_model(text, vf_merge_level=0)
        self.assertTrue(prepared.blockers)
        result = evaluate_configuration(
            prepared,
            intra_cache_num=3,
            multibuffer_num=4,
            baseline=None,
        )
        self.assertIsNone(result.total_from_11_bytes)
        self.assertEqual(result.verdict, "unknown")

    def test_ambiguous_region_local_ssa_names_fail_open(self):
        text = self.ir_text("fused_attention", 1).replace(
            "%alpha_82 =",
            "%alpha =",
            1,
        )
        prepared = prepare_cost_model(text, vf_merge_level=0)
        self.assertEqual(prepared.dynamic_buffers, ())
        self.assertTrue(any("ambiguous region-local SSA names" in item
                            for item in prepared.blockers))
        result = evaluate_configuration(
            prepared,
            intra_cache_num=3,
            multibuffer_num=4,
            baseline=None,
        )
        self.assertIsNone(result.total_from_11_bytes)
        self.assertEqual(result.verdict, "unknown")


class StructuralWhitelistTest(unittest.TestCase):
    def prepare(self, text: str, profile=DEFAULT_PROFILE):
        return prepare_cost_model(
            text,
            vf_merge_level=0,
            profile=profile,
        )

    def test_mark_gm_load_fixed_count_and_dedup(self):
        prepared = self.prepare(synthetic_boundary_ir(duplicate_copy=True))
        fixed = [item for item in prepared.excluded_buffers if item.classification == "fixed_gm_load"]
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0].fixed_count, 2)
        self.assertEqual(fixed[0].aligned_bytes, 4096)
        self.assertEqual(prepared.ordinary_step_table_bytes[0], 4096)

    def test_mark_gm_load_hint_and_function_exception_expose_ordinary_load(self):
        hinted = self.prepare(synthetic_boundary_ir(gm_load_hint=1))
        skip_profile = replace(DEFAULT_PROFILE, gm_load_skip_functions=("kernel_sdpa_fwd", ))
        skipped = self.prepare(
            synthetic_boundary_ir(function_name="kernel_sdpa_fwd"),
            profile=skip_profile,
        )
        for prepared in (hinted, skipped):
            ordinary_loads = [item for item in prepared.ordinary_buffers if item.path_kind == "gm_load"]
            self.assertEqual(len(ordinary_loads), 1)
            self.assertEqual(prepared.ordinary_step_table_bytes[0], 8192)

    def test_scf_while_is_supported(self):
        prepared = self.prepare(synthetic_boundary_ir(loop_kind="scf.while"))
        self.assertEqual(prepared.blockers, ())

    def test_unsupported_loop_fails_open(self):
        prepared = self.prepare(synthetic_boundary_ir(loop_kind="scf.parallel"))
        self.assertTrue(prepared.blockers)
        result = evaluate_configuration(
            prepared,
            intra_cache_num=1,
            multibuffer_num=4,
            baseline=None,
        )
        self.assertEqual(result.verdict, "unknown")
        self.assertIsNone(result.total_from_11_bytes)
        self.assertFalse(result.prune_allowed)

    def test_dynamic_shape_is_rejected(self):
        with self.assertRaises(UnsupportedModelError):
            self.prepare(synthetic_boundary_ir(dynamic_shape=True))


class ShapeValidationSupportTest(unittest.TestCase):
    def test_source_variants_are_exact_auditable_copies(self):
        original_path = REPO_ROOT / "experiment_operators/candidates/fused_attention.py"
        original = original_path.read_text(encoding="utf-8")
        old = "        alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max\n"
        new = "        alpha_base = tl.math.exp(m_i - m_ij)\n        alpha = alpha_base * 1.0\n"
        with TemporaryDirectory() as directory:
            output = Path(directory)
            variant_path, digest = _materialize_kernel_source(
                {
                    "case_id": "alpha_identity",
                    "source_edits": [{"old": old, "new": new}],
                },
                REPO_ROOT,
                output,
            )
            variant = variant_path.read_text(encoding="utf-8")
            self.assertNotEqual(variant, original)
            self.assertIn(new, variant)
            self.assertNotIn(old, variant)
            self.assertEqual(len(digest), 64)
            self.assertEqual(original_path.read_text(encoding="utf-8"), original)
            with self.assertRaises(ValueError):
                _materialize_kernel_source(
                    {
                        "case_id": "bad_edit",
                        "source_edits": [{"old": "not in the kernel", "new": "x"}],
                    },
                    REPO_ROOT,
                    output,
                )

    def test_fused_shape_overrides_recompute_contiguous_strides(self):
        _entry, _signature, constants = kernel_config(
            "fused_attention",
            {
                "Z": 2,
                "H": 16,
                "N_CTX": 512,
                "HEAD_DIM": 128,
                "BLOCK_M": 32,
                "BLOCK_N": 32,
            },
        )
        self.assertEqual(constants["stride_qm"], 128)
        self.assertEqual(constants["stride_qh"], 512 * 128)
        self.assertEqual(constants["stride_qz"], 16 * 512 * 128)
        self.assertEqual(constants["stride_vz"], constants["stride_qz"])
        with self.assertRaises(ValueError):
            kernel_config("fused_attention", {"UNKNOWN_SIZE": 32})
        with self.assertRaises(ValueError):
            kernel_config("fused_attention", {"N_CTX": 1000, "BLOCK_M": 64})
        with self.assertRaises(ValueError):
            kernel_config("fused_attention", {"HEAD_DIM": 48})
        with self.assertRaises(ValueError):
            kernel_config("fused_attention", {"STAGE": 2})
        _entry, _signature, flash_constants = kernel_config(
            "flash_attention_npu_v8", {"SPARSE_OPT": True})
        self.assertTrue(flash_constants["SPARSE_OPT"])
        with self.assertRaises(ValueError):
            kernel_config("flash_attention_npu_v8", {"SPARSE_OPT": 1})

    def test_reference_compiler_log_matches_cut_and_raw_alloc_oracles(self):
        plan = (MLIR_ROOT / "fused_attention" / "dynamic_1" /
                "after-plan-compute-block.mlir").read_text(encoding="utf-8")
        prepared = prepare_cost_model(plan, vf_merge_level=0)
        for dynamic in range(1, 4):
            with self.subTest(dynamic=dynamic):
                log = (MLIR_ROOT / "fused_attention" / f"dynamic_{dynamic}" /
                       "mlir-pass-dump.log").read_text(encoding="utf-8", errors="replace")
                self.assertTrue(_block_rewrites_match(log, prepared.dynamic_buffers))
                self.assertEqual(
                    _inner_alloc_counter(log),
                    _expected_inner_allocs(prepared.dynamic_buffers, dynamic),
                )


class PlanLiteTest(unittest.TestCase):
    def test_no_reuse_branch_is_exact(self):
        result = plan_lite(
            (
                MemoryEntry("a", 64, (LifeInterval(0, 3), ), multiplicity_kind="ordinary"),
                MemoryEntry("b", 32, (LifeInterval(1, 2), )),
            ),
            multibuffer_num=1,
            ub_capacity_bytes=128,
            entry_graph_complete=True,
        )
        self.assertEqual(result.status, "exact_no_reuse")
        self.assertEqual(result.predicted_peak_bytes, 96)
        self.assertTrue(result.reliable)
        self.assertFalse(result.prune_allowed)

    def test_disjoint_lifetimes_can_reuse_one_address(self):
        result = plan_lite(
            (
                MemoryEntry("a", 96, (LifeInterval(0, 2), )),
                MemoryEntry("b", 96, (LifeInterval(3, 5), )),
            ),
            multibuffer_num=1,
            ub_capacity_bytes=128,
            entry_graph_complete=True,
        )
        self.assertEqual(result.status, "feasible_placement")
        self.assertEqual(result.no_reuse_required_bytes, 192)
        self.assertEqual(result.live_lower_bound_bytes, 96)
        self.assertEqual(result.predicted_peak_bytes, 96)
        self.assertFalse(result.prune_allowed)

    def test_simultaneously_live_lower_bound_proves_overflow(self):
        result = plan_lite(
            (
                MemoryEntry("a", 96, (LifeInterval(0, 5), ), multiplicity_kind="ordinary"),
                MemoryEntry("b", 64, (LifeInterval(0, 5), )),
            ),
            multibuffer_num=2,
            ub_capacity_bytes=192,
            entry_graph_complete=True,
        )
        self.assertEqual(result.status, "proven_overflow")
        self.assertEqual(result.live_lower_bound_bytes, 256)
        self.assertTrue(result.prune_allowed)

    def test_incomplete_graph_never_prunes(self):
        result = plan_lite(
            (MemoryEntry("a", 256, (LifeInterval(0, 5), )), ),
            multibuffer_num=1,
            ub_capacity_bytes=128,
            entry_graph_complete=False,
        )
        self.assertFalse(result.prune_allowed)
        self.assertFalse(result.reliable)

    def test_alias_merge_uses_max_size_and_multiplicity(self):
        result = plan_lite(
            (
                MemoryEntry(
                    "a",
                    64,
                    (LifeInterval(0, 2), ),
                    multiplicity_kind="ordinary",
                    alias_group="x",
                ),
                MemoryEntry(
                    "b",
                    64,
                    (LifeInterval(3, 5), ),
                    multiplicity_kind="fixed",
                    fixed_count=3,
                    alias_group="x",
                ),
            ),
            multibuffer_num=2,
            ub_capacity_bytes=256,
            entry_graph_complete=True,
        )
        self.assertEqual(result.no_reuse_required_bytes, 192)
        self.assertEqual(len(result.placements), 3)

    def test_random_complete_graph_never_returns_an_invalid_proof_or_placement(self):
        rng = random.Random(7)
        for case_index in range(100):
            entries = tuple(
                MemoryEntry(
                    f"e{entry_index}",
                    rng.choice((32, 64, 96)),
                    (LifeInterval(start := rng.randrange(0, 8), start + rng.randrange(0, 4)), ),
                    multiplicity_kind="ordinary" if entry_index == 0 else "single",
                ) for entry_index in range(rng.randrange(1, 7)))
            multibuffer_num = rng.randrange(1, 5)
            capacity = rng.choice((96, 128, 160, 192, 256, 320))
            result = plan_lite(
                entries,
                multibuffer_num=multibuffer_num,
                ub_capacity_bytes=capacity,
                entry_graph_complete=True,
            )
            expanded = []
            for entry in entries:
                count = multibuffer_num if entry.multiplicity_kind == "ordinary" else 1
                expanded.extend((entry.entry_id, copy_index, entry) for copy_index in range(count))
            brute_lower_bound = max(
                sum(entry.size_bytes for _entry_id, _copy_index, entry in expanded
                    if entry.lifetimes[0].start <= point <= entry.lifetimes[0].end) for point in range(12))
            if result.prune_allowed:
                self.assertGreater(brute_lower_bound, capacity, msg=f"case {case_index}")
            if result.reliable and result.predicted_peak_bytes is not None:
                by_key = {(entry_id, copy_index): entry for entry_id, copy_index, entry in expanded}
                for index, left in enumerate(result.placements):
                    for right in result.placements[index + 1:]:
                        left_entry = by_key[(left.entry_id, left.copy_index)]
                        right_entry = by_key[(right.entry_id, right.copy_index)]
                        live_overlap = left_entry.lifetimes[0].overlaps(right_entry.lifetimes[0])
                        address_overlap = (left.offset_bytes < right.offset_bytes + right.size_bytes
                                           and right.offset_bytes < left.offset_bytes + left.size_bytes)
                        self.assertFalse(live_overlap and address_overlap, msg=f"case {case_index}")


if __name__ == "__main__":
    unittest.main()
