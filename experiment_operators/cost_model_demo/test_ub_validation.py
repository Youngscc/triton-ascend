"""Protect the compiler-oracle status and finite-difference comparisons."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from .run_ub_validation import compare_case, native_observation, effective_counts
from .stages.validate_context import DEFAULT_PROFILE, validate_context
from .model_types import UnsupportedModelError


class UBValidationTest(unittest.TestCase):
    def grid(self):
        rows=[]
        for d in (1,2,3):
            for m in (1,2,3,4):
                dynamic=2*(d-1); ordinary=3*(m-1); interaction=5*(d-1)*(m-1)
                rows.append(dict(dynamic_cv=d,multibuffer_num=m,
                    compiler_status="measured",compiler_ub_bytes=100+dynamic+ordinary+interaction,
                    normalized_plan_sha256="same-ir",requested_dynamic_resolved=True,profile_options_match=True,
                    model_delta_bytes=dynamic+ordinary+interaction,
                    model_dynamic_delta_bytes=dynamic,model_multibuffer_delta_bytes=ordinary,
                    model_interaction_bytes=interaction))
        return rows

    def test_joint_differences_preserve_nonzero_interaction(self):
        rows=self.grid(); compare_case(rows)
        self.assertTrue(all(r["validation_outcome"]=="exact_prediction" for r in rows))
        self.assertEqual(rows[-1]["compiler_interaction_bytes"],30)

    def test_missing_baseline_does_not_become_zero(self):
        rows=self.grid(); rows[0]["compiler_ub_bytes"]=None; compare_case(rows)
        self.assertTrue(all(r["validation_outcome"]=="compiler_incomplete" for r in rows))

    def test_changed_ir_invalidates_a_coincidental_numerical_match(self):
        rows=self.grid(); rows[-1]["normalized_plan_sha256"]="different-ir"; compare_case(rows)
        self.assertTrue(all(r["validation_outcome"]=="unsupported_config" for r in rows))

    def test_changed_effective_options_cannot_validate_the_profile(self):
        rows=self.grid(); rows[-1]["profile_options_match"]=False; compare_case(rows)
        self.assertEqual(rows[-1]["validation_outcome"],"unsupported_config")

    def test_ir_counts_override_requested_metadata(self):
        counts=effective_counts("module attributes {ssbuffer.intra_buf_count = 2 : i32, "
            "ssbuffer.inter_core_buf_count = 1 : i32, ssbuffer.load_store_buf_count = 1 : i32}")
        self.assertEqual(counts["buf_slot_num_of_veccore"],2)
        rows=self.grid(); rows[0]["requested_dynamic_resolved"]=False; compare_case(rows)
        self.assertTrue(all(r["validation_outcome"]=="unsupported_config" for r in rows))

    def test_wrong_model_prediction_is_retained_as_mismatch(self):
        rows=self.grid(); rows[-1]["model_delta_bytes"]+=32; compare_case(rows)
        self.assertEqual(rows[-1]["validation_outcome"],"mismatch")
        self.assertEqual(rows[-1]["error_bytes"],32)

    def test_ub_from_a_failed_native_compile_is_not_a_measurement(self):
        result=native_observation(1,"Allocated UB size = 256 bits\n")
        self.assertEqual(result["compiler_status"],"compile_failed")
        self.assertIsNone(result["compiler_ub_bytes"])
        self.assertEqual(result["ub_observations_bits"],[256])

    def test_multiple_functions_use_the_maximum_successful_span(self):
        result=native_observation(0,"UB size = 128 bits\nUB size = 256 bits\n")
        self.assertEqual(result["compiler_ub_bytes"],32)
        self.assertEqual(native_observation(0,"UB size = 0 bits")["compiler_status"],"ub_missing")

    def test_native_identity_ignores_printed_names_but_requires_every_row(self):
        rows=self.grid()
        for i,row in enumerate(rows):
            row["normalized_plan_sha256"]=f"names-{i}"
            row["structural_plan_sha256"]="same-structure"
        compare_case(rows)
        self.assertTrue(all(r["validation_outcome"]=="exact_prediction" for r in rows))
        del rows[-1]["structural_plan_sha256"]
        compare_case(rows)
        self.assertTrue(all(r["validation_outcome"]=="unsupported_config" for r in rows))

    def test_unmodeled_operand_substitution_is_rejected(self):
        with self.assertRaisesRegex(UnsupportedModelError,"enable_vf_operand_substitution"):
            validate_context("",SimpleNamespace(target="Ascend950"),0,
                             replace(DEFAULT_PROFILE,enable_vf_operand_substitution=True),None)

    def test_operand_substitution_changes_the_profile_identity(self):
        enabled=replace(DEFAULT_PROFILE,enable_vf_operand_substitution=True)
        self.assertNotEqual(DEFAULT_PROFILE.fingerprint,enabled.fingerprint)


if __name__=="__main__":
    unittest.main()
