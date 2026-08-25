import importlib.util
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class CompilerCostmodelContractTest(unittest.TestCase):

    @staticmethod
    def _load_compiler_module():
        ctypes_stub = types.ModuleType("ctypes")
        ctypes_stub.c_int64 = int
        sys.modules["ctypes"] = ctypes_stub

        class Dummy:

            def __getattr__(self, _):
                return Dummy()

            def __call__(self, *args, **kwargs):
                return Dummy()

        triton_mod = types.ModuleType("triton")
        triton_c_mod = types.ModuleType("triton._C")
        ascend_backend_mod = types.ModuleType("triton.backends.ascend")
        ascend_backend_mod.__path__ = []
        ascend_backend_mod._apply_ascend_patch = lambda: None
        debug_line_rewriter_mod = types.ModuleType("triton.backends.ascend.debug_line_rewriter")
        debug_line_rewriter_mod.rewrite_debug_line = lambda artifact, **_kwargs: artifact
        libtriton_mod = types.ModuleType("triton._C.libtriton")
        libtriton_ascend_mod = types.ModuleType("triton._C.libtriton.ascend")
        libtriton_ascend_mod.ir = Dummy()
        libtriton_mod.ir = Dummy()
        libtriton_mod.passes = Dummy()
        libtriton_mod.ascend = libtriton_ascend_mod
        libtriton_mod.buffer_ir = Dummy()
        utils_mod = types.ModuleType("triton.backends.ascend.utils")
        for name in [
                "_check_bishengir_api_change",
                "_check_bishengir_able_save_ir",
                "_check_bishengir_is_regbased",
                "_enable_print_ub_bits",
                "_enable_dump_memory_info",
                "_enable_msdebug",
                "_get_kernel_target",
                "_get_npucompiler_path",
                "_get_triton_adapter_opt_path",
                "_get_triton_mlir_opt_path",
                "_get_triton_opt_path",
                "_get_bishengir_opt_path",
                "_is_ascend_sanitizer_enabled",
                "_is_debug_line_info_disabled",
                "_is_auto_map_parallel_blocks_enabled",
                "_get_auto_blockify_blacklist_reasons",
                "_warn_auto_blockify_disabled",
                "downgrade_llir",
                "force_disable_ffts",
                "get_cann_version_file_hash",
        ]:
            setattr(utils_mod, name, lambda *args, **kwargs: False)
        utils_mod._get_auto_blockify_blacklist_reasons = lambda *args, **kwargs: []
        utils_mod._is_auto_map_parallel_blocks_enabled = lambda *args, **kwargs: False
        utils_mod._warn_auto_blockify_disabled = lambda *args, **kwargs: None
        def remove_deprecated_npu_options(options, *, in_place=False):
            normalized = options if in_place else dict(options)
            for old_name, new_name in {
                    "intra_cache_num": "buf_slot_num_of_veccore",
                    "inter_cache_num": "buf_slot_num_of_crosscore",
                    "load_cache_num": "buf_slot_num_of_gm",
            }.items():
                if old_name in normalized:
                    normalized.setdefault(new_name, normalized[old_name])
                    normalized.pop(old_name)
            return normalized

        utils_mod._remove_deprecated_npu_options = remove_deprecated_npu_options
        utils_mod._warn_deprecated_npu_option = lambda *_args, **_kwargs: None
        utils_mod._warn_deprecated_ascend_env_vars = lambda: None
        utils_mod.get_cann_version_file_hash = lambda *args, **kwargs: ""
        utils_mod.graph_ub_budget_bytes_for_arch = lambda *args, **kwargs: 0

        driver_mod = types.ModuleType("triton.backends.ascend.driver")
        driver_mod.NPUUtils = Dummy

        compiler_base_mod = types.ModuleType("triton.backends.compiler")

        class BaseBackend:

            def __init__(self, target):
                self.target = target

        class GPUTarget:

            def __init__(self, backend="npu", arch="910B"):
                self.backend = backend
                self.arch = arch

        compiler_base_mod.AttrsDescriptor = Dummy
        compiler_base_mod.BaseBackend = BaseBackend
        compiler_base_mod.GPUTarget = GPUTarget
        compiler_base_mod.register_descriptor = lambda cls: cls

        runtime_mod = types.ModuleType("triton.runtime")
        runtime_mod.driver = Dummy()

        cache_mod = types.ModuleType("triton.runtime.cache")

        class DumpManager:

            def __init__(self):
                self.cache_dir = "/tmp/fake_cache"
                self.records = []

            def put(self, payload, file_name, binary=False):
                self.records.append((payload, file_name, binary))

        dump_mgr = DumpManager()
        cache_mod.get_dump_manager = lambda *args, **kwargs: dump_mgr
        cache_mod._base32 = lambda value: value

        utils_mod.is_compile_on_910_95 = lambda *_args: False

        sys.modules.update({
            "triton": triton_mod,
            "triton._C": triton_c_mod,
            "triton._C.libtriton": libtriton_mod,
            "triton._C.libtriton.ascend": libtriton_ascend_mod,
            "triton.backends.ascend": ascend_backend_mod,
            "triton.backends.ascend.debug_line_rewriter": debug_line_rewriter_mod,
            "triton.backends.ascend.utils": utils_mod,
            "triton.backends.ascend.driver": driver_mod,
            "triton.backends.compiler": compiler_base_mod,
            "triton.runtime": runtime_mod,
            "triton.runtime.cache": cache_mod,
        })

        module_path = Path(__file__).resolve().parents[2] / "backend" / "compiler.py"
        module_name = "triton.backends.ascend.compiler_costmodel_contract_under_test"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, dump_mgr, GPUTarget

    def test_obsolete_costmodel_and_bytecode_switches_are_not_npu_options(self):
        cmplr, _dump_mgr, GPUTarget = self._load_compiler_module()

        backend = cmplr.AscendBackend(GPUTarget(backend="npu", arch="910B"))
        options = backend.parse_options({
            "enable_costmodel_backend": True,
            "use_bytecode": True,
        })

        self.assertFalse(hasattr(options, "enable_costmodel_backend"))
        self.assertFalse(hasattr(options, "use_bytecode"))

    def test_experiment_options_preserve_main_dev_controls(self):
        cmplr, _dump_mgr, GPUTarget = self._load_compiler_module()

        backend = cmplr.AscendBackend(GPUTarget(backend="npu", arch="Ascend950PR_9579"))
        raw_options = {
            "enable_dynamic_cv_pipeline": True,
            "intra_cache_num": 4,
            "inter_cache_num": 1,
            "load_cache_num": 1,
            "multibuffer_num": 3,
            "vf_merge_level": 1,
        }
        options = backend.parse_options(raw_options)

        self.assertTrue(options.enable_dynamic_cv_pipeline)
        self.assertEqual(options.buf_slot_num_of_veccore, 4)
        self.assertEqual(options.buf_slot_num_of_crosscore, 1)
        self.assertEqual(options.buf_slot_num_of_gm, 1)
        self.assertEqual(options.multibuffer_num, 3)
        self.assertEqual(options.limit_auto_multi_buffer_buffer, "no-limit")
        self.assertEqual(options.vf_merge_level, 1)
        self.assertNotIn("intra_cache_num", raw_options)
        self.assertNotIn("inter_cache_num", raw_options)
        self.assertNotIn("load_cache_num", raw_options)

        with self.assertRaises(ValueError):
            cmplr.NPUOptions(multibuffer_num=0)
        with self.assertRaises(ValueError):
            cmplr.NPUOptions(vf_merge_level=3)

    def test_bytecode_writer_targets_bishengir_compatible_version(self):
        cmplr, _dump_mgr, _GPUTarget = self._load_compiler_module()

        def fake_run(command, **_kwargs):
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_bytes(b"MLIR-bytecode")

        with mock.patch.object(
                cmplr,
                "_get_triton_mlir_opt_path",
                return_value="/llvm22/bin/triton-mlir-opt",
        ), mock.patch.object(cmplr.subprocess, "run", side_effect=fake_run) as run:
            result = cmplr.linalg_to_bc_by_triton_mlir_opt(
                "module {}\n",
                {"hash": "test"},
                types.SimpleNamespace(debug=False),
            )

        self.assertEqual(result, b"MLIR-bytecode")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/llvm22/bin/triton-mlir-opt")
        self.assertIn("--emit-bytecode", command)
        self.assertIn("--emit-bytecode-version=4", command)

    def test_bishengir_failure_includes_captured_diagnostics(self):
        cmplr, _dump_mgr, _GPUTarget = self._load_compiler_module()
        error = subprocess.CalledProcessError(
            7,
            ["/project/bin/bishengir-compile", "kernel.mlir"],
            output=b"compiler stdout",
            stderr=b"actual pass diagnostic",
        )

        message = cmplr._format_bishengir_compile_failure(error)

        self.assertIn("returncode: 7", message)
        self.assertIn("bishengir-compile kernel.mlir", message)
        self.assertIn("compiler stdout", message)
        self.assertIn("actual pass diagnostic", message)


if __name__ == "__main__":
    unittest.main()
