# VFMerge PR/MR regression cases

AutoVectorizeV2、VF 的形成过程和 VFMerge Level 1 的算子案例分析见
[`AUTOVECTORIZE_AND_VFMERGE.md`](../AUTOVECTORIZE_AND_VFMERGE.md)。

This directory contains compiler-level MLIR cases selected from the
AscendNPU-IR VFMerge (`hfusion-merge-vf` / `MergeVecScope`) history.  The files
are copied here so experiments can use stable inputs without modifying the
submodule's tests.

| Local file | Operator or scenario | Merge level | Historical source |
| --- | --- | --- | --- |
| `flash_attention_hstu_level1.mlir` | Flash Attention forward plus HSTU attention backward ordering/synchronization cases | 1 | Present in the initial A5 snapshot (`ca90b6e3b`); migrated to the current branch by `4ddead06f` |
| `rmsnorm_level2.mlir` | RMSNorm f32 split-N with four outlined vector functions | 2 | Present in the initial A5 snapshot (`ca90b6e3b`); migrated to the current branch by `4ddead06f` |
| `dependency_update_level1.mlir` | Producer, zero-init, and two-reduction dependency-graph regression | 1 | AscendNPU-IR-Dev MR `!1772`, commits `324031a7f` / `d4791607f` |
| `extract_kind_level1.mlir` | Merge eligibility for matching and mismatching extract kinds | 1 | AscendNPU-IR-Dev MR `!1350`, commit `b6627d3fe` |

The first two files are the best operator-shaped inputs for comparing VFMerge
levels.  The final two are smaller regression inputs for validating merge
legality and dependency bookkeeping.

Each file retains its original `RUN` and `FileCheck` directives.  The two
operator-shaped files are marked `REQUIRES: regbase`, because their imported
A5 target descriptions and pass preconditions require the RegBase build.

## SHA-256

```text
aa6ad021284035e72a5e42793913c4265b59bf822c00c22b7c77c9dcf4fd32be  dependency_update_level1.mlir
b3d68c321d6a4d9c49502be0a423c760df22c4407ad756957aefddb7ee19fa9a  extract_kind_level1.mlir
d88c07e8186cdc8fbe18dd14566c874f30316d105be8a20428a13283d93a022e  flash_attention_hstu_level1.mlir
9f89c7ee4f0f6c095d584e3c2039705652afae17eac3de52dd9097aff5c02d5c  rmsnorm_level2.mlir
```
