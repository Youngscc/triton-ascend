# VFMerge level-1 before/after IR

These files were regenerated locally from the unmodified regression inputs
with a compiler built from the latest fetched `upstream/main-dev` revision on
2026-09-03:

```text
triton-ascend: 13479dbed1c68c83cd64b4eafaf3853e24f34985
AscendNPU-IR:  aea934a66646e837c54fea11e87db54d42eb3221
LLVM:          d3fea2c7ae5436f63fa35b4d01e0aa76d1071396

/private/tmp/triton-ascend-main-dev-vfmerge/third_party/ascend/AscendNPU-IR/build-vfmerge/bin/bishengir-opt
bishengir-opt 1.2.0 (aea934a66646, 2026-09-03)
llvm 19.1.7 d3fea2c7ae54
Release build

bishengir-opt SHA-256:
077a66b7736fdb57acc07b8cac5ce54fdf3ceca3c6913324dbef051915a53cc8
```

For each case, `*.before.mlir` is parsed and reprinted without an optimization
pass, while `*.after.mlir` is produced with:

```text
--hfusion-merge-vf="merge-level=1"
```

The corresponding `*.diff` is a unified before/after diff.

## Files

- `flash_attention_hstu.before.mlir`
- `flash_attention_hstu.after.mlir`
- `flash_attention_hstu.diff`
- `dependency_update.before.mlir`
- `dependency_update.after.mlir`
- `dependency_update.diff`
- `extract_kind.before.mlir`
- `extract_kind.after.mlir`
- `extract_kind.diff`

The source tests containing `// -----` separators were processed with
`--split-input-file`, matching their lit `RUN` directives.

## Input compatibility

The latest compiler parses the Flash Attention/HSTU fixture's original A5
target entry directly:

```text
#dlti.dl_entry<"ARCH", "dav-c310">
```

No target metadata, function, operation, SSA edge, VF attribute, type, or
memory access was removed or rewritten in the input before running VFMerge.

All three generated `after` files pass their original `FileCheck` assertions
using the `FileCheck` binary from the same build.

## Diff size

| Case | Insertions | Deletions |
| --- | ---: | ---: |
| Flash Attention/HSTU | 116 | 140 |
| Dependency update | 7 | 15 |
| Extract kind | 9 | 17 |
