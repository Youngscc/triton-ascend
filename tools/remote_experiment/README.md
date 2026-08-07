# Local-to-server experiment loop

This uses an SSH alias, synchronizes a local checkout into a configurable
server checkout, and runs commands inside an existing experiment container.
Configure the two checkout paths once by editing
`tools/remote_experiment/config.sh`:

```bash
LOCAL_PROJECT="/local/path/to/triton-ascend"
REMOTE_PROJECT="/server/path/to/triton-ascend"
```

The SSH alias `huawei-server` and container name `sgl-sky` are fixed by the
workflow and do not need to be exported.

See `experiment_operators/EXECUTION_GUIDE.md` for initial SSH-key and
`~/.ssh/config` setup. The container must be able to see `REMOTE_PROJECT` at
the same absolute path as the server host (the current container mounts the
server's `/home`).

From the repository root:

```bash
# 1. Sync source. Generated outputs and .git metadata are excluded.
./tools/remote_experiment/sync.sh

# 2. Rebuild the custom BishengIR compiler after changing AscendNPU-IR C++.
./tools/remote_experiment/rebuild-compiler.sh

# 3. Start a detached experiment with the current source and custom compiler.
REMOTE_MODE=dev ./tools/remote_experiment/run.sh \
  python -u path/to/experiment.py --arg value

# 4. Follow the newest log. Ctrl-C only stops log following, not the job.
./tools/remote_experiment/logs.sh

# Follow a specific run printed by run.sh:
./tools/remote_experiment/logs.sh 20260803-180000-12345
```

Pull generated result directories from the server back into the local checkout:

```bash
./tools/remote_experiment/pull-results.sh
```

This uses `--progress` for compatibility with the older macOS rsync. Set
`PULL_SESSION_LOGS=1` to pull `.codex-remote/logs` too. Result pulling is
additive by default; `RSYNC_DELETE=1` makes the server result directory an exact
local mirror and may remove local-only historical results.

The default sync is additive. To mirror local deletions on the server, use
`RSYNC_DELETE=1 ./tools/remote_experiment/sync.sh`; this removes remote files
under the target that are absent locally.

`run.sh` uses the container's original preinstalled environment by default.
The isolated development environment is selected only with
`REMOTE_MODE=dev`:

- Python environment: the path selected by `REMOTE_VENV`
- Python and Ascend backend: this repository's `python/` tree
- BishengIR compiler: `.codex-remote/ascendnpu-ir-build-explicit/bin`
- Triton cache: `.codex-remote/triton-cache`

The container's preinstalled Torch, torch-npu, CANN, and compatible
`libtriton.so` remain the baseline. The remote-only build, cache, and logs are
excluded from rsync. AscendNPU-IR and its vendored LLVM are exact-mirrored from
the local gitlinks; mixing a new AscendNPU-IR source tree with an older patched
LLVM tree causes C++ API/ABI build failures.

`rebuild-compiler.sh` configures the pinned source with the Triton and A5/NPUIR
switches used by the upstream build, then builds `bishengir-compile`. It also
generates the matching meta-op/host bitcode from the pinned Template sources
using CANN's `ccec`. The current compiler looks for `meta_op.*.c220.bc`; if a
generated file is unexpectedly absent, the script falls back to a private link
to CANN 9.0's legacy-named `meta_op.*.bc`. No system file is renamed or
replaced. Override `REMOTE_BISHENG_COMPILER_BIN` or
`REMOTE_SYSTEM_COMPILER_LIB` when a container stores these tools elsewhere.

There is also an explicit compatibility mode:

```bash
REMOTE_MODE=dev-compatible ./tools/remote_experiment/run.sh \
  python -u experiment_operators/candidates/fused_attention.py
```

`dev-compatible` loads the current repository's Triton Python/core and Ascend
backend, but uses the CANN-matched preinstalled BishengIR 1.1 and `hivmc` 0.2.
Use it for end-to-end Python/correctness/benchmark smoke tests. It does not
prove that options implemented only in the custom BishengIR 1.2 are available.
The three modes use separate Triton cache subdirectories.

To run a control with the container's fully preinstalled Triton toolchain,
either omit `REMOTE_MODE` or set it explicitly:

```bash
REMOTE_MODE=baseline ./tools/remote_experiment/run.sh \
  python -u third_party/ascend/tutorials/01-vector-add.py
```

`REMOTE_MODE=dev` loads the current source tree and the custom BishengIR build.
The default baseline mode deliberately loads neither of them.

Useful overrides are available without editing files:

```bash
REMOTE_CONTAINER=other-container ./tools/remote_experiment/run.sh bash -lc '...'
LINES=200 ./tools/remote_experiment/logs.sh latest
```
