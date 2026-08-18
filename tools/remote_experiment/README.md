# Server-native experiment workflow

The server checkout is the primary working tree. Source updates normally use
Git directly on the server. Triton-Ascend builds, BishengIR builds, experiments,
log following, and report generation all run on the server; build and foreground
experiment commands run inside the existing experiment container.

The scripts default to the current checkout and container name
`triton-ascend-exp`, so no local configuration is needed for that layout. To
override either value, create the ignored local configuration:

```bash
cd /absolute/server/path/to/triton-ascend
cp tools/remote_experiment/config.local.sh.example \
  tools/remote_experiment/config.local.sh
vi tools/remote_experiment/config.local.sh
```

The optional overrides are:

```bash
REMOTE_PROJECT="/absolute/server/path/to/triton-ascend"
REMOTE_CONTAINER="existing-container-name"
```

The project must be mounted at the same absolute path on the server host and
inside the container. `LOCAL_PROJECT` and `REMOTE_HOST` are optional and are
used only by the offline source-transfer and result-retrieval helpers.

## Update source on the server

Run in the server checkout:

```bash
git fetch origin
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive
```

Use `sync.sh` from a workstation only when the server cannot access GitHub. The
fallback transfer excludes `.codex-remote`, virtual environments, build/cache
directories, Python caches, coverage/test caches, generated packages, and
locally built binaries under `python/triton/_C`. Tracked Python sources are
transferred, including `python/build_helpers.py` and the Triton package source.
The cache/build exclusion policy is also applied to AscendNPU-IR. The fallback
still transfers the Git metadata needed by the build without replacing server
experiment artifacts. Set
`REMOTE_SOURCE_MODE="rsync"` in the workstation's `config.local.sh` for this
fallback; a normal server clone uses the default `auto` mode and its own `.git`.

Excluded files already present on the server are deliberately preserved. This
includes locally built files under `python/triton/_C`, the venv, compiler build,
cache, and experiment results.
Do not add `--delete-excluded` to `sync.sh`. If an older sync already copied a
workstation binary such as `python/triton/_C/libtriton.so`, run the documented
`clean-environment.sh rebuild --execute` step once inside the server container
and rebuild there.

## Create or repair the development environment

Enter the existing container from the server host:

```bash
source tools/remote_experiment/config.sh
docker exec -u root -it "$REMOTE_CONTAINER" bash
cd "$REMOTE_PROJECT"
source tools/remote_experiment/config.sh
```

When changing to a container with a different Python or CANN toolchain, preview
and remove the old project build environment before rebuilding:

```bash
./tools/remote_experiment/clean-environment.sh rebuild
./tools/remote_experiment/clean-environment.sh rebuild --execute
```

Use `runtime` to remove experiment logs and Triton caches. Use `results` to
remove all stored measurements and generated reports, or `all` to remove every
listed build/runtime/result artifact. Cleanup is preview-only unless
`--execute` is present. Source files, Git metadata, the offline LLVM directory,
`top-git`, and `config.local.sh` are always preserved.

To retain only the latest complete full sweep for each operator, together with
the generated `latest-summary`, run:

```bash
./tools/remote_experiment/clean-environment.sh latest-results
./tools/remote_experiment/clean-environment.sh latest-results --execute
```

This uses the same completeness and timestamp rules as `summarize_latest.py`.
It refuses to delete anything when no complete sweep exists and preserves
unrecognized directories.

Run the idempotent setup inside the container:

```bash
JOBS=32 ./tools/remote_experiment/setup-dev-environment.sh
JOBS=32 ./tools/remote_experiment/rebuild-compiler.sh
source tools/remote_experiment/activate-dev-environment.sh
```

When the container cannot reach the Triton LLVM artifact server, stage the
repository-selected prebuilt LLVM under the persistent project directory and
pass it only to the Triton build:

```bash
TRITON_OFFLINE_BUILD=1 \
LLVM_SYSPATH="/absolute/project/path/.codex-remote/llvm/<matching-llvm-directory>" \
TRITON_PARALLEL_LINK_JOBS=2 \
JOBS=4 ./tools/remote_experiment/setup-dev-environment.sh
```

When `LLVM_SYSPATH` is set, the setup validates `bin/FileCheck`,
`lib/cmake/mlir/MLIRConfig.cmake`, and
`lib/cmake/lld/LLDConfig.cmake`, then passes the MLIR and LLD package
directories explicitly to CMake. The path must be the LLVM installation root,
not its parent directory.

`setup-dev-environment.sh` creates `.codex-remote/venv` when it is absent,
installs a private CMake 3.28+ only when required, builds this checkout's
Triton-Ascend and `libtriton.so`, performs the editable install, and verifies
the import paths. It uses the server clone's own `.git`; the mirrored
`.codex-remote/top-git` is only a compatibility fallback for offline rsync.
The host build requires Clang and automatically prefers the Ubuntu
clang-15/clang++-15/lld-15 tools, including their version-suffixed paths. It
does not fall back to GCC. The CANN device compiler path is unchanged.

The setup exits unsuccessfully if the project MLIR 22 tool cannot emit
bytecode version 4 containing `llvm.inttoptr` that the bootstrap MLIR 19
BishengIR reader can consume, if Python's `sys.prefix` is not the project venv, or if either
`triton` or `libtriton` resolves outside
the current checkout. A successful run prints `MLIR_BYTECODE_ROUNDTRIP_OK` and
ends with `TRITON_DEV_IMPORT_OK`. Because the venv
uses `--system-site-packages`, a manual development invocation must put
`$REMOTE_PROJECT/python` first in `PYTHONPATH`; `run_all_sweeps.sh` and
`REMOTE_MODE=dev` do this automatically.

`rebuild-compiler.sh` builds both `bishengir-compile` and `bishengir-opt` from
the repository-pinned AscendNPU-IR. It additionally round-trips
`#hivm.address_space<ssbuf>` from the checkout's MLIR 22 writer through the
project-built MLIR 19 reader and prints `HIVM_SSBUF_BYTECODE_ROUNDTRIP_OK`.
Development experiments require that project-built reader; CANN's older
`bishengir-opt` does not recognize DynamicCV's SSBUF enum. CANN still supplies
`hivmc` and the downstream device compiler.

The venv interpreter may be a symlink to `/usr/local/bin/python`. That resolved
file location is normal and is not used to decide whether the venv is active;
the setup checks `sys.prefix` instead.

Do not copy a venv from another host, container, or project path. A Python venv
contains interpreter and script paths and must be created inside the container
at the final mounted project path.

After both builds, source `activate-dev-environment.sh` once in each new
container shell. It activates the project venv, places the checkout's Python
tree and BishengIR compiler first, and detects the NPU architecture. On A5 it
also enables the native project RegBase pipeline; on A3 it leaves that mode
disabled. A successful activation prints `DEV_ENVIRONMENT_OK`.

## Run experiments

Foreground execution runs inside the container:

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

Detached execution starts from the server host and calls its local Docker
daemon:

```bash
ASCEND_RT_VISIBLE_DEVICES=<physical-device-id> REMOTE_MODE=dev \
  ./tools/remote_experiment/run.sh \
  ./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py

./tools/remote_experiment/logs.sh latest
```

`run.sh` writes host-visible logs under `.codex-remote/logs`. `Ctrl-C` stops
only log following. Failed candidates print their complete subprocess output
in the session log and retain a separate per-configuration log.

## Reports and optional result retrieval

Generate the latest report inside the container:

```bash
source .codex-remote/venv/bin/activate
python experiment_operators/summarize_latest.py
./experiment_operators/generate_latest_report.sh
```

Results remain under `.codex-remote/results` on the server. If a workstation
copy is needed, configure the optional `LOCAL_PROJECT` and `REMOTE_HOST` values
there and run `./tools/remote_experiment/pull-results.sh` from the workstation.

See `experiment_operators/EXECUTION_GUIDE.md` for the complete A3/A5 procedure.
