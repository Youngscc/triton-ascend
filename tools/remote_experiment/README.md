# Experiment environment scripts

The experiment runs from the project checkout mounted into a CANN container. The complete procedure is in `experiment_operators/EXECUTION_GUIDE.md`; this page only identifies the scripts used by that procedure.

## Container baseline

A5 uses the `linux/amd64` devel image
`quay.io/ascend/cann:9.1.0-950-ubuntu22.04-py3.12-devel`. Create an experiment
container from that image with the NPU and project paths required by the host.
Container names and device IDs are intentionally not part of this repository.

## Build inside the container

From the project root:

```bash
./tools/remote_experiment/setup-dev-environment.sh
./tools/remote_experiment/rebuild-compiler.sh
source tools/remote_experiment/activate-dev-environment.sh
```

Script roles:

| Script | Purpose |
| --- | --- |
| `setup-dev-environment.sh` | Create or repair `.codex-remote/venv` and build this checkout's Triton Python/core. |
| `rebuild-compiler.sh` | Build the pinned AscendNPU-IR compiler and architecture-matched bitcode package. |
| `activate-dev-environment.sh` | Select the project venv, backend, compiler, CANN tools, SoC, and project-local temporary directory for the current shell. |
| `load-cann-environment.sh` | Locate and source the installed CANN environment. |
| `config.sh` | Define repository-local build, cache, result, and temporary paths. |

Build products, caches, temporary files, and experiment results remain under `.codex-remote/` or `tmp/`; they are not written into tracked source directories.

## Offline source sync

For an rsync-managed environment machine, preview the deletion-enabled transfer first:

```bash
RSYNC_DELETE=1 RSYNC_DRY_RUN=1 ./tools/remote_experiment/sync.sh
```

After checking the itemized output, run it without the preview flag:

```bash
RSYNC_DELETE=1 ./tools/remote_experiment/sync.sh
```

Deletion is restricted to `experiment_operators/`, `tools/remote_experiment/`, and the top-level DynamicCVPipeline `include/lib` source. The AscendNPU-IR allowlist is `bishengir/{cmake,include,lib,python,test,tools,unittests}` plus `bishengir/triton/{bin,cmake,include,lib,test,unittest,utils}`. AscendNPU-IR's `third-party/` dependencies, candidate kernels, archived originals, `config.local.sh`, caches, logs, results, profiles, artifacts, and temporary files are protected. The project root, `python/`, every non-allowlisted `third_party/` path, `.codex-remote/`, and every other directory remain additive-only. The `.codex-remote/top-git` metadata mirror remains independently replaceable and cannot remove build or experiment output.

## Run the experiment

```bash
./run_all_sweeps.sh experiment_operators/candidates/fused_attention.py
```

Parameter values are edited only in `experiment_operators/experiment_config.py`. The command automatically writes the complete result set and refreshes the HTML report.
