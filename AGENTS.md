# Project memory: local Codex + remote Ascend experiments

This repository is developed locally and experiments run on `huawei-server`.
Read this file before changing the remote-experiment workflow.

## Repositories and paths

- Local checkout: `/Users/YokeLove/huawei/triton-ascend`
- Fork remote: `origin` (`Youngscc/triton-ascend`)
- Source remote: `upstream` (`triton-lang/triton-ascend`)
- Experiment branch: local `codex/experiment`, tracking `origin/experiment`
- Server SSH alias: `huawei-server`
- Server project path: `/home/yuanye/code/triton-ascend`
- Experiment container: `sgl-sky`

The `sgl-sky` container bind-mounts the host's `/home`, so the server project
path is also the project path inside the container. Do not use `docker cp` for
normal source synchronization.

## Standard experiment loop

Run these commands from the repository root:

```bash
./tools/remote_experiment/sync.sh
./tools/remote_experiment/run.sh python -u path/to/experiment.py --arg value
./tools/remote_experiment/logs.sh latest
```

`run.sh` starts a detached command inside `sgl-sky` and prints a run ID, log
path, and host PID. `logs.sh` follows the newest log with `tail -F`; pressing
Ctrl-C only stops log following. The container is intentionally invoked with
non-login `bash -c`; this image's `bash -lc` initialization can block.

The scripts and their full options are documented in
`tools/remote_experiment/README.md`.

## Synchronization safety

The default rsync is additive and excludes `.git`, `.codex-remote`, virtual
environments, build/cache directories, and generated experiment output. Keep
this default for normal iterations. `RSYNC_DELETE=1` enables
`--delete-delay` and can remove files under the server target that are absent
locally; use it only when an exact mirror is intended.

The first sync can traverse many files because this project contains nested
third-party source trees. Later syncs are incremental. Avoid syncing large
generated snapshots or output directories into Git or the experiment mirror.

## Submodule caution

`third_party/ascend/AscendNPU-IR` is a Git submodule with further nested
third-party submodules. A dirty submodule status does not necessarily mean
the top-level Triton source changed. Inspect the nested repository before
staging a submodule pointer, and never reset or commit a large nested deletion
without confirming it is intentional.
