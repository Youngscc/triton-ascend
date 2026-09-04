# Server-native experiment workflow

The server checkout is the primary working tree. Source updates normally use
Git directly on the server. Triton-Ascend builds, BishengIR builds, experiments,
log following, and report generation all run on the server; build and foreground
experiment commands run inside the existing experiment container.

Create the ignored server configuration once in the server checkout:

```bash
cd /absolute/server/path/to/triton-ascend
cp tools/remote_experiment/config.local.sh.example \
  tools/remote_experiment/config.local.sh
vi tools/remote_experiment/config.local.sh
```

The required values are:

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
fallback transfer excludes `.codex-remote`, `offline-wheel/`, virtual
environments, build/cache directories, and generated results. It also
transfers the Git metadata needed
by the build without replacing server experiment artifacts. Set
`REMOTE_SOURCE_MODE="rsync"` in the workstation's `config.local.sh` for this
fallback; a normal server clone uses the default `auto` mode and its own `.git`.

## Create or repair the development environment

Enter the existing container from the server host:

```bash
source tools/remote_experiment/config.sh
docker exec -u root -it "$REMOTE_CONTAINER" bash
cd "$REMOTE_PROJECT"
source tools/remote_experiment/config.sh
```

Run the idempotent setup inside the container:

```bash
JOBS=32 ./tools/remote_experiment/setup-dev-environment.sh
JOBS=32 ./tools/remote_experiment/rebuild-compiler.sh
```

`setup-dev-environment.sh` creates `.codex-remote/venv` when it is absent,
installs a private CMake 3.28+ only when required, builds this checkout's
Triton-Ascend and `libtriton.so`, performs the editable install, and verifies
the import paths. It uses the server clone's own `.git`; the mirrored
`.codex-remote/top-git` is only a compatibility fallback for offline rsync.
The host build prefers a complete Clang/Lld pair and otherwise uses GCC/G++
with the default linker; the CANN device compiler path is unchanged.

Do not copy a venv from another host, container, or project path. A Python venv
contains interpreter and script paths and must be created inside the container
at the final mounted project path.

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

See `experiment_operators/EXECUTION_GUIDE.md` for the complete general/A3
procedure and `experiment_operators/A5_EXECUTION_GUIDE.md` for an existing A5
container.
