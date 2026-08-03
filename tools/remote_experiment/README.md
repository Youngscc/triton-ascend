# Local-to-server experiment loop

This uses the SSH alias `huawei-server`, synchronizes the project into
`/home/yuanye/code/triton-ascend`, and runs commands inside the existing
`sgl-sky` container. The container mounts `/home`, so the same project path is
visible inside and outside the container.

From the repository root:

```bash
# 1. Sync source. Generated outputs and .git metadata are excluded.
./tools/remote_experiment/sync.sh

# 2. Start a detached experiment in sgl-sky.
./tools/remote_experiment/run.sh python -u path/to/experiment.py --arg value

# 3. Follow the newest log. Ctrl-C only stops log following, not the job.
./tools/remote_experiment/logs.sh

# Follow a specific run printed by run.sh:
./tools/remote_experiment/logs.sh 20260803-180000-12345
```

The default sync is additive. To mirror local deletions on the server, use
`RSYNC_DELETE=1 ./tools/remote_experiment/sync.sh`; this removes remote files
under the target that are absent locally.

Useful overrides are available without editing files:

```bash
REMOTE_CONTAINER=other-container ./tools/remote_experiment/run.sh bash -lc '...'
LINES=200 ./tools/remote_experiment/logs.sh latest
```

