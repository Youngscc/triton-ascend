# Offline source sync

Use this helper only when the target machine cannot update the checkout with
Git. Copy `config.local.sh.example` to the ignored `config.local.sh`, then set
the local project path, remote SSH alias, remote project path, and
`REMOTE_SOURCE_MODE="rsync"`.

Preview a deletion-enabled transfer first:

```bash
RSYNC_DELETE=1 RSYNC_DRY_RUN=1 ./tools/remote_experiment/sync.sh
```

After checking the itemized output, run the transfer:

```bash
RSYNC_DELETE=1 ./tools/remote_experiment/sync.sh
```

`offline-wheel/` is always excluded from every source transfer, including the
deletion-enabled allowlisted passes. Build products, caches, results, local
configuration, and nested dependency repositories are also protected by the
script's shared exclusion list.
