#!/usr/bin/env bash

# Source the CANN environment inside an experiment container. Both layouts are
# used by supported images: A5 images commonly expose /usr/local/Ascend/cann,
# while A3 images commonly expose /usr/local/Ascend/ascend-toolkit.
if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
  _cann_set_env=/usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  _cann_set_env=/usr/local/Ascend/ascend-toolkit/set_env.sh
else
  printf '%s\n' \
    'CANN set_env.sh was not found under /usr/local/Ascend/cann or ascend-toolkit.' >&2
  return 1 2>/dev/null || exit 1
fi

_cann_restore_nounset=0
case $- in
  *u*) _cann_restore_nounset=1; set +u ;;
esac
# shellcheck source=/dev/null
source "$_cann_set_env"
if (( _cann_restore_nounset )); then
  set -u
fi
unset _cann_set_env _cann_restore_nounset
