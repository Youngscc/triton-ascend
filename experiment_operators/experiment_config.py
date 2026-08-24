"""Editable settings for the three-axis NPU experiment.

Change this file before starting a new full sweep.  The command line only
selects an operator or one existing case to rerun.
"""

# A3 always disables DynamicCV and varies the native static CV depth.
A3_DEPTH_VALUES = (1, 2, 3, 4)

# A5 uses "off" for a real DynamicCV-disabled baseline. Numeric values enable
# DynamicCV and select buf_slot_num_of_veccore.
A5_BUF_SLOT_NUM_OF_VECCORE_VALUES = ("off", 1, 2, 3, 4)

# "off" passes multibuffer=False and omits set-local-multibuffer. Numeric
# values enable ordinary local multibuffering with that explicit count.
MULTIBUFFER_NUM_VALUES = ("off", 1, 2, 3, 4)

# Level 2 remains excluded because it currently produces invalid HIVM IR in
# part of the operator corpus. Add 2 here when that compiler issue is fixed.
VF_MERGE_LEVEL_VALUES = (0, 1)

# Every successful configuration uses the same benchmark policy.
WARMUP = 5
ACTIVE = 30

# A timed-out case is deferred until the initial sweep finishes, then retried.
CASE_TIMEOUT_SECONDS = 120
TIMEOUT_RETRIES = 1
