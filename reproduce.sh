#!/bin/sh
# Regenerate the tables and figures from results/.
# A failing step is reported as a failure and the script exits non-zero.
#
# Some per-record outputs are withheld under a data-use agreement (see README,
# "Data not included"); a step consuming one cannot regenerate its figure, which is
# shipped as built. Any other failure is a real failure.
export PYTHONPATH=$PWD/src:$PWD/experiments:$PWD/figures:$PYTHONPATH
command -v python >/dev/null 2>&1 || { echo "ERROR: no 'python' on PATH. Create and activate the virtualenv first (see README), or put python3 on PATH as python."; exit 127; }
fail=0
t=$(mktemp); trap 'rm -f "$t"' EXIT
echo "-- inflating compressed result files"; for f in $(find results -name "*.csv.gz"); do gunzip -kf "$f"; done
echo "-- distribution-law figure"; if ( python experiments/fig_distribution_law.py ) >"$t" 2>&1; then cat "$t"; else cat "$t"; if grep -qE "FileNotFoundError|No such file|No objects to concatenate" "$t"; then echo "   not regenerated: distribution-law figure -- an input is not present. Per-record outputs are withheld under a data-use agreement (README, Data not included); the shipped figure/table stands."; else echo "   FAILED: distribution-law figure"; fail=1; fi; fi
echo "-- gradient figure"; if ( python experiments/fig_gradient.py ) >"$t" 2>&1; then cat "$t"; else cat "$t"; if grep -qE "FileNotFoundError|No such file|No objects to concatenate" "$t"; then echo "   not regenerated: gradient figure -- an input is not present. Per-record outputs are withheld under a data-use agreement (README, Data not included); the shipped figure/table stands."; else echo "   FAILED: gradient figure"; fail=1; fi; fi
echo "-- policy figure"; if ( python experiments/fig_policy.py ) >"$t" 2>&1; then cat "$t"; else cat "$t"; if grep -qE "FileNotFoundError|No such file|No objects to concatenate" "$t"; then echo "   not regenerated: policy figure -- an input is not present. Per-record outputs are withheld under a data-use agreement (README, Data not included); the shipped figure/table stands."; else echo "   FAILED: policy figure"; fail=1; fi; fi
[ "$fail" -eq 0 ] || { echo; echo "reproduce.sh: one or more steps FAILED (see above)."; exit 1; }
echo; echo "reproduce.sh: no step failed."
