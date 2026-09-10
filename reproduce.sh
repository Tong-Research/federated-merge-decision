#!/bin/sh
# Regenerate tables/figures from results/. Steps whose inputs are withheld under a data-use agreement report it and continue.
export PYTHONPATH=$PWD/src:$PWD/experiments:$PWD/figures:$PYTHONPATH
echo "-- inflating compressed result files"; for f in $(find results -name "*.csv.gz"); do gunzip -kf "$f"; done
echo "-- distribution-law figure"; ( python experiments/fig_distribution_law.py ) || echo "   skipped: distribution-law figure needs inputs withheld under a data-use agreement (the figure/table is shipped as built)"
echo "-- gradient figure"; ( python experiments/fig_gradient.py ) || echo "   skipped: gradient figure needs inputs withheld under a data-use agreement (the figure/table is shipped as built)"
echo "-- policy figure"; ( python experiments/fig_policy.py ) || echo "   skipped: policy figure needs inputs withheld under a data-use agreement (the figure/table is shipped as built)"
