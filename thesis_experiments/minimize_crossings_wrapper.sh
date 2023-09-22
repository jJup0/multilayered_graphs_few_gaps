#!/bin/sh

echo "I am running"
sync

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=$HOME/multilayered_graphs_few_gaps
fi





# GUROBI
export GUROBI_HOME="/home1/share/gurobi/gurobi/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export GRB_LICENSE_FILE="/home1/share/gurobi/gurobi.lic"
if [ -n "$LD_LIBRARY_PATH" ]; then
  export LD_LIBRARY_PATH="${GUROBI_HOME}/lib"
else
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
fi

echo "python path: " $PYTHONPATH " PATH " $PATH " LD_LIBRARY_PATH " $LD_LIBRARY_PATH
sync

/home1/e52009269/.pyenv/versions/3.10.9/bin/python -m thesis_experiments.minimize_crossings "$@"