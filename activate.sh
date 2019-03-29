#!/bin/bash
# Shell script to activate the python virtual environment and set path variables
export PYTHONPATH=$(pwd)/scripts/TFUtils/models/:$(pwd)/scripts/TFUtils/models/research/:$(pwd)/scripts/TFUtils/models/research/slim/:$(pwd)/scripts/TFUtils/models/research/gan:

VENV_ACTIVATE=tf-utils-venv/bin/activate

if [ -f "$VENV_ACTIVATE" ]; then
    echo "Python Virtual Environment Activated ..."
    source $VENV_ACTIVATE
fi

source ".env"
