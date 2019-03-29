# Shell script to activate the python virtual environment and set path variables
export PYTHONPATH=$(pwd)/scripts/TFUtils/models/:$(pwd)/scripts/TFUtils/models/research/:$(pwd)/scripts/TFUtils/models/research/slim/:$(pwd)/scripts/TFUtils/models/research/gan:

source tf-utils-venv/bin/activate

echo "Virtual Environment Activated . . ."
source ".env"
