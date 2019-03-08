
# Creating aand Activating Python Virtual Environment
python3 -m venv tf-utils-venv
source tf-utils-venv/bin/activate

# Installing Python package requirements
pip install -r requirements_cpu.txt
echo "\nPython Package Requirements Installed Sucessfully!!"

# Setting Up DataSet directory and setting environment variables
mkdir DataSets
echo "export DCNN_DATASETS_PATH=$(pwd)/DataSets/" >> activate.sh
echo "export TFUTILS_ROOT=$(pwd)" >> activate.sh
source activate.sh

# Building protobufs
cd scripts/TFUtils/models/research/
protoc object_detection/protos/*.proto --python_out=.
cd $TFUTILS_ROOT

# Testing Installation
python object_detection/model_lib_test.py
echo "\nOfficial TensorFlow research tools have passed all tests!!"
python scripts/test.py --all
echo "\nIMPL TensorFlow Utils have passed all tests!!"

# Deactivating Virtual Environment
deactivate
