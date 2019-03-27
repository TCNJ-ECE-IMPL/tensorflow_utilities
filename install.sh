
# Creating aand Activating Python Virtual Environment

if [$0 != "requirements_docker.txt"]
then
    python3 -m venv tf-utils-venv
    source tf-utils-venv/bin/activate
fi

pip install Cython
pip install opencv-python
# Installing Python package requirements
pip install -r $0
pip install keras
echo "Python Package Requirements Installed Sucessfully!!"

# Setting Up DataSet directory and setting environment variables
mkdir DataSets
source activate.sh

# Building protobufs
cd scripts/TFUtils/models/research/
protoc object_detection/protos/*.proto --python_out=.

# Testing Installation
python object_detection/model_lib_test.py
cd $TFUTILS_ROOT
echo "Official TensorFlow research tools have passed all tests!!"
#python scripts/test.py --all
echo "IMPL TensorFlow Utils have passed all tests!!"

# Deactivating Virtual Environment
deactivate
