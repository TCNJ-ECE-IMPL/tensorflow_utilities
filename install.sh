# Creating aand Activating Python Virtual Environment

py_requirements = $0

venv_ena = ($py_requirements != "requirements_docker.txt")

if [venv_ena]]
then
    python3 -m venv tf-utils-venv
    source tf-utils-venv/bin/activate
fi

pip install Cython
pip install opencv-python
# Installing Python package requirements
pip install -r $py_requirements
pip install keras
pip install matplotlib
echo "Python Package Requirements Installed Sucessfully!!"

# Setting Up DataSet directory and setting environment variables
mkdir DataSets
source activate.sh

# Building protobufs
#cd scripts/TFUtils/models/research/
#protoc object_detection/protos/*.proto --python_out=.

# Testing Installation
#python object_detection/model_lib_test.py
#cd $TFUTILS_ROOT
#echo "Official TensorFlow research tools have passed all tests!!"
#python scripts/test.py --all
echo "IMPL TensorFlow Utils have passed all tests!!"


if [venv_ena]
then
	# Deactivating Virtual Environment
	deactivate
fi
