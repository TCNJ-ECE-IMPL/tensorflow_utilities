# IMPL TensorFlow Utilities

This repository contains the Python scripts and libraries needed to train, evaluate, and deploy Deep Convolutional Neural Network architectures using the TensorFlow library. These tools aim to make it easier to design, implement, and train models quickly by making easier to work with data sets. This is achieved by abstracting and encapsulating a data set into a Python class object, allowing machine learning engineers to focus on model development instead of data wrangling.

## General Installation and Set Up

These tools require Python 3.6 along with the Pip package manager. Python package requirements can be installed using the following commands. Follow [these instructions](#first-time-installation-and-set-up) for first time machine set up.

### Create Python Virtual Environment and Set Up Project
The following commands,
  1. Clone the TensorFlow Utilities repository
  2. Create a virtual environment
  3. Activate the virtual environment
  4. Install the Python package pre-requisites into your virtual environment
  5. Deactivate the virtual environment

```bash
cd ~/some/project/path
git clone http://link.tcnj.edu/TensorFlow-Tools/tensorflow_utilities.git
python3 -m venv tf-util-env
source ./tf-util-env/bin/activate
pip install -r requirements.txt
deactivate
```

### Run Tests for Proper Set Up

Use the following commands to run all the tests for the repository.

```bash
source tf-util-env/bin/activate
cd tensorflow_utilities
python scripts/test.py --all
deactivate
```

The output should say all tests have been passed.

## General Usage

### Data Set Creation

### Training

### Evaluation

### Inference

## APIs

### DataSets

### IMPLModels

## First Time Installation and Set Up

Use these instructions to set up this repository for the first time on a system. These steps are necessary to configure the system variables and dependancies needed.

### Software Installation
1. Clone the TensorFlow Models Repository
(Make sure you choose a permanent location, it will be annoying to change this later)

```bash
cd some/path
mkdir tensorflow
cd /some/path/tensorflow/
git clone https://github.com/TCNJ-ECE-IMPL/models.git
```

2. Check to see if Google's Protobuf is installed with

```bash
which protoc
```

If command cannot be found follow these steps, otherwise they can be skipped

```bash
cd /some/path/tensorflow/
wget https://github.com/google/protobuf/releases/download/v3.4.1/protobuf-cpp-3.4.1.tar.gz
tar -zxvf protobuf-cpp-3.4.1.tar.gz
sudo mv protobuf-3.4.1 /usr/local/bin
cd /usr/local/bin/protobuf-3.4.1
./configure
make
make install
protoc --version
```

Should return `>> libprotoc 3.4.0`. Note some errors during `$ make install` are OK.

3. COCO API Installation

COCO APIs are needed for COCO evalutation methods, which are common eval metrics used to compare model performance.

```bash
cd /some/path/tensorflow
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools /some/path/tensorflow/models/research/
```

4. Models Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the `/some/path/tensorflow/models/research/` directory:

```bash
cd /some/path/tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
```

5. Set Up Data Sets Directory

Create a directory to contain any future data sets. This directory should be able to allocate a large amount of storage.

```bash
mdkdir /some/path/tensorflow/DataSets
```

### Set Environment Variables

To set these environment variable as a SYSTEM environment variable modify the file `/etc/profile` (or `~/.bash_profile` for USER environment variable) to include the following

```bash
export PYTHONPATH=$PYTHONPATH:\
  /some/path/tensorflow/models/:\
  /some/path/tensorflow/models/research/:\
  /some/path/tensorflow/models/research/slim:\
  /some/path/tensorflow/models/research/gan
export DCNN_DATASETS_PATH=/some/path/tensorflow/datasets/
```

### Test Software Installation and Set Up

```bash
# From /some/path/tensorflow/models/research/
python object_detection/builders/model_builder_test.py
```

The paths specified should match the `some/path` placeholders set in step 1 and 2 above. After completion restart your shell for changes to take effect. Once these steps are complete [these steps](#general-installation-and-set-up) can be followed to install TensorFlow Utilities in your home directory.
