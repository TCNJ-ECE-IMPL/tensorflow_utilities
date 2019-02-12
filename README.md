# IMPL TensorFlow Utilities

This repository contains the Python scripts and libraries needed to train, evaluate, and deploy Deep Convolutional Neural Network architectures using the TensorFlow library. These tools aim to make it easier to design, implement, and train models quickly by making easier to work with data sets. This is achieved by abstracting the concept of a data set to allow machine learning engineers to focus on model development instead of data wrangling.

## General Installation and Set Up

These tools require Python 3.6 along with the Pip package manager. Python package requirements can be installed using the following commands. See here for first time set up.

### Create Python Virtual Environment and Project Set Up
The following commands,
  1. Clone the TensorFlow Utilities repository
  2. Create a virtual environment
  3. Activate the virtual environment
  4. Install the Python package pre-requisites into your virtual environment
  5. Deactivate the virtual environment

```bash
$ git clone http://link.tcnj.edu/TensorFlow-Tools/tensorflow_utilities.git

$ cd tensorflow_utilities

$ python3 -m venv tf-util-env

$ source ./tf-util-env/bin/activate

$ pip install -r requirements.txt

$ deactivate
```

### Test for Proper Set Up

```bash
$ source tf-util-env/bin/activate

$ python scripts/test.py --all

$ deactivate
```

## Usage

### DataSet

#### Creation

#### DataSet Class Methods

## First Time Installation and Set Up

Use these instructions to set up this repository for the first time on a system. These steps are necessary to configure the system variables and dependancies needed.

### Directory Set Up
1. Clone the TensorFlow Models Repository
(Make sure you choose a permanent location, it will be annoying to change this later)

```bash
$ cd /some/path/tensorflow/

$ git clone https://github.com/tensorflow/models.git
```

2. Create a directory to contain any future data sets. This directory should be able to allocate a large amount of storage.

```bash
$ mdkdir /some/path/DataSets
```

### Set Environment Variables

To set these environment variable as a SYSTEM environment variable modify the file `/etc/profile` (or `~/.bash_profile` for USER environment variable) to include the following

```bash
$ export PYTHONPATH=$PYTHONPATH:\
  /some/path/tensorflow/models/:\
  /some/path/tensorflow/models/research/:\
  /some/path/tensorflow/models/research/slim:\
  /some/path/tensorflow/models/research/gan
$ export DCNN_DATASETS_PATH=/some/path/tensorflow/datasets/
```

The paths specified should match the `some/path` placeholders set in step 1 and 2 above. After completion restart your shell for changes to take effect.
