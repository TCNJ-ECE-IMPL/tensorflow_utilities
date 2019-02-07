# IMPL TensorFlow Utilities

This repository contains the Python scripts and libraries needed to train, evaluate, and deploy Deep Convolutional Neural Network architectures using the TensorFlow library. These tools aim to make it easier to design, implement, and train models quickly by making easier to work with data sets. This is achieved by abstracting the concept of a data set to allow machine learning engineers to focus on model development instead of data wrangling.

## Installation and Set Up

These tools require Python 3.6 along with the Pip package manager. Python package requirements can be installed using the following commands. See here for first time set up.

```bash
$ python3 -m venv tf-util-env

$ source ./tf-util-env/bin/activate

$ pip install -r requirements.txt

$ deactivate
```

If installation issues occur see below

## Usage

### DataSet

#### Creation

#### DataSet Class Methods

## First Time Installation and Set Up

Use these instructions to set up the necessary system variables and dependancies necessary for these tools.

### TensorFlow Models Repository Set Up
1. Clone the TensorFlow Models Repository
(Make sure you choose a permanent location, it will be annoying to change this later)
```bash
$ cd /some/path/tensorflow/

$ git clone ...
```

2. Set Environment Variables

To set these environment variable as a SYSTEM environment variable modify the file '/etc/profile' to include the following
```bash
$ export PYTHON_PATH=$PYTHON_PATH:/some/path/tensorflow/models/
```

### Data Set Directory Set Up
1. Create a directory to contain any future data sets. This directory should be able to allocate a large amount of storage.

2. Set Environment Variables
To set these environment variable as a SYSTEM environment variable modify the file '/etc/profile' to include the following
```bash
$ export DCNN_DATASETS_PATH=/some/path/tensorflow/datasets/
```
