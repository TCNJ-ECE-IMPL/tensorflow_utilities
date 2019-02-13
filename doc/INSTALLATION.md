# TensorFlow Utilities Installation

These tools require Python 3.6 along with the Pip package manager. Python package requirements can be installed using the following commands. Follow [these instructions](#first-time-installation-and-set-up-procedure) for first time machine set up.

## General Installation Procedure

Follow the following instructions to install user project pre-requisites and to test set up. Only follow these instructions if you are working on an IMPL Linux workstation or if you've completed [these](#first-time-installation-and-set-up) first

### Create Python Virtual Environment and Set Up Project
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

### Run Tests for Proper Set Up

```bash
$ source tf-util-env/bin/activate

$ python scripts/test.py --all

$ deactivate
```

## First Time Installation and Set Up Procedure

Use these instructions to set up this repository for the first time on a system. These steps are necessary to configure the system variables and dependancies needed. If you are working on an IMPL Linux workstation than skip the rest of these steps.

#### Directory Set Up
1. Clone the TensorFlow Models Repository
(Make sure you choose a permanent location, it will be annoying to change this later)

```bash
$ cd /some/path/tensorflow/

$ git clone https://github.com/TCNJ-ECE-IMPL/models.git
```

2. Create a directory to contain any future data sets. This directory should be able to allocate a large amount of storage.

```bash
$ mdkdir /some/path/datasets
```

#### Set Environment Variables

To set these environment variable as a SYSTEM environment variable modify the file `/etc/profile` (or `~/.bash_profile` for USER environment variable) to include the following

```bash
$ export PYTHONPATH=$PYTHONPATH:\
  /some/path/tensorflow/models/:\
  /some/path/tensorflow/models/research/:\
  /some/path/tensorflow/models/research/slim:\
  /some/path/tensorflow/models/research/gan
$ export DCNN_DATASETS_PATH=/some/path/tensorflow/datasets/
```

The paths specified should match the `some/path` placeholders set in step 1 and 2 above. After completion restart your shell for changes to take effect. Once these steps are complete [these steps](#general-installation-and-set-up) can be followed to install TensorFlow Utilities in your home directory.
