# IMPL TensorFlow Utilities

This repository contains the Python scripts and libraries needed to train, evaluate, and deploy Deep Convolutional Neural Network architectures using the TensorFlow library. These tools aim to make it easier to design, implement, and train models quickly by making easier to work with data sets. This is achieved by abstracting and encapsulating a data set into a Python class object, allowing machine learning engineers to focus on model development instead of data wrangling.

## Installation

These tools require Python 3.6 along with the Pip package manager.

```bash
git clone --recursive http://link.tcnj.edu/TensorFlow-Tools/tensorflow_utilities.git

cd tensorflow_utilities

./install
```

## General Usage

In order to use the scripts in this repo youll have to activate a Python Virtual Environment that was created during Installation. Activate the virtual env by.

```bash
source activate
```

### Data Set Creation

To view the arguments required run
```bash
python scripts/create_data_set.py --h
```

Usage Example:
```bash
python scripts/create_data_set.py \
	--data_set_type classification \
	--data_set_name GrassWeeds \
	--input_image_dir /home/data/images/GrassWeeds
```

### Model Training

Now that we have created a dataset we can train a model on the dataset. There are a few pre-designed models that we can choose from or design our own. OPTIONAL: See the IMPL_Models section below to see how the package works.

To view the arguments required to run
```bash
python scripts/train_model.py --h
```

Usage Example:
```bash
python scripts/train_model.py \
	--dataset GrassWeeds \
	--model Discriminator \
	--epochs 10 \
	--output_dir grass_weeds_train_test
```

### Model Evaluation

### Model Inference

## TensorFlow Utilities' APIs

### DataSets

### IMPLModels

IMPL_Models is a repository for holding the IMPLs deep learning models. This package can be added as a subtree to any project and facilitates easy Python `import`s as a Python package. To import a model into your program from the package see below. See the script `tensorflow_utilities/scripts/train_model.py` to see how we can import a model from the IMPL_Models package dynamically (based on user input).

```python3
from IMPL_Models import MobileNetModel

model = MobileNetModel
```

The IMPL_Models package utilizes a Git `subtree`. Below are commands to perform common operations on the `subtree` such as pulling and pushing to the remote

Useful Link: https://andrey.nering.com.br/2016/git-submodules-vs-subtrees/

Updating your local repo (pull)

```bash
git subtree pull --prefix=<path/to/submodule/with/trailing/slash/> <remote_tag> <branch>
git subtree pull --prefix=scripts/IMPL_Models/ impl-models master
```

Updating the remote repo (push)

```bash
# After committing files
git subtree push --prefix=<path/to/submodule/with/trailing/slash/> <remote_tag> <branch>
git subtree push --prefix=scripts/IMPL_Models/ impl-models master
```