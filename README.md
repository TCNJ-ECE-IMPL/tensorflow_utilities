# IMPL TensorFlow Utilities

This repository contains the Python scripts and libraries needed to train, evaluate, and deploy Deep Convolutional Neural Network architectures using the TensorFlow library. These tools aim to make it easier to design, implement, and train models quickly by making easier to work with data sets. This is achieved by abstracting and encapsulating a data set into a Python class object, allowing machine learning engineers to focus on model development instead of data wrangling.

## General Installation and Set Up

These tools require Python 3.6 along with the Pip package manager. Installation instructions can be found [here](doc/INSTALLATION.md). 

## General Usage

### Working with IMPL_Models

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

### Data Set Creation

### Training

### Evaluation

### Inference

## APIs

### DataSets

### IMPLModels
