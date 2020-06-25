# Microtubules Classifiaction

## Description

The antimitotic drug Taxol is an important new drug for the treatment of certain types of cancer. This substance impacts the cytoskeleton of the cell, entangling it. As a result, the cell cannot perform mitosis. 

We are given images of the cytoskeleton of cells under the influence of a different dose of the drug from the laboratory. Different dosages of taxol act differently on the cytoskeleton formed by microtubules, which can be seen in the pictures under a microscope.

In this repo, we develop a toolset that will help to infer the dosage of the drug for each organism.

Paper: <coming soon>

W&B Report: <coming soon>

# Quickstart

Download and unpack the data
```
wget <link>
unzip data.zip
rm data.zip
```

To use Weight & Biases for logging copy your token to `configs/wandb-token.txt`
 
## How to run locally

```
# setup the docker environment
make -f Makefile.local setup

# reproduce experiments passing the config name and device
# for example 
make -f Makefile.local train CONFIG=local_gpu_config.yaml DEVICE=device=1

# reproduce experiment with SVM in the same manner
# if you don't pass the parameter will simple use default
make -f Makefile.local train-svm
```
## How to run on Neu.ro

You can run this project on [Neuro Platform](https://neu.ro).

```
# setup the environment
make setup

# reproduce experiments 
make train CONFIG=<config>

# reproduce experiments with SVM 
make train ENTRYPOINT=train_svm.py
```


## Directory Structure

| Mount Point                                  | Description           | Neuro Storage URI                                                                  |
|:-------------------------------------------- |:--------------------- |:---------------------------------------------------------------------------- |
|`microtubules_classifiaction/data/`           | Data                  | `storage:microtubules_classifiaction/data/`                              |
|`microtubules_classifiaction/modules/`        | Python modules        | `storage:microtubules_classifiaction/modules/` |
|`microtubules_classifiaction/notebooks/`      | Jupyter notebooks     | `storage:microtubules_classifiaction/notebooks/`                         |
|`microtubules_classifiaction/results/`        | Logs and results      | `storage:microtubules_classifiaction/results/`                           |
|`microtubules_classifiaction/configs/`        | Configuration files   | `storage:microtubules_classifiaction/configs/`                           |
