# Conceptual Learning and Causal Reasoning for Semantic Communication

(Description of the project here)

The project experimental results can be recreated using the code in this repository. Steps for carrying out the experiments are detailed below. Note that all experiments were performed on a Linux machine running Ubuntu 22.04.

## Environment Setup

> Note: if planning to train the model on a GPU with CUDA, be sure all required software is installed before setting up the environment. See the documented software requirements [here](https://www.tensorflow.org/install/pip#software_requirements).

An Anaconda environment was created to carry out the experiments. This environment can be created using the file `env/setup.yaml`. First, be sure to [install Anaconda](https://docs.anaconda.com/anaconda/install/). Then open a terminal and navigate to the `env/` subdirectory in this repository and run the command:

```bash
conda env create -f setup.yml
```

Once the environment has been created, to activate it run the command:

```bash
conda activate sccsr-env
```

> Note: newer versions of Tensorflow/Keras (versions 2.16.1/3.3.2, respectfully, at the time of writing) seem to have trouble finding the CUDA libraries for training on a GPU. One workaround is to use the method described in [this comment](https://github.com/tensorflow/tensorflow/issues/63362#issuecomment-1988630226) on GitHub.

## Steps

The project experiments consist of six primary steps:

1. Prepare the data
2. Learn the domains using the CSLearn package
3. Carry out the causal learning workflow
4. Train the semantic decoder
5. Prepare baseline models
6. Perform wireless simulations

### 1. Prepare the data

This project uses the German traffic sign dataset [1]. The original dataset can be downloaded [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html). For this project, the following files were downloaded:

- GTSRB_Final_Training_Images.zip (Training data)
- GTSRB_Final_Test_Images.zip (Test data)
- GTSRB_Final_Test_GT.zip (Test data labels)

To get the data ready for use by the preparation script, first download the files and place them in the `local/` subdirectory. Then run the following commands:

```bash
chmod u+x get_raw_data.sh
./get_raw_data.sh
```

This bash script downloads the data, extracts the raw data from the `.zip` files and places them into the correct subdirectories.

### Learn the domains

(domain learning here)

### Carry out causal learning

(causal learning here)

### Train the semantic decoder

(decoder training here)

### Prepare the baseline models

(baseline models here)

#### Basic classifier models

(basic classifier training here)

#### End-to-end semantic models

(end-to-end semantic model training here)

### Perform wireless simulations

(wireless simulation here)
