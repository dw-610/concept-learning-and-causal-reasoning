#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda create -n sccsr-env-test python=3.10.12 -y
conda activate sccsr-env-test

conda install -c conda-forge -c anaconda -c defaults numpy=1.26.4 -y
conda install -c conda-forge -c anaconda -c defaults matplotlib=3.8.4 -y
conda install -c conda-forge -c anaconda -c defaults pandas=2.2.2 -y
conda install -c conda-forge -c anaconda -c defaults pip=24.0 -y

pip install tensorflow[and-cuda]==2.16.1
pip install opencv-python==4.10.0.84
pip install tqdm==4.66.4
pip install sentence-transformers==3.0.1
pip install lingam==1.8.3
pip install dowhy==0.11.1
pip install econml==0.15.0
