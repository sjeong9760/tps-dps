#!/bin/bash

# Create a conda environment
conda create -n tps-dps python=3.11 -y
source activate tps-dps

# Install the openmmtools for MD simulation
conda install -c conda-forge openmmtools -y

# Install other packages
pip install tqdm matplotlib joblib pyemma torch==1.13.1