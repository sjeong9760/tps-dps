# TPS-DPS: Transition Path Sampling with Improved Off-Policy Training of Diffusion Samplers

This repository contains the code and instructions to reproduce the results of the paper.

![Alanine](figures/alanine.gif)
![Chignolin](figure/chignolin.gif)
![Poly](figure/poly.gif)

## Installation

1. First, create a new Conda environment:
    ```
    conda create -n tps-dps python=3.9
    ```

2. Activate the newly created environment:
    ```
    conda activate tps-dps
    ```

4. Install the openmmtools for Molecular Dynamics (MD) simulation using the following commands:
    ```
    conda install -c conda-forge openmmtools
    ```

5. Install the openmmforcefields for forcefields of Polyproline Helix and Chignolin using the following commands:
    ```
    git clone https://github.com/openmm/openmmforcefields.git
    ```
6. Install another packages using the following commands:
    ```
    pip install torch tqdm wandb mdtraj matplotlib joblib pyemma
    ```

## Reproduce

- **Sampling**: Run the following command to sample paths
    ```
    python synthetic/sample.py --model_path models/synthetic/pot.pt
    python src/sample.py --molecule alanine --model_path models/alanine/force.pt --save_dir paths/alanine/force --bias force
    python src/sample.py --molecule poly --start_state pp2 --end_state pp1 --num_steps 10000 --sigma 0.2 --bias scale --save_dir paths/poly/scale --model_path models/poly/scale.pt
    python src/sample.py --molecule chignolin --start_state unfolded --end_state folded --num_steps 5000 --sigma 0.5 --bias scale --save_dir paths/chignolin/scale --model_path models/chignolin/scale.pt
    ```

- **Evaluation**: Run the following command to evaluate sampled paths
    ```
    python synthetic/eval.py 
    python src/eval.py --molecule alanine
    python src/eval.py --molecule poly --start_state pp2 --end_state pp1 --save_dir paths/poly/scale
    python src/eval.py --molecule chignolin --start_state unfolded --end_state folded --save_dir paths/chignolin/scale 
    ```

- **Training**: Run the following command to start training:
    ```
    python synthetic/train.py 
    python src/train.py --molecule alanine
    ```