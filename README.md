Please refer to our [Project Page](https://kiyoung98.github.io/tps-dps/)

# TPS-DPS: Transition Path Sampling with Improved Off-Policy Training of Diffusion Path Samplers

This repository contains the code and instructions to reproduce the results of the paper.

## Installation
```
bash install.sh
```

## Reproduce

- **Sampling**: Run the following command to sample paths
    ```
    python synthetic/sample.py --model_path models/synthetic/pot.pt
    python src/sample.py --molecule alanine --model_path models/alanine/force.pt --save_dir paths/alanine/force --bias force
    python src/sample.py --molecule poly --start_state pp2 --end_state pp1 --num_steps 10000 --sigma 0.2 --bias scale --save_dir paths/poly/scale --model_path models/poly/scale.pt
    python src/sample.py --molecule chignolin --start_state unfolded --end_state folded --num_steps 5000 --sigma 0.5 --bias scale --save_dir paths/chignolin/scale --model_path models/chignolin/scale.pt
    python src/sample.py --molecule trpcage --start_state unfolded --end_state folded --num_steps 5000 --sigma 0.5 --bias scale --save_dir paths/trpcage/scale --model_path models/trpcage/scale.pt --temperature 400
    python src/sample.py --molecule bba --start_state unfolded --end_state folded --num_steps 5000 --sigma 0.5 --bias scale --save_dir paths/bba/scale --model_path models/bba/scale.pt --temperature 400
    python src/sample.py --molecule bbl --start_state unfolded --end_state folded --num_steps 5000 --sigma 0.5 --bias scale --save_dir paths/bbl/scale --model_path models/bbl/scale.pt --temperature 400

    ```

- **Evaluation**: Run the following command to evaluate sampled paths
    ```
    python synthetic/eval.py 
    python src/eval.py --molecule alanine
    python src/eval.py --molecule poly --start_state pp2 --end_state pp1 --save_dir paths/poly/scale
    python src/eval.py --molecule chignolin --start_state unfolded --end_state folded --save_dir paths/chignolin/scale 
    python src/eval.py --molecule trpcage --start_state unfolded --end_state folded --save_dir paths/trpcage/scale --temperature 400
    python src/eval.py --molecule bba --start_state unfolded --end_state folded --save_dir paths/bba/scale --temperature 400
    python src/eval.py --molecule bbl --start_state unfolded --end_state folded --save_dir paths/bbl/scale --temperature 400
    ```

- **Training**: Run the following command to start training. For better results, try many seeds
    ```
    python synthetic/train.py --seed 2
    python src/train.py --molecule alanine --seed 2
    ```