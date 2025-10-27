#!/bin/bash

python -m simulation.examples.generate_datasets --config=simulation/configs/simulations.yaml --config=simulation/configs/experiments.yaml

python -m simulation.examples.generate_noisy_datasets --config=simulation/configs/experiments.yaml

python -m analysis.examples.run_exp --config=analysis/configs/analysis.yaml --config=analysis/configs/experiment.yaml

python -m analysis.examples.run_noise_reg --config=analysis/configs/analysis.yaml --config=analysis/configs/experiment.yaml