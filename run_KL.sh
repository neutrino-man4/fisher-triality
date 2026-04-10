#!/bin/bash

python3 create_features.py --num-jets 100000 --num-particles 75
python3 generate_MC_KL_results.py