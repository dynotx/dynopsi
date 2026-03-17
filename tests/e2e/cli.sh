#!/bin/bash

dynopsi check $HOME/dynopsi-dev/example/configurations/vegf_1bj1_multichain_target.yaml

dynopsi sample $HOME/dynopsi-dev/example/configurations/vegf_1bj1_multichain_target.yaml --num_samples 2

dynopsi sample $HOME/dynopsi-dev/tests/test_files/test_simulation.yaml --num_samples 2 --batch_size 2