#!/bin/sh
python NN_training.py --mode dm --params '{ "n_gnn_layers": 10, "n_dim_gnn": 2, "n_output_gnn": 50, "n_output_gnn_last": 50, "n_dense_layers": 4, "n_dense_nodes": 200, "wiring_mode": "m3" }' /data/results/run102
#python NN_training.py --mode dm --params '{ "n_gnn_layers": 10, "n_dim_gnn": 2, "n_output_gnn": 50, "n_output_gnn_last": 50, "n_dense_layers": 4, "n_dense_nodes": 500, "wiring_mode": "m3" }' /data/results/run104
