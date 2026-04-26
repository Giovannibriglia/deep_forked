source .venv/bin/activate
python3 scripts/gnn_exp/create_all_training_data.py exp/rl_exp/batch0_merged --deep_exe cmake-build-release-nn/bin/deep --dataset-max-creation 5000
python3 scripts/gnn_exp/create_all_training_data.py exp/rl_exp/batch0_merged --deep_exe cmake-build-release-nn/bin/deep --dataset-name "test_data" --dataset-max-creation 1000 --training-folder "Test"
python3 scripts/rl_exp/train_models.py exp/rl_exp/batch0_merged --onnx-frontier-size 8 16 32 64

# python3 scripts/gnn_exp/create_all_training_data.py exp/rl_exp/batch0_merged --deep_exe cmake-build-release-nn/bin/deep --dataset-name "test_data" --dataset-max-creation 25000 --training-folder "Test"