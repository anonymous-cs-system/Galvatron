nvprof --profile-child-processes -o nvvps/mlp_test_results_%p.nvvp sh test_mlp.sh
nvprof --profile-child-processes -o nvvps/mlp_test_ddp_results_%p.nvvp sh test_mlp_dp.sh