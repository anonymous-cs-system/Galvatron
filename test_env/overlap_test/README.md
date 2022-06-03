# overlap_test
This directory contains scripts to test mlp model on single gpu and on multi gpus with DDP.

The following hyper-parameters for the cost model will be obtained throught this test: 

- `bct_fct_coe: coefficient representing backward_computing_time / forward_computing_time`
- `dp_overlap_coe: data parallel communication delay coefficient when overlapped with backward computing`
- `bct_overlap_coe: backward computing delay coefficient when overlapped with data parallel communication`

Run `sh profile_mlp.sh` to generate nvvp files, download nvvp files and open in NVIDIA Visual Profiler to find out the following time ($\mu s$):
- `forward_computing_time` $\approx$ `127`
- `backward_computing_time_without_overlap` $\approx$ `140 + 127 = 267`
- `dp_comm_time_without_overlap` $\approx$ `23`
- `dp_comm_time_with_overlap` $\approx$ `30`
- `backward_computing_time_with_overlap` $\approx$ `190 + 150 = 340`

Using the time above, we can calculate hyper-parameters approximately:
- `bct_fct_coe` $\approx$ `2`
- `dp_overlap_coe` $\approx$ `1.3`
- `bct_overlap_coe` $\approx$ `1.3`