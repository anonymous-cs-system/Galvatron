# Vision Transformer

This directory contains scripts to search and train ViT model using Galvatron.

## Usage

### Data preparation
We use standard ImageNet 1K dataset, which you can download from http://image-net.org/. Please use zipped ImageNet to speedup data reading. The ImageNet folder should be placed in this directory and look like this:

```
- ImageNet
    - train_map.txt
    - train.zip
    - val_map.txt
    - val.zip
```

### Training on Single GPU
To train ViT model on a single GPU, run the following scripts:
``` shell
sh scripts/train_vit_large.sh
sh scripts/train_vit_huge.sh
```
### Training on Multiple GPUs
To train ViT model using multiple GPUs in hybrid parallel fashion, run the following scripts:
``` shell
sh scripts/train_vit_large_hp_layerwise.sh
sh scripts/train_vit_huge_hp_layerwise.sh
```
In this mode, user can train the model using any combination of data parallel (DP), tensor parallel (TP), pipeline parallel (PP) and sharded data parallel (SDP). You can apply these strategies either globally or layerwisely.

#### Global Hybrid Parallel Training
To train the model using any global hybrid parallel strategy, simply modify the following arguments in the scripts: first set ```apply_strategy``` to 0, and then modify ```nproc_per_node, pp_deg, global_tp_deg, global_tp_consec, fsdp```. 

```pp_deg``` refers to PP degree, ```global_tp_deg``` refers to TP degree, ```global_tp_consec``` refers to whether the TP communication group is consecutive (eg., [0,1,2,3] is consecutive while [0,2,4,6] is not). If TP is divided before DP and SDP on the decision tree, ```global_tp_consec=1```, else ```global_tp_consec=0```. ```fsdp``` refers to whether to use SDP instead of DP, and ```nproc_per_node``` equals to tp_deg*dp_deg. 

Here are several examples, and the strategies are given following the top-to-bottom order on the decision tree: 

- Strategy 1: 2-way PP, 2-way DP, 2-way TP

    ```nproc_per_node=4, pp_deg=2, global_tp_deg=2, global_tp_consec=0, fsdp=0```

- Strategy 2: 2-way PP, 2-way TP, 2-way SDP

    ```nproc_per_node=4, pp_deg=2, global_tp_deg=2, global_tp_consec=1, fsdp=1```

- Strategy 3: 1-way PP, 2-way TP, 4-way SDP

    ```nproc_per_node=8, pp_deg=1, global_tp_deg=2, global_tp_consec=1, fsdp=1```

- Strategy 4: 4-way PP, 2-way SDP, 1-way TP

    ```nproc_per_node=2, pp_deg=4, global_tp_deg=1, global_tp_consec=0, fsdp=1```

#### Layerwise Hybrid Parallel Training
To train the model using any layerwise hybrid parallel strategy, please set ```apply_strategy``` to 1, set ```nproc_per_node``` to ```gpu_num/pp_deg```, and specify the hybrid parallel strategies for each layer in function ```apply_layerwise_hybrid_strategy()``` in ```train_hp_layerwise.py```. In this mode, PP deg is specified globally and other parallelism degrees can be specified layerwisely. 

A sample layerwise hybrid parallel strategy for ViT-Huge-32 is given in the code: 

For layer 0-19, apply strategy 1-way PP, 1-way TP, 8-way DP;
for layer 20-31, apply strategy 1-way PP, 1-way TP, 8-way SDP.

Set ```nproc_per_node=8``` and apply following codes: 

``` python
### A sample strategy
pp_deg = 1
tp_sizes_enc = [1]*32
tp_consecutive_flags = [1]*32
dp_types_enc = [0]*20+[1]*12
```

### Searching using Galvatron
To search the optimal layerwise hybrid parallel strategy on the given device, please first enter ```../test_enc``` directory, run the environment tests and obtain communication coefficient, overlap coefficient and forward computation time on the given device, and modify the corresponding values in ```search_layerwise_hp.py```. Then run the following scripts to search for the optimal strategy:

``` shell
sh scripts/search_layerwise_hp.sh
```
Given the memory budget, Galvatron provides the layerwise hybrid parallel strategy with maximum throughput. User can train the model with the provided optimal strategy using ```train_hp_layerwise.py``` to obtain the optimal throughput.