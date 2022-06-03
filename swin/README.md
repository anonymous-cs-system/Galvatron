# Swin Transformer

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