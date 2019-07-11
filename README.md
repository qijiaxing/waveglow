# Waveglow
This is an implementation of the Waveglow model based on Tensorflow. The model architecture follows the NVIDIA public Waveglow model. It includes the inference part only for now. The main purpose is to benchmark its inference performance on GPU. You may use it freely.

File an issue if you have any questions.

# Content

This repo has the following files:
* `glow.py`, `wn.py` and `upsample.py`: waveglow model
* `config.py`: model parameters
* `benchmark.py`: benchmark utilities

# Usage

To run waveglow inference:
```
python ./benchmark.py --gpu=0
```

# Inference performance

On a NVidia V100 32GB NVLink GPU, we measure the inference RTF to be 0.024 in `fp16` mode with `n_channels = 512`.
The docker image we used is `nvcr.io/nvidia/tensorflow:19.06-py3`
