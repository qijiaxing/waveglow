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

# Benchmark
