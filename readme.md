# Implementation
This is the official implementation of the paper "Effective In-Context Example Selection through Data Compression" based on PyTorch.

## Reproduction
Check the following instructions for reproducing experiments.

### Quick Start
#### 1. Get the BM25 results.

#### 2. Get our results.

#### Step 1：
You can get the results of BM25.

```bash
python stage1_bm25.py
```

#### Step 2：

Note: 
You need to interrupt the forward propagation of GPT2 at the first layer firstly by changing the modeling_gpt2.py, which in the pre-downloaded file.

You can get the examples of different datasets and models by the following instructions.

For example:

```bash
python stage2_ours.py --data_type cola --model gpt2
python stage2_ours.py --data_type sick --model gpt2-medium
```

### Testing
We referred to https://github.com/juny116/ICL-DeepSpeed testing process for testing.

You can modify the config file to test different tasks.
For example, you can test with the following command:

```bash
cd ICL-DeepSpeed-main
python single_main_ours.py
```

### Environments
We conducted the experiments based on the following environments:
* CUDA Version: 11.4
* torch version: 1.10.0
* OS: Ubuntu 18.04.5 LTS
* GPU: NVIDIA Geforce RTX 3090
* CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
