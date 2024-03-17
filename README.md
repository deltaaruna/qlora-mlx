# qlora-mlx
This is a [QLoRA](https://arxiv.org/abs/2305.14314) implementation with [mlx](https://github.com/ml-explore/mlx) framework

## Contents

* [Setup](#Setup)
* [Save](#Save)
* [Train](#Train)
* [Generate](#Generate)
* [Example](#Example)

## Setup 

Install the dependencies:

```
pip install -r requirements.txt
```
## Save 

Download a model from huggingface and save it as a **4-bit quantized model**:

```
python save.py --hd-path <hf_repo> -q --mlx-path <location>
```

## Train 

```
python run.py --train --model <model_location> --data <data_location> --batch-size <batch_size> --lora-layers <layers>
```
After training is finished, a new file called **adapters.npz** will be created at the working directory. This is the adapter for the newly trained data.

## Generate 

```
python run.py --model <model_location> \
               --adapter-file <adapter_file_location> \
               --max-tokens <> \
               --prompt <>
```

### Example 
Download TinyLlama-1.1B-Chat-v1.0 and save the quintized model in your current directory 

```
python save.py --hf-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 -q --mlx-path ./TinyLlama-1.1B-Chat-v1.0
```

Add or generate data:
This is how to generate some data for a given topic

```
brew install chenhunghan/homebrew-formulae/mlx-training-rs
export OPENAI_API_KEY=[Your openai key]
mlxt --topic="the topic you are interested" -n=100
```
Train
```
python run.py --train --model ./TinyLlama-1.1B-Chat-v1.0 --data ./data --batch-size 1 --lora-layers 4
```
Generate

```
python run.py --model ./TinyLlama-1.1B-Chat-v0.6 \
               --adapter-file ./adapters.npz \
               --max-tokens 50 \
               --prompt "
Q: Your question here
A: "
```



