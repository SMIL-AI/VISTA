# Vista-Vision-informed-Safety-and-Transportation-Assessment


<br>
<span>
<b>Authors:</b> 
<a class="name" target="_blank" href="https://winstonyang117.github.io/">Yunxiang Yang<sup>*</sup></a>, 
<a class="name" target="_blank" href="https://zhigao2017.github.io/">Ningning Xu<sup>*</sup></a>, 
<a class="name" target="_blank" href="https://engineering.uga.edu/team_member/jidong-yang/">Jidong Yang<sup>†</sup></a>, 
<br>
<sup>*</sup>Equal Contribution. 
<sup>†</sup>Corresponding Author.
</span>


# 🔥News
- [2025/05/21] We released our paper — discussions and feedback are warmly welcome!
- [2025/07/09] We released our SFT dataset, model, training, and evaluation code. Welcome to download and explore them.

<br>

# Overview

![overview](./assets/teaser.jpg)

<details><summary>Abstract</summary> 
Vision language models (VLMs) have achieved impressive performance across a variety of computer vision tasks. However, the multimodal reasoning capability has not been fully explored in existing models. In this paper, we propose a Chain-of-Focus (CoF) method that allows VLMs to perform adaptive focusing and zooming in on key image regions based on obtained visual cues and the given questions, achieving efficient multimodal reasoning. To enable this CoF capability, we present a two-stage training pipeline, including supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we construct the MM-CoF dataset, comprising 3K samples derived from a visual agent designed to adaptively identify key regions to solve visual tasks with different image resolutions and questions. We use MM-CoF to fine-tune the Qwen2.5-VL model for cold start. In the RL stage, we leverage the outcome accuracies and formats as rewards to update the Qwen2.5-VL model, enabling further refining the search and reasoning strategy of models without human priors. Our model achieves significant improvements on multiple benchmarks. On the V* benchmark that requires strong visual reasoning capability, our model outperforms existing VLMs by 5% among 8 image resolutions ranging from 224 to 4K, demonstrating the effectiveness of the proposed CoF method and facilitating the more efficient deployment of VLMs in practical applications.
</details>

## Visual Search Agent
![visual_search_agent](./assets/train_test_pipeline.png)

## Framework
![framework](./assets/framework.png)

<br>

# Training

## SFT Stage

### Installation

Please follow the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository to install the environment.

### Data Preparation

1. Download the dataset (including images and annotations) from [Hugging Face – Cof SFT Dataset](https://huggingface.co/datasets/xintongzhang/CoF-SFT-Data-5.4k)

2. Modify the configuration file `configs/sft_lora-7b.yaml` to match your data paths and training settings.
3. Copy `configs/dataset_info.json` to your image folder.

### Launch Training

Training can be started with the following script.

```bash
conda activate llamafactory
bash ./slurm_jobs/sft/train_7b_lora.sh
```

<br>

# Evaluation

### Installation

Set up an environment with `vllm`.

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm==0.8.2
```

### Prepare Data and Model

The Vstar Benchmark serves as an example dataset and can be downloaded from [Vstar benchmark](https://huggingface.co/datasets/craigwu/vstar_bench).

The model can be downloaded from [Hugging Face – Cof SFT Model 7B](https://huggingface.co/xintongzhang/CoF-sft-model-7b).

### Inference

Run the inference script using vllm.

```bash
conda activate vllm
bash ./slurm_jobs/eval/inference_vstar.sh
```


### Performance Metrics

To evaluate the model's performance on the VSTAR benchmark, begin by launching a dedicated vllm server process to serve the evaluation model (e.g., a judge model):

```bash
vllm serve /path/to/Qwen2.5-VL-72B-Instruct \
    --served-model-name judge \
    --port 51232 \
    --limit-mm-per-prompt image=1 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --disable-log-requests
```

Once the vllm service is running, execute the evaluation script to compute metrics:

```bash
bash ./slurm_jobs/eval/metrics_vstar.sh
```




