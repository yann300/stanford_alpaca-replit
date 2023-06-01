
<p align="center" width="100%">
<a href="https://crfm.stanford.edu/alpaca/" target="_blank"><img src="assets/logo.png" alt="Stanford-Alpaca" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

# Stanford Alpaca Trainer - Updated for use training Replit's Code Model

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This was the repo for the Stanford Alpaca project, which is edited to become a trainer for Alpaca-format datasets over Replit's 3B Code Model:

- The Base Model: [Replit 3B Code](https://huggingface.co/replit/replit-code-v1-3b)

- The code for [fine-tuning the model](#fine-tuning).


## Overview

A trainer for Replit's 3B parameter code model.

## Dataset Format

Alpaca format datasets should be in the following format, in json:

- `instruction`: `str`, describes the task the model should perform. Each of the 52K instructions is unique.
- `input`: `str`, optional context or input for the task. For example, when the instruction is "Summarize the following article", the input is the article. Around 40% of the examples have an input.
- `output`: `str`, the answer to the instruction as generated by `text-davinci-003`.

### Here is an example of a dataset:
```  
[
  {
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
  },
  {
    "instruction": "What are the three primary colors?",
    "input": "",
    "output": "The three primary colors are red, blue, and yellow. These colors are called primary because they cannot be created by mixing other colors and all other colors can be made by combining them in various proportions. In the additive color system, used for light, the primary colors are red, green, and blue (RGB)."
  },  
]
  ```  

We used the following prompts for fine-tuning the Replit model:

- for examples with a non-empty input field:

 ```
 ### Instruction:
 {instruction}
 
 ### Input:
 {input}
 
 ### Response:
 ```

- for examples with an empty input field:

 ```
 ### Instruction:
 {instruction}
 
 ### Response:
 ```

## Fine-tuning

To fine-tune for Replit's model, first install the requirements

```bash
pip install -r requirements.txt
```

The train.py script defaults to 2000 sequence length for training. It runs in small batch size at this sequence length on an a100 80gb. You will save a significant amount of vram, and thus, can train faster, with a smaller sequence length. Training on 2x a100 80gb with what is possible with 2000 token sequence length takes about 2.5 hours, with 512 token length, only 45~ minutes. 

Below is a command that fine-tunes Replit-3B with an alpaca-formated dataset on a machine with 2 A100 80G GPUs with 2000 token sequence length.

Replace `<your_random_port>` with a port of your own, `<path_to_replit_model>` with the path to your converted checkpoint and tokenizer or leave default for Replit's base code model, and `<your_output_dir>` with where you want to store your outputs.

```bash
torchrun --nproc_per_node=2 --master_port=<your_random_port> train.py \
    --model_name_or_path <path_to_replit_model> \
    --data_path ./<your_dataset>.json \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
```

Note the given training script is meant to be simple and easy to use, and is not particularly optimized.
To run on more gpus, you may prefer to turn down `gradient_accumulation_steps` to keep a global batch size of 128. Global batch size has not been tested for optimality.

### Addressing OOM

Naively, fine-tuning a 7B model requires about 7 x 4 x 4 = 112 GB of VRAM. Commands given above enable parameter sharding, so no redundant model copy is stored on any GPU.
If you'd like to further reduce the memory footprint, here are some options:

- Turn on CPU offload for FSDP with `--fsdp "full_shard auto_wrap offload"`. This saves VRAM at the cost of longer runtime.
- In our experience, DeepSpeed stage-3 (with offload) can at times be more memory efficient than FSDP with offload. Here's an example to use DeepSpeed stage-3 with 4 GPUs with both parameter and optimizer offload:
    ```bash
    pip install deepspeed
    torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
        --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
        --data_path ./alpaca_data.json \
        --bf16 True \
        --output_dir <your_output_dir> \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --deepspeed "./configs/default_offload_opt_param.json" \
        --tf32 True
    ```
  - The DeepSpeed library also provides some [helpful functions](https://deepspeed.readthedocs.io/en/latest/memory.html) to estimate memory usage. 
- [LoRA](https://arxiv.org/abs/2106.09685) fine-tunes low-rank slices of the query, key, and value embedding heads. This can reduce the total memory footprint from 112GB to about 7x4=28GB. We may release our re-implemention of this in the future, but for now the [peft](https://github.com/huggingface/peft) codebase can be a useful resource.

### Original Authors of the Alpaca paper

All grad students below contributed equally and the order is determined by random draw.

- [Rohan Taori](https://www.rohantaori.com/)
- [Ishaan Gulrajani](https://ishaan.io/)
- [Tianyi Zhang](https://tiiiger.github.io/)
- [Yann Dubois](https://yanndubs.github.io/)
- [Xuechen Li](https://www.lxuechen.com/)

All advised by [Tatsunori B. Hashimoto](https://thashim.github.io/). Yann is also advised by [Percy Liang](https://cs.stanford.edu/~pliang/) and Xuechen is also advised by [Carlos Guestrin](https://guestrin.su.domains/).

### Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```

Naturally, you should also cite the original LLaMA paper [1] and the Self-Instruct paper [2].

### Acknowledgements

We thank Yizhong Wang for his help in explaining the data generation pipeline in Self-Instruct and providing the code for the parse analysis plot.
We thank Yifan Mai for helpful support, and members of the Stanford NLP Group as well as the Center for Research on Foundation Models (CRFM) for their helpful feedback.
