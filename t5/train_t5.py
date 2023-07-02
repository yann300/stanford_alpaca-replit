import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import pandas as pd
import random
import torch
from simplet5 import SimpleT5
from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
import torch
import gc
from prepare_t5_data import build_train_test_split

# @title Train model (significantly undertrained as of 2022/12)
# 1 epoch takes about 25hours for A100 40G, right now only trained 3 hours

class MySimpleT5(SimpleT5):
  def __init__(self) -> None:
    super().__init__()
    self.device = torch.device("cuda")

  def load_base_codet5_model(self, use_gpu: bool = True):
    # self.tokenizer = T5Tokenizer.from_pretrained("Salesforce/codet5-large")
    # self.model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
    self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
    self.model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")

model = MySimpleT5()
model.load_base_codet5_model()
model.model = model.model.to('cuda')

(train_df, eval_df) = build_train_test_split()

print(train_df.head(10))

model.train(train_df=train_df,
            eval_df=eval_df,
            source_max_token_len=160, # Why 160? Check code below for distribution
            target_max_token_len=512, 
            batch_size=8,
            max_epochs=3,
            use_gpu=True,
            outputdir=f"./models")

# error: ValueError: text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples)

# # Release GPU memory if needed
# gc.collect()
# torch.cuda.empty_cache()
# del model
# torch.cuda.empty_cache()

# # Check distribution of input/output token length
# ins, outs, outs_reduce_whitespace = [], [], []
# if_truncate = False
# tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
# for i in trange(len(eval_df)):
#   s, o = eval_df['source_text'].to_numpy()[i], eval_df['target_text'].to_numpy()[i]
#   ins.append(int(tokenizer(s, return_tensors="pt", truncation=if_truncate).input_ids.shape[-1]))
#   outs.append(int(tokenizer(o, return_tensors="pt", truncation=if_truncate).input_ids.shape[-1]))

# plt.hist(ins, bins=50)
# plt.show() 

# plt.hist(outs, bins=50)
# plt.show() 

# To watch GPU usage, use this command
# watch -n 0.5 nvidia-smi
'''
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8
sudo apt-get install python3-pip

sudo ubuntu-drivers install --gpgpu
'''
