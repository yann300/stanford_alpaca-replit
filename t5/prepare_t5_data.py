import re
import pandas as pd
from clean_t5 import clean, split
from tqdm import trange
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

# @title Generate T5-friendly data func `prepare_t5_data(src)`
def _get_single_ancestor_metadata(an, seg_map):
  if an not in seg_map:
    return ""
  pub_func_str = " ".join(seg_map[an]['pub_funcs'])
  const_str = " ".join(seg_map[an]['constants'])
  return f"// Context: {an} | Functions: {pub_func_str} | Constants: {const_str}"

def _reduce_out_whitespace(out_src):
  # remove extra spaces (ignore identation) and replace "; " with ";\n"
  out_src = re.sub("\s+", " ", out_src)
  out_src = out_src.replace("; ", ";\n")
  out_src = out_src.replace("{ ", "{\n")
  out_src = out_src.replace("} ", "}\n")
  return out_src.strip()

my_src = ""
my_seg = None
my_raw = ''
def prepare_t5_data(src):
  my_src = src
  seg_map = process_single_line(src)
  my_seg = seg_map
  ins, outs = [], []
  for k, v in seg_map.items():
    # Some headers does not have content
    if '{' not in v['clean_src']:
      continue
    s = v['v'] + "\n"
    for a in v['ancestors']:
      s += _get_single_ancestor_metadata(a, seg_map) + "\n"
    raw_src_code = v['clean_src']
    my_raw = raw_src_code
    header_split_indx = raw_src_code.index('{')
    s += raw_src_code[:header_split_indx + 1] # include "{"
    o = _reduce_out_whitespace(raw_src_code[header_split_indx + 2:])
    ins.append(s)
    outs.append(o)
  return ins, outs

def prepare_t5_data_custom(src):
  my_src = src
  src = split(clean(src))
  # print(src)
  ins, outs = [], []
  window = 50
  splitted = np.array_split(src, len(src) // window)
  for context in splitted:
    # print(context)
    for sentence in range(len(context)):
      ins.append(" ".join(context[:sentence + 1]))
      outs.append("".join(context[sentence + 1:sentence + 2]))
  
  return ins, outs

def load_solidity_dataset():
  # @title Load all raw data (train, validation, test), ~3 mins
  # Available: ['all-plain-text', 'all-multilabel', 'big-plain-text', 'big-multilabel', 'small-plain-text', 'small-multilabel']
  # Checksum error as of Dec 2022, have to set ignore_verifications to True
  HF_DATA_SOURCE = "mwritescode/slither-audited-smart-contracts"
  DATA_TYPE = "all-plain-text"  # change to 'small-plain-text for debugging
  all_ds = load_dataset(HF_DATA_SOURCE, DATA_TYPE, split="train",
                        revision="main", ignore_verifications=True)
  # Small data types has validation/test as well
  print("DS size", len(all_ds))

  all_source_ds = all_ds['source_code']
  print("all_source_ds size", len(all_source_ds))
  return all_source_ds

  # Why set 50k limit? Too large, and it covers 80% already
  # lens = [len(all_source_ds[i]) for i in range(len(all_source_ds))]
  # lens = [l for l in lens if l < 50000]
  # print(len(lens))
  # plt.hist(lens)
  # plt.show()

# @title Convert to DataFrame for simpleT5, ~15 mins
TEST_RATE = 0.05
bad_sample = []
def convert_to_df(ds):
  all_ins, all_outs = [], []
  for i in trange(len(ds)):
    src = ds[i]
    my_src2 = src
    try:
      ins, outs = prepare_t5_data_custom(src)
    except:
      bad_sample.append(src)
      continue
    all_ins.extend(ins)
    all_outs.extend(outs)
  return pd.DataFrame({
      'source_text': all_ins,
      'target_text': all_outs,
  })

def build_train_test_split():
  all_source_ds = load_solidity_dataset()
  filtered_all_source_ds = [s for s in all_source_ds if len(s) < 50000 and len(s.strip()) > 100 and '{\n' in s]
  print("filtered_all_source_ds size", len(filtered_all_source_ds))
  # 19614
  all_df = convert_to_df(filtered_all_source_ds) # change to samples if needed
  all_df = all_df.sample(frac=1) # Shuffle
  train_df, eval_df = train_test_split(all_df, test_size=TEST_RATE)

  # # Debug only
  # for i in range(19613+63619, 19613+63619 + 4):
  #   print(i)
  #   src = filtered_all_source_ds[i] # ?? stuck
  #   src = clean(src)
  #   segs =  _split_segments(src)
  #   seg_map = _prepare_seg_map(segs)
  #   seg_map = _find_ancestors(seg_map)

  # for i in range(len(segs)):
  #   print(i)
  #   _extract_base_parents(segs[i])

  # print(segs[4])
  print("Notice bad samples: ", len(bad_sample))

  # Save a copy for future reuse
  # train_df.to_parquet(f"./processed_data_train.parquet")
  # eval_df.to_parquet(f"./processed_data_eval.parquet")

  train_df.to_pickle(f"./processed_data_train.parquet")
  eval_df.to_pickle(f"./processed_data_eval.parquet")
  # train_df = pd.read_parquet(f"{PATH}/test_data_train")
  # eval_df = pd.read_parquet(f"{PATH}/test_data_eval")
  return train_df, eval_df