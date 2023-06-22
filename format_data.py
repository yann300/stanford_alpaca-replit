import re
from tqdm import trange
import pandas as pd

# @title Clean text library: `clean(src)`
# Notice we also removed all comments, might need 2nd thought

SEPRATORS = ('\nabstract contract', '\ncontract', '\nlibrary', '\ninterface', '\nstruct')

def _remove_comment(src_in):
  # multi line
  src_in = re.sub("\/\*(\*(?!\/)|[^*])*\*\/", "", src_in)
  # single line, maybe keep?
  src_in = re.sub("\/\/*.*", "", src_in)
  return src_in

def _remove_header_txt(src_in):
  if '\npragma solidity' not in src_in:
    return src_in
  p = src_in.index('\npragma solidity')
  if p > 0:
    return src_in[p + 1:]  # new line no need
  return src_in

def _remove_extra_new_line(src_in):
  src_in = src_in.strip()
  # remove empty content lines
  src_in = re.sub("(\s)+(\n)", "\n", src_in)
  src_in = re.sub("(\n)+", "\n", src_in)
  return src_in

def _replace_addr(src_in):
  return re.sub("0x[A-Fa-f0-9]{40}", "YOUR_ADDR", src_in)

def _format_src(src_in):
  # remove extra space before new line
  src_in = re.sub("\s+\n", "\n", src_in)
  # format the method or class desclaration so each { has exactly one space before
  src_in = re.sub(r"(.){", r"\1 {", src_in)
  src_in = re.sub("\s+{", r" {", src_in)
  src_in = src_in.replace("( ", "(")
  src_in = src_in.replace(" )", ")")
  src_in = src_in.replace("[ ", "[")
  src_in = src_in.replace(" ]", "]")
  # Remove unnecessary spaces in method declare
  src_in = re.sub("\n\s+external\s ", r" external ", src_in)
  src_in = re.sub("\n\s+internal\s", r" internal ", src_in)
  src_in = re.sub("\n\s+public\s", r" public ", src_in)
  src_in = re.sub("\s+poolOnly\s", r" poolOnly ", src_in)
  src_in = re.sub("\s+returns\(", r" returns(", src_in)
  # '\nabstract contract', '\ncontract', '\nlibrary', '\ninterface'
  src_in = re.sub("}\s+abstract contract ", r"}\nabstract contract ", src_in)
  src_in = re.sub("}\s+contract ", r"}\ncontract ", src_in)
  src_in = re.sub("}\s+library ", r"}\nlibrary ", src_in)
  src_in = re.sub("}\s+interface ", r"}\ninterface ", src_in)
  src_in = re.sub("}\s+struct ", r"}\nstruct ", src_in)
  src_in = re.sub(";\s+abstract contract ", r";\nabstract contract ", src_in)
  src_in = re.sub(";\s+contract ", r";\ncontract ", src_in)
  src_in = re.sub(";\s+library ", r";\nlibrary ", src_in)
  src_in = re.sub(";\s+interface ", r";\ninterface ", src_in)
  src_in = re.sub(";\s+struct ", r";\nstruct ", src_in)
  # special, typo "ontract"
  src_in = re.sub("}\s+ntract ", r"}\ncontract ", src_in)
  src_in = src_in.replace("}contract ", "}\ncontract ")
  src_in = src_in.replace("}interface ", "}\ninterface ")
  src_in = src_in.replace("}struct ", "}\nstruct ")
  return src_in

def clean(src):
  src = _remove_comment(src)
  src = _remove_header_txt(src)
  src = _remove_extra_new_line(src)
  src = _replace_addr(src)
  src = _format_src(src)
  return src

# @title Split to segments (e.g. contracts) `process_single_line(src)`
def _extract_pub_funcs(seg):
  pub_funcs = re.findall("function [A-Za-z0-9_]+\(", seg)
  if pub_funcs:
    pub_funcs = [s[len('function '):-1] for s in pub_funcs
                 if not s[len('function '):-1].startswith('_') and not s[len('function '):-1].endswith('_')]
  return pub_funcs

def _extract_constants(seg):
  constants = re.findall(r"constant [A-Za-z0-9_]+", seg)
  if constants:
    constants = [s[len('constant '):] for s in constants]
  return constants


def _extract_base_parents(seg):
  base_with_parents = re.findall("[A-Za-z0-9]+ is [A-Za-z0-9, \n]+ {", seg)
  base, parents = None, []
  if base_with_parents:
    assert 1 == len(base_with_parents), "base_with_parents pattern can only have 1 match"
    splits = base_with_parents[0].split(' is ')
    assert 2 == len(splits), "cannot have more than 2 splits for base extraction"
    base = splits[0]
    parents = [p.strip() for p in splits[1][:-2].split(',')]
  else:
    base_only = re.findall("[A-Za-z0-9]+\s+{", seg)
    if base_only:
      base = base_only[0].split()[0]
      parents = []
  return base, parents

DEFAULT_SOL_VERSION = "pragma solidity ^0.6.0;";
def _prepare_seg_map(segs):
  if not segs[0].startswith('pragma solidity'):
    segs.insert(0, DEFAULT_SOL_VERSION)
  seg_map = {}
  for s in segs:
    base, parents =  _extract_base_parents(s)
    if base:
      seg_map[base] = {
          'parents': parents,
          'constants': _extract_constants(s),
          'pub_funcs': _extract_pub_funcs(s),
          'v': segs[0], # version first line
          'clean_src': s,
      }
  return seg_map

#@title Split the text now
def _split_segments(src):
  start = 0
  segments = []
  while True:
    # Find the next closest seprator position
    next_sep = len(src) + 1
    seg_keyword = ""
    seg_type = ''
    for sep in SEPRATORS:
      # print("next_sep", next_sep)
      # print("start", start)
      cur_src = src[start:]
      if sep in cur_src:
        sep_ind = cur_src.index(sep)
        if sep_ind > 0 and next_sep > sep_ind:
          next_sep = sep_ind
          seg_keyword = cur_src[sep_ind + len(sep) + 1:].split()[0]
          seg_type = sep[1:]
    if next_sep > len(src):
      if start < len(src) - 1:
        segments.append(src[start:].strip())
      break
    else:
      segments.append(src[start:start + next_sep].strip())
      start += next_sep + 1
  return segments

def _find_ancestors(seg_map):
  for k in seg_map:
    parents = seg_map[k]['parents']
    if parents:
      ancestors = parents.copy()
      idx = 0
      while (idx < len(ancestors)):
        if ancestors[idx] in seg_map:
          # Be careful of cycle dependency
          for more_parent in seg_map[ancestors[idx]]['parents']:
            if more_parent not in ancestors and ancestors != k:
              ancestors.append(more_parent)
        idx += 1
      seg_map[k]['ancestors'] = ancestors
    else:
      seg_map[k]['ancestors'] = []
  return seg_map

def process_single_line(src):
  """Clean text, split to segments, prepare segment map with ancestors."""
  src = clean(src)
  segs =  _split_segments(src)
  seg_map = _prepare_seg_map(segs)
  seg_map = _find_ancestors(seg_map)
  return seg_map

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
    if '{\n' not in v['clean_src']:
      continue
    s = v['v'] + "\n"
    for a in v['ancestors']:
      s += _get_single_ancestor_metadata(a, seg_map) + "\n"
    raw_src_code = v['clean_src']
    my_raw = raw_src_code
    header_split_indx = raw_src_code.index('{\n')
    s += raw_src_code[:header_split_indx + 1] # include "{"
    o = _reduce_out_whitespace(raw_src_code[header_split_indx + 2:])
    ins.append(s)
    outs.append(o)
  return ins, outs

bad_sample = []
def convert_to_df(ds):
  all_ins, all_outs = [], []
  for i in trange(len(ds)):
    src = ds[i]
    my_src2 = src
    try:
      ins, outs = prepare_t5_data(src)
    except:
      bad_sample.append(src)
      continue
    all_ins.extend(ins)
    all_outs.extend(outs)
  return pd.DataFrame({
      'instruction': all_ins,
      'output': all_outs,
  })
