import re
from clean_t5 import clean

SEPRATORS = ('\nabstract contract', '\ncontract', '\nlibrary', '\ninterface', '\nstruct')

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
  # segs =  _split_segments(src)
  # seg_map = _prepare_seg_map(segs)
  # seg_map = _find_ancestors(seg_map)
  return seg_map
