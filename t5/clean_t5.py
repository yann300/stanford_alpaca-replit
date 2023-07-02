# @title Clean text library: `clean(src)`
# Notice we also removed all comments, might need 2nd thought

import re

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

def split (src):
    return src.split()

def clean(src):
  src = _remove_comment(src)
  # src = _remove_header_txt(src)
  src = _remove_extra_new_line(src)
  src = _replace_addr(src)
  src = _format_src(src)
  return src