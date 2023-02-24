# coding: utf-8
import numpy as np
import pandas as pd
from fractions import Fraction
from math import gcd, floor
import torch
from tqdm import tqdm, tqdm_notebook


import deepinv as dinv
from poly_template_gen import setup_polynomial_data
from inv_postprocessing import filter_coeffs, decompose_coeffs

def main():
  df = pd.read_csv("../benchmarks/nla/csv/ps2_1.csv", skipinitialspace=True)
  data = df.drop(columns=['trace_idx', 'init', 'final', 'while_counter', 'run_id'], errors='ignore')
  inputs_np = np.array(data, copy=True)
  inputs = torch.from_numpy(inputs_np).float()
  inputs = torch.add(torch.abs(inputs), 1)
  inputs_log = torch.log(inputs)
  print(inputs_log)
  print("Tensor sizes:")
  print(inputs_log.size())
  print(inputs_log.size(0))
  print(inputs_log.size(1))


if __name__ == "__main__":
  main()
