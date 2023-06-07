import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import pipeline
classify = pipeline(model="l3cube-pune/hing-bert-lid")

myfile = pd.read_csv("/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Datasets/Raw/all-data-v2.csv", dtype=str)

tweetscol = list(myfile['tweet'])
Langlabs = classify(tweetscol)

new_list = list()
for i in range(0, len(Langlabs)):
  temp_list = list()
  for x in Langlabs[i]:
    temp_list.append(x['entity'])
  new_list.append(temp_list)

# Map 0 --> English monolingual; 1 --> Hindi-English code-mixed
final_list = []
for i in range(0, len(new_list)):
  hi = 0
  for x in new_list[i]:
    if x == 'HI':
      hi += 1
  try:
    if round(hi/len(new_list[i]), 2) >= 0.12:
      final_list.append('1')
    else:
      final_list.append('0')
  except:
    final_list.append('-1')
print(final_list)

myfile['codemixed'] = final_list

myfile.to_csv('all-data-v3.csv')
