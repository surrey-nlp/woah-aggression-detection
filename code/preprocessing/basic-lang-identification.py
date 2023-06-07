import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import pipeline
classify = pipeline(model="papluca/xlm-roberta-base-language-detection", max_length=510, truncation=True)

df = pd.read_csv("/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Datasets/Raw/all-data-v1.csv", dtype = str)

tweetcol1 = list(df['tweet'])
langLabels = classify(tweetcol1)

final_labels = [i['label'] for i in langLabels]

df['language'] = final_labels

for i in range(len(final_labels)):
    if final_labels[i] == 'en' or final_labels[i] == 'hi':
        continue
    else:
        df = df.drop(i)
        
df.to_csv('all-data-v2.csv')
