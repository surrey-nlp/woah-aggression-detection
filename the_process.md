</br>

# THE PROCESS

</br>

| # | Action 	| Details 	|
|:-----:|--------	|---------	|
| 01   	| Scrape tweets    	| <ul> <li>50 topics, around 4k tweets per topic, a total of 200k+ tweets</li> <li>Details about the topics are [here](https://docs.google.com/spreadsheets/d/1M8wwLU5D1V7Wiis3q7Z2mam1IM129AfawyZZG2W6Z9E/edit#gid=688753965).</li> <li>Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/scrapeT1.py).</li> <li>Data is at "/content/drive/MyDrive/MTechThesis/FFFallrawdata.csv" and also [here](https://github.com/nazianafis/MastersThesis/blob/main/data/FFFallrawdata.csv) in the [data](https://github.com/nazianafis/MastersThesis/tree/main/data) folder.</li> <li>213523 rows and 3 columns (count, id, tweet)</li></ul> |
| 02   	| Remove non-EN and non-HI data    	| <ul> <li>Used [xlm-roberta-base](https://huggingface.co/papluca/xlm-roberta-base-language-detection) language detection model.</li> <li>Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/basicLangDetection.ipynb).</li> <li>x rows and 4 columns (count, id, tweet, language).</li> </ul> 	|
| 03   	| Mask unames 	| <ul> <li>Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/maskUsernames.ipynb).</li> </ul>  	|
| 04   	| Create subsets 	| <ul> <li>Sampled 10k tweets for manual labeling, Data is at [here]()</li> <li>Calculated Inter-Annotator Agreement for each task.</li><li>Another 50k labeled with weak supervision, Data is at ""</li> <li>Remaining x k labeled with existing models, Data is at [here]()</li> </ul>        	|
| 05   	| Monolingual vs Code-mixed | <ul> <li>Used [this](https://huggingface.co/l3cube-pune/hing-bert-lid) EN-HI LID model for monolingual vs code-mix separation, Data is [here]()</li> </ul> 	|
| 06   	| Split train/val/test  	|         	|
| 07   	| Hyperparameter search       	|         	|
| 08   	|        	|         	|
| 09   	|        	|         	|
| 10   	|        	|         	|
