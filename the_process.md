</br>

# THE PROCESS

</br>

| # | Action 	| Details 	|
|:-----:|--------	|---------	|
| 01   	| Scrape tweets    	| <ul> <li>50 topics, around 4k tweets per topic, a total of 200k+ tweets</li> <li>Details about the topics are [here](https://docs.google.com/spreadsheets/d/1M8wwLU5D1V7Wiis3q7Z2mam1IM129AfawyZZG2W6Z9E/edit#gid=688753965).</li> <li>Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/scrapeT1.py).</li> <li> All data is in the [data](https://github.com/nazianafis/MastersThesis/tree/main/data) folder.</li> <li>213523 records.</li></ul> |
| 02   	| Filter tweets that are not EN/HI 	| <ul> <li>Used [xlm-roberta-base](https://huggingface.co/papluca/xlm-roberta-base-language-detection) language detection model.</li> <li>Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/basicLangDetection.ipynb).</li>  </ul> 	|
| 03   	| Mask unames 	| <ul> <li>Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/maskUsernames.ipynb).</li> </ul>  	|
| 04    | Create subsets 	| <ul> <li>Sampled 10k tweets for manual labeling.</li> <li>Another 50k labeled with weak supervision.</li> <li>Remaining 110k labeled with existing models.</li> </ul> |
| 05    | Annotations 	| <ul> <li>Get annotations and calculate the inter-annotator agreement.</li> <li>Use Cohen's Kappa.</li> <li>Python code is [here.](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/cohen.py)</li> |
| 06   	| Topic modeling | <ul> <li>Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/TopicModeling/onModifiedData/TopicModelingFinal.ipynb).</li> <li>Results are [here](https://drive.google.com/drive/u/3/folders/1pC9-FRhxGZRAoW7BfyMBmZrrLIRtuMav) (Google Drive link).</li> </ul>
| 07   	| Monolingual seperation | <ul> <li>Used [hing-bert-lid](https://huggingface.co/l3cube-pune/hing-bert-lid) model for monolingual vs code-mix language separation.</li> <li>Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/hyperparam-tuning/AGG_hing-bert_50k.py)</li> </ul> 	|
| 08   	| * Hyperparameter search	| <ul> <li>Find best set of hyperparameters for each model-dataset combination.</li> <li>Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/hyperparam-tuning/AGG_hing-bert_50k.py).</li> </ul>	|
| 09   	| * Experimentation | <ul> <li>Experiment and evaluate metrics for each model-dataset combination.</li> <li>Python code is [here]().</li> </ul> |
| 10   	| * Error Analysis	| <ul> <li>Qualitative analysis of results.</li> </ul> |
