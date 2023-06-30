</br>

# THE PROCESS

</br>

| # | Steps 	| Details 	|
|:-----:|--------	|---------	|
| 01 | Scrape tweets	| [keywords](); [code](https://github.com/surrey-nlp/woah-aggression-detection/blob/main/code/data-scraping/scrape-tweets.py) |
| 02 | Filter tweets that are not EN/HI 	| used [xlm-roberta-base](https://huggingface.co/papluca/xlm-roberta-base-language-detection); [code](https://github.com/surrey-nlp/woah-aggression-detection/blob/main/code/preprocessing/basic-lang-identification.py) |
| 03 | Monolingual vs Code-mixed separation | used [hing-bert-lid](https://huggingface.co/l3cube-pune/hing-bert-lid); [code](https://github.com/surrey-nlp/woah-aggression-detection/blob/main/code/preprocessing/codemix-identification.py)	|
| 03 | Mask usernames 	| [code](https://github.com/surrey-nlp/woah-aggression-detection/blob/main/code/preprocessing/mask-usernames.py)	|
| 04 | Calculate inter-annotator agreement	| used Cohen's Kappa; [code](https://github.com/surrey-nlp/woah-aggression-detection/blob/main/code/inter-annotator-agreement/cohen-kappa-calculation.py) |
| 05 | Hyperparameter search	| [code](https://github.com/surrey-nlp/woah-aggression-detection/blob/main/code/hyperparameter-tuning/aggression-detection/hing-roberta-my10k-codemixed-HT.py)	|
| 06 | Experimentation | [code](https://github.com/surrey-nlp/woah-aggression-detection/blob/main/code/experiments/aggression-detection/hing-roberta-my10k-codemixed.py) |
