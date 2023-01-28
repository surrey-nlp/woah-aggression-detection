THE PROCESS (explained)

1. Scraped tweets for approx 50 topics.
  - Python code used for obtaining data per topic is [here](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/scrapeT1.py).
  - Details about the topics are [here](https://docs.google.com/spreadsheets/d/1M8wwLU5D1V7Wiis3q7Z2mam1IM129AfawyZZG2W6Z9E/edit#gid=688753965).
  - Around 4k tweets per topic, a total of 213k+ tweets.

2. All inital raw data is at "/content/drive/MyDrive/MTechThesis/FFFallrawdata.csv" and also in the [data](https://github.com/nazianafis/MastersThesis/tree/main/data) folder here.
  - 213523 rows and 3 columns (count, id, tweet).

3. Used [this](https://huggingface.co/papluca/xlm-roberta-base-language-detection) language detection model.
  - Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/basicLangDetection.ipynb).
  - This removed all non-EN/HI tweets.
  - x rows and 4 columns (count, id, tweet, language).

4. Masked all <@usernames> with <@MASK>.
  - Python code is [here](https://github.com/nazianafis/MastersThesis/blob/main/preprocessing/maskUsernames.ipynb).

5. Sampled 10k tweets for manual labeling.
  - Data is at ""
  - Calculated Inter-Annotator Agreement for each task.

6. Another 50k labeled with weak supervision (rules)
  - Data is at ""

7. Remaining x k labeled with existing models.
  - Data is at ""

8. Used [this](https://huggingface.co/l3cube-pune/hing-bert-lid) EN-HI LID model for monolingual vs code-mix separation.
  - Data is at "" and ""

9. Created train, val, and test splits for each dataset.

10. Did Random hyperparameter search.

11. Trianed models and noted down performance
  - Details of results are [here]().
