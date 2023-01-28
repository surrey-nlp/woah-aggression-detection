THE PROCESS (explained)

1. Collected data for approx 50 topics.
  - Details about the topics are [here](https://docs.google.com/spreadsheets/d/1M8wwLU5D1V7Wiis3q7Z2mam1IM129AfawyZZG2W6Z9E/edit#gid=688753965).
  - Python code used for obtaining data per topic is [here]().

2. All inital raw data is at "/content/drive/MyDrive/MTechThesis/FFFallrawdata.csv"
  - 213523 rows and 3 columns (count, id, tweet).

3. Used [this](https://huggingface.co/papluca/xlm-roberta-base-language-detection) language detection model.
  - This removed all non-EN/HI tweets.
  - x rows and 3 columns (count, id, tweet).
  - Masked all @usernames with @MASK.

4. Sampled 10k tweets for manual labeling.
  - Data is at ""
  - Calculated Inter-Annotator Agreement for each task.

5. Another 50k labeled with weak supervision (rules)
  - Data is at ""

6. Remaining x k labeled with existing models.
  - Data is at ""

7. Used [this](https://huggingface.co/l3cube-pune/hing-bert-lid) EN-HI LID model for monolingual vs code-mix separation.
  - Data is at "" and ""

8. Created train, val, and test splits for each dataset.

9. Did Random hyperparameter search.

10. Trianed models and noted down performance
  - Details of results are [here]().
