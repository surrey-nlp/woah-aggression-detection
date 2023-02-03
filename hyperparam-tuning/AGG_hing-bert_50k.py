import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from torch import nn
from transformers import Trainer
import json
import os



class AggressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)




class Seq_Classifier_HT:

    model_ckpt = '/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Models/hing-bert'
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    device = torch.device('cuda')
    ht_results = {
    'Classifier' : [0], 'F1' : [-10]
    }

    def __init__(self):
        pass

    def read_data(self):
        train_set = pd.read_csv('/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Datasets/Finalized/Aggression/AGG_50k_train.csv')
        valid_set = pd.read_csv('/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Datasets/Finalized/Aggression/AGG_50k_test.csv')
        return (train_set, valid_set)


    def process_DS(self):
        train_df, valid_df= self.read_data()

        train_encodings = Seq_Classifier_HT.tokenizer(list(train_df['tweet']), truncation=True, padding=True)
        valid_encodings = Seq_Classifier_HT.tokenizer(list(valid_df['tweet']), truncation=True, padding=True)
        
        train_labels = list(train_df['aggression'])
        valid_labels = list(valid_df['aggression'])

        train_dataset = AggressionDataset(train_encodings, train_labels)
        valid_dataset = AggressionDataset(valid_encodings, valid_labels)

        return (train_dataset, valid_dataset)


    def model_init(self):
        model = (AutoModelForSequenceClassification.from_pretrained(Seq_Classifier_HT.model_ckpt, num_labels=3))
        return model


    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        f1 = f1_score(labels, preds, average='macro')
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    def find_best(self, log_dir):
        scores = []
        rootdir = log_dir
        for it in os.scandir(rootdir):
            if it.is_dir():
                if it.path != rootdir+'.ipynb_checkpoints':
                    f = open(it.path+'/trainer_state.json')
                    scores.append(json.load(f)['best_metric'])

        if scores[0] >= scores[1]:
            return scores[0]
        else:
            return scores[1]


    def store_in_dict(self, best_val, itr):
        Seq_Classifier_HT.ht_results['Classifier'].append(itr)
        Seq_Classifier_HT.ht_results['F1'].append(best_val)


    def fine_tune_args(self, log_dir, bs, lr, itr):
      # Defining hyperparameters
        # logging_steps = train_size // eval_batch_size
        # model_name = f"{model_ckpt}-finetuned-Ours-DS"
        training_args = TrainingArguments(output_dir=log_dir,
                                          num_train_epochs=2,
                                          learning_rate=lr,
                                          per_device_train_batch_size=bs,
                                          per_device_eval_batch_size=bs,
                                          evaluation_strategy='epoch',
                                          save_strategy='epoch',
                                          logging_dir=log_dir,
                                          logging_strategy='epoch',
                                          metric_for_best_model="eval_f1",
                                          greater_is_better=True,
                                          load_best_model_at_end=True, 
                                          disable_tqdm=False,
                                          log_level='info', report_to="none")
        return training_args


    def fine_tune_model(self):
        train_DS, valid_DS = self.process_DS()
        model_name = "AGG-50k-hing-bert" 

        batch_size = [8, 16, 32]
        learn_rate = [1e-3, 1e-4, 1e-5, 1e-6, 3e-3, 3e-4, 3e-5, 3e-6, 5e-3, 5e-4, 5e-5, 5e-6]
        itr = 0
        for bs in batch_size:
            for lr in learn_rate:
                itr += 1
                log_dir = model_name + "/" + "clf-" + str(itr) + "/"
                train_args = self.fine_tune_args(log_dir, bs, lr, itr)
                trainer = Trainer(model_init=self.model_init,
                                    args=train_args,
                                    compute_metrics = self.compute_metrics,
                                    train_dataset = train_DS,
                                    eval_dataset = valid_DS,
                                    tokenizer = Seq_Classifier_HT.tokenizer)

                trainer.train()
                best_from_clf = self.find_best(log_dir)
                self.store_in_dict(best_from_clf, itr)

        res_df = pd.DataFrame(Seq_Classifier_HT.ht_results)
        res_df.to_csv('AGG_50k_hing-bert.csv')



def main():
    SC = Seq_Classifier_HT()
    SC.fine_tune_model()


if __name__ == "__main__":
    main()
