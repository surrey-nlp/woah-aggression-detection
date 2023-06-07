# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from torch import nn
from transformers import Trainer


class OffenseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        device = torch.device('cuda')
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.33, 0.67]).to(device))    # Compute custom loss
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class Seq_Classifier:

    model_ckpt = '/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Models/hing-roberta' # Path to the locally saved model checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    device = torch.device('cuda')

    def __init__(self):
        pass

    def read_data(self):
        train_set = pd.read_csv('/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Datasets/Finalized/Offense/codemixed/CMoff-train.csv', encoding='latin-1')
        valid_set = pd.read_csv('/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Datasets/Finalized/Offense/codemixed/CMoff-val.csv', encoding='latin-1')
        test_set = pd.read_csv('/jmain02/home/J2AD011/hxc16/dxk50-hxc16/workspace/nn007/Datasets/Finalized/Offense/codemixed/CMoff-test.csv', encoding='latin-1')
        return (train_set, valid_set, test_set)

    def process_DS(self):
        train_df, valid_df, test_df = self.read_data()

        train_encodings = Seq_Classifier.tokenizer(list(train_df['tweet']), max_length=510, truncation=True, padding=True)
        valid_encodings = Seq_Classifier.tokenizer(list(valid_df['tweet']), max_length=510, truncation=True, padding=True)
        test_encodings = Seq_Classifier.tokenizer(list(test_df['tweet']), max_length=510, truncation=True, padding=True)
        
        train_labels = list(train_df['offense'])
        valid_labels = list(valid_df['offense'])
        test_labels = list(test_df['offense'])

        train_dataset = OffenseDataset(train_encodings, train_labels)
        valid_dataset = OffenseDataset(valid_encodings, valid_labels)
        test_dataset = OffenseDataset(test_encodings, test_labels)

        return (train_dataset, valid_dataset, test_dataset)

    def model_init(self):
        model = (AutoModelForSequenceClassification.from_pretrained(Seq_Classifier.model_ckpt, num_labels=2))
        return model

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        f1 = f1_score(labels, preds, average='macro')
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

    def classification_report_csv(self, report, log_dir, s):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:]:
            if len(line) != 0:
                row = {}
                row_data = line.split()
                if len(row_data) == 3:
                    row['class'] = row_data[0]
                    row['precision'] = '-'
                    row['recall'] = '-'
                    row['f1_score'] = float(row_data[1])
                    row['support'] = int(row_data[2])
                elif len(row_data) == 6:
                    row['class'] = row_data[0]+" "+row_data[1]
                    row['precision'] = float(row_data[2])
                    row['recall'] = float(row_data[3])
                    row['f1_score'] = float(row_data[4])
                    row['support'] = int(row_data[5])
                else:
                    row['class'] = row_data[0]
                    row['precision'] = float(row_data[1])
                    row['recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    row['support'] = int(row_data[4])
                report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(log_dir+'/'+'classification_report_'+str(s)+'.csv', index = False)

    def gen_results_on_test(self, trainer, test_dataset, log_dir, s):
        preds_output_test = trainer.predict(test_dataset)
        # preds_output_test.metrics
        y_preds_test = np.argmax(preds_output_test.predictions, axis=1)
        y_valid_test = np.array(test_dataset.labels)
        map_dt = {0:'NOT', 1:'OFF'}    # Map labels to aggression classes
       
        rep = classification_report(y_valid_test, y_preds_test, target_names=list(map_dt.values()), digits=4)
        self.classification_report_csv(rep, log_dir, s)
        
        y_valid_trying = map(lambda x : map_dt[x], y_valid_test)
        y_valid_trying = list(y_valid_trying)
        y_preds_trying = map(lambda x : map_dt[x], y_preds_test)
        y_preds_trying = list(y_preds_trying)
        cm_labels = np.unique(y_valid_trying)
        cm_array = confusion_matrix(y_valid_trying, y_preds_trying)
        cm_array_df = pd.DataFrame(cm_array, index=cm_labels, columns=cm_labels)
        sns.heatmap(cm_array_df, annot=True, annot_kws={"size": 12})  # Save this cm to log_dir
        plt.savefig(log_dir+"/"+"Codemixed_"+str(s)+".png")
        plt.clf()

    def fine_tune_args(self, log_dir, rd_seed):
      # Define hyperparameters
        training_args = TrainingArguments(output_dir=log_dir,
                                          num_train_epochs=5,
                                          learning_rate=3e-5,
                                          per_device_train_batch_size=8,
                                          per_device_eval_batch_size=8,
                                          evaluation_strategy='epoch',
                                          save_strategy='epoch',
                                          logging_dir=log_dir,
                                          logging_strategy='epoch',
                                          # max_steps=-1,
                                          # warmup_ratio=0.0,
                                          seed=42,
                                          data_seed=rd_seed,
                                          metric_for_best_model="eval_f1",
                                          greater_is_better=True,
                                          load_best_model_at_end=True, 
                                          disable_tqdm=False,
                                          log_level='info', report_to="none")
                                          # push_to_hub=True)
        return training_args

    def fine_tune_model(self):
        train_DS, valid_DS, test_DS = self.process_DS()
        model_name = "hing-roberta-my10k-codemixed"   # Give a name to save the fine-tuned model checkpoint

        for s in range(1, 6):
            log_dir = model_name + "/" + "run-" + str(s) + "/"
            train_args = self.fine_tune_args(log_dir, s)
            trainer = CustomTrainer(model_init=self.model_init,
                                args=train_args,
                                compute_metrics = self.compute_metrics,
                                train_dataset = train_DS,
                                eval_dataset = valid_DS,
                                tokenizer = Seq_Classifier.tokenizer)

            trainer.train()

            self.gen_results_on_test(trainer, test_DS, log_dir, s)


def main():
    SC = Seq_Classifier()
    SC.fine_tune_model()


if __name__ == "__main__":
    main()
