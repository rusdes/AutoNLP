import pandas as pd
from transformers import AlbertTokenizerFast, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from datasets import Dataset
import time
from sklearn.metrics import accuracy_score
import os
import torch
from datasets import load_metric
import numpy as np


torch.cuda.empty_cache()
os.environ["WANDB_DISABLED"] = "true"

class Albert():
    def __init__(self, path, parameters):
        self.path = path
        self.parameters = parameters
    
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
        }

    def pipeline(self):
        batch_size = 1


        data = pd.read_csv(self.path)
        
        if len(data) > 30000:
            data = data[:30000]

        data.rename(columns = {list(data)[0]:'content', list(data)[1]:'label'}, inplace=True)

        print("len:", len(data))

        X_train, X_test = train_test_split(data, test_size=0.5, random_state=42)

        le = preprocessing.LabelEncoder()
        le.fit(data.iloc[:,1])
        X_train.iloc[:,1] = le.transform(X_train.iloc[:,1])
        X_test.iloc[:,1] = le.transform(X_test.iloc[:,1])

        print(X_train.head())

        model_name = "albert-base-v2"

        # max sequence length for each document/sentence sample
        max_length = 4

        tokenizer = AlbertTokenizerFast.from_pretrained(model_name)

        def preprocess_function(examples):
            return tokenizer(examples["content"], padding='max_length', truncation=True, max_length=max_length)

        dataset_train = Dataset.from_pandas(X_train)
        dataset_train = dataset_train.map(preprocess_function, batched=True)

        dataset_test = Dataset.from_pandas(X_test)
        dataset_test = dataset_test.map(preprocess_function, batched=True)

        begin = time.time()

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(list(le.classes_))).to("cuda")

        training_args = TrainingArguments(
            output_dir='./albert_results',          # output directory
            num_train_epochs=self.parameters['epochs'],              # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=batch_size,   # batch size for evaluation
            weight_decay=self.parameters['weight_decay'],
            learning_rate=self.parameters['learning_rate'],
            adam_beta1=self.parameters['adam_beta1'],
            adam_beta2=self.parameters['adam_beta2'],
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
            logging_steps=400,               # log & save weights each logging_steps
            save_steps=400,
            evaluation_strategy="steps",     # evaluate each `logging_steps`
        )

        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=dataset_train,         # training dataset
            eval_dataset=dataset_test,          # evaluation dataset
            compute_metrics=self.compute_metrics,     # the callback that computes metrics of interest
        )

        trainer.train()

        end = time.time()

        predictions = trainer.predict(dataset_test)
        preds = np.argmax(predictions.predictions, axis=-1)
        metric = load_metric("glue", "mrpc")
        res = metric.compute(predictions=preds, references=predictions.label_ids)

        # debug
        print('Albert done')

        return res, end - begin

if __name__ == "__main__":
    path = "/home/rushil/Desktop/Coding/Synapse/AutoNLP/datasets/sarcasm.csv"
    parameters = {'epochs': 3, 'weight_decay': 0.01, 'learning_rate': 2e-5, 'adam_beta1': 0.8, 'adam_beta2': 0.9}
    albert = Albert(path, parameters)
    score, time = albert.pipeline()
    print("Accuracy:", score)
    print("Time:", time)