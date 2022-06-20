import pandas as pd
from naive_bayes import NB
from BERT import BERT
from Albert import Albert
from xlnet import XLNET

from os import path
import glob


def initialize_df():
    if path.exists('accuracy.csv'):
        scores = pd.read_csv('accuracy.csv', index_col=[0])
    else:
        scores = pd.DataFrame({ 'Dataset': pd.Series(dtype='str'),
                                'NB': pd.Series(dtype='float'),
                                'Bert_1': pd.Series(dtype='float'),
                                'Bert_2': pd.Series(dtype='float'),
                                'Albert_1': pd.Series(dtype='float'),
                                'Albert_2': pd.Series(dtype='float'),
                                'XLNet_1': pd.Series(dtype='float'),
                                'XLNet_2': pd.Series(dtype='float'),
                                })

    if path.exists('training_time.csv'):
        time = pd.read_csv('training_time.csv', index_col=[0])
    else:
        time = pd.DataFrame({'Dataset': pd.Series(dtype='str'),
                                'NB': pd.Series(dtype='float'),
                                'Bert_1': pd.Series(dtype='float'),
                                'Bert_2': pd.Series(dtype='float'),
                                'Albert_1': pd.Series(dtype='float'),
                                'Albert_2': pd.Series(dtype='float'),
                                'XLNet_1': pd.Series(dtype='float'),
                                'XLNet_2': pd.Series(dtype='float'),
                                })
    return scores, time

if __name__ == "__main__":
    scores, times = initialize_df()

    datasets_dir = "datasets/*.csv"

    for filename in glob.glob(datasets_dir):
        print(filename[9:-4])
        score = []
        time = []

        score.append(filename[9:-4])
        time.append(filename[9:-4])

        # Naive Bayes
        model = NB(filename)
        a, t = model.pipeline()
        score.append(a)
        time.append(t)
        
        # BERT
        # model 7, 23, 13
        # skipped 23 since parameters were too similar. Dont mention skipping it in paper.
        parameters = [
            {'epochs': 5, 'warmup_steps': 0, 'learning_rate': 2e-5, 'adam_beta1': 0.9, 'adam_beta2': 0.999},
            {'epochs': 5, 'warmup_steps': 0, 'learning_rate': 1e-5, 'adam_beta1': 0.9, 'adam_beta2': 0.9},
        ]
        for p in parameters:
            model = BERT(filename, p)
            a, t = model.pipeline()
            score.append(a)
            time.append(t)
        
        # Albert
        # model 35, 24, 47
        parameters = [
            {'epochs': 5, 'warmup_steps': 1, 'learning_rate': 2e-5, 'adam_beta1': 0.8, 'adam_beta2': 0.9},
            {'epochs': 3, 'warmup_steps': 0, 'learning_rate': 1e-5, 'adam_beta1': 0.9, 'adam_beta2': 0.9},
        ]
        for p in parameters:
            model = Albert(filename, p)
            a, t = model.pipeline()
            score.append(a)
            time.append(t)

        # XLNet
        # model 24, 18, 2
        parameters = [
            {'epochs': 3, 'warmup_steps': 1, 'learning_rate': 1e-5, 'adam_beta1': 0.8, 'adam_beta2': 0.9},
            {'epochs': 3, 'warmup_steps': 1, 'learning_rate': 2e-5, 'adam_beta1': 0.8, 'adam_beta2': 0.999},
        ]
        for p in parameters:
            model = XLNET(filename, p)
            a, t = model.pipeline()
            score.append(a)
            time.append(t)

        score_df = pd.DataFrame([score], columns = ["Dataset", "NB", "Bert_1", "Bert_2", "Albert_1", "Albert_2","XLNet_1", "XLNet_2"])
        time_df = pd.DataFrame([time], columns = ["Dataset", "NB", "Bert_1", "Bert_2", "Albert_1", "Albert_2","XLNet_1", "XLNet_2"])
        scores = scores.append(score_df, ignore_index = True)
        times = times.append(time_df, ignore_index = True)

    scores.to_csv('accuracy.csv', index=False)
    times.to_csv('training_time.csv', index=False)