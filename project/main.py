import pandas as pd
# from naive_bayes import NB
# from BERT import BERT
# from Albert import Albert
# from xlnet import XLNET

from models.BERT import BERT
from models.Albert import Albert
from models.xlnet import XLNET
from models.naive_bayes import NB

from os import path
import glob


def initialize_df():
    if path.exists('accuracy.csv'):
        scores = pd.read_csv('accuracy.csv', index_col=[0])
    else:
        scores = pd.DataFrame({ 'Dataset': pd.Series(dtype='str'),
                                'NB': pd.Series(dtype='float'),
                                'Bert_1': pd.Series(dtype='float'),
                                'Albert_1': pd.Series(dtype='float'),
                                'XLNet_1': pd.Series(dtype='float'),
                                })

    if path.exists('training_time.csv'):
        time = pd.read_csv('training_time.csv', index_col=[0])
    else:
        time = pd.DataFrame({'Dataset': pd.Series(dtype='str'),
                                'NB': pd.Series(dtype='float'),
                                'Bert_1': pd.Series(dtype='float'),
                                'Albert_1': pd.Series(dtype='float'),
                                'XLNet_1': pd.Series(dtype='float'),
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
        parameters = [
            {'epochs': 3, 'weight_decay': 0.01, 'learning_rate': 2e-5, 'adam_beta1': 0.8, 'adam_beta2': 0.9}
        ]
        for p in parameters:
            # model = BERT(filename, p)
            a, t = model.pipeline()
            score.append(a)
            time.append(t)
        
        # Albert
        parameters = [
            {'epochs': 3, 'weight_decay': 0.01, 'learning_rate': 2e-5, 'adam_beta1': 0.8, 'adam_beta2': 0.9}
        ]
        for p in parameters:
            # model = Albert(filename, p)
            a, t = model.pipeline()
            score.append(a)
            time.append(t)

        # XLNet
        parameters = [
            {'epochs': 3, 'weight_decay': 0.01, 'learning_rate': 2e-5, 'adam_beta1': 0.8, 'adam_beta2': 0.9}
        ]
        for p in parameters:
            # model = XLNET(filename, p)
            a, t = model.pipeline()
            score.append(a)
            time.append(t)

        score = pd.DataFrame([score], columns = ["Dataset", "NB", "Bert_1", "Albert_1", "XLNet_1"])
        time = pd.DataFrame([time], columns = ["Dataset", "NB", "Bert_1", "Albert_1", "XLNet_1"])
        scores = scores.append(score, ignore_index = True)
        times = times.append(time, ignore_index = True)
        print(score.head())

    score.to_csv('accuracy.csv', index=False)
    times.to_csv('training_time.csv', index=False)