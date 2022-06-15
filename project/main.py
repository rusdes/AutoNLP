import os
import pandas as pd

def initialize_df():
    scores = pd.DataFrame({'NB': pd.Series(dtype='float'),
                            'Bert_1': pd.Series(dtype='float'),
                            'Albert_1': pd.Series(dtype='float'),
                            'XLNet_1': pd.Series(dtype='float'),
                            })

    time = pd.DataFrame({'NB': pd.Series(dtype='float'),
                            'Bert_1': pd.Series(dtype='float'),
                            'Albert_1': pd.Series(dtype='float'),
                            'XLNet_1': pd.Series(dtype='float'),
                            })

    return scores, time

if __name__ == "__main__":
    scores, times = initialize_df()

    datasets_dir = "datasets"

    for filename in os.scandir(datasets_dir):
        score = []
        time = []



        score = pd.DataFrame([score], columns = ["NB", "Bert_1", "Albert_1", "XLNet_1"])
        time = pd.DataFrame([time], columns = ["NB", "Bert_1", "Albert_1", "XLNet_1"])
        scores = scores.append(score, ignore_index = True)
        times = times.append(time, ignore_index = True)