import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import time

class NB():
    def __init__(self, path):
        self.path = path

    def pipeline(self):
        data = pd.read_csv(self.path)
        data = data.iloc[:,-2:]

        if len(data) > 30000:
            data = data[:30000]

        data.rename(columns = {list(data)[0]:'content', list(data)[1]:'label'}, inplace=True)

        data['content'] = data['content'].str.replace('\d+', '')
        data.dropna(inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,0], data.iloc[:,1], test_size=0.2, random_state=42)

        start = time.time()
        le = preprocessing.LabelEncoder()
        le.fit(data.iloc[:,1])
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

        # We can use Pipeline to add vectorizer -> transformer -> classifier all in a one compound classifier
        text_clf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])
        # Fitting our train data to the pipeline
        text_clf.fit(X_train, y_train)
        end = time.time()
        # Predicting our test data
        predicted = text_clf.predict(X_test)
        t = end - start
        # debug
        print('NB done')
        return np.mean(predicted == y_test)*100, t

if __name__ == "__main__":
    path = "//home/rushil/Desktop/Coding/Synapse/AutoNLP/datasets/SMS.csv"
    NB = NB(path)
    accuracy, time = NB.pipeline()

    print("Accuracy:", accuracy, "Time:", time)