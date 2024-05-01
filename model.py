import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


## model
class MultiStack:
    def __init__(self, clfs=None, finalEstimator=None, nStack=0, nRepeat=1, cv=5) -> None:
        self.clfsList = []
        if finalEstimator is None:
            finalEstimator = LogisticRegression()
        for i in range(nStack+1):
            self.clfsList.append(clone(clfs))
        self.finalEstimator = finalEstimator
        self.nRepeat = nRepeat
        self.cv = cv

    def fit(self, X, y):
        y_m = None
        skf = StratifiedKFold(self.cv, shuffle=True, random_state=42)
        for clfs in self.clfsList:
            if y_m is not None:
                X = pd.DataFrame(np.hstack((X, y_m)))
            print('X.shape', X.shape)
            dataset_training = np.zeros((X.shape[0], len(clfs), self.nRepeat))
            for i in range(self.nRepeat):
                baggingData = list(skf.split(X, y))
                for train_index, test_index in baggingData:
                    X_train, y_train, X_test, y_test = X.iloc[train_index, :], y[train_index], X.iloc[test_index, :], y[test_index]
                    for m, clf in enumerate(clfs):
                        clf.fit(X_train, y_train)
                        y_sub = clf.predict_proba(X_test)[:, 1]
                        dataset_training[test_index, m, i] = y_sub
            y_m = dataset_training.mean(2)
            print('y_m.shape', y_m.shape)
    
        self.finalEstimator.fit(y_m, y)

    def predict(self, X):
        y_m = None
        for clfs in self.clfsList:
            if y_m is not None:
                X = np.hstack((X, y_m))
            y_m = np.zeros((X.shape[0], len(clfs)))
            for m, clf in enumerate(clfs):
                y_sub = clf.predict_proba(X)[:, 1]
                y_m[:, m] = y_sub
        return self.finalEstimator.predict(y_m)
    
    def predict_proba(self, X):
        y_m = None
        for clfs in self.clfsList:
            if y_m is not None:
                X = np.hstack((X, y_m))
            y_m = np.zeros((X.shape[0], len(clfs)))
            for m, clf in enumerate(clfs):
                y_sub = clf.predict_proba(X)[:, 1]
                y_m[:, m] = y_sub
        return self.finalEstimator.predict_proba(y_m)