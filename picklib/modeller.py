import pandas as pd
import numpy as np
import os
import glob
import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def custom_accuracy(prediction, true, moe=1):
    score1 =  np.mean([1 if abs(t - p) < t*0.15 else 0 for p, t in zip(prediction, true) if t >20]) * 100
    print(score1,len([1 if abs(t - p) < t*0.15 else 0 for p, t in zip(prediction, true) if t >20]) )
    
    score3 =  np.mean([1 if abs(t - p) <= moe+2 else 0 for p, t in zip(prediction, true) if t >10 and p <=20]) * 100
    print(score3,len([1 if abs(t - p) <= moe+2 else 0 for p, t in zip(prediction, true) if t >10 and p <=20]))
    
    score2 =  np.mean([1 if abs(t - p) < moe else 0 for p, t in zip(prediction, true) if t <=10]) * 100
    print(score2, len([1 if abs(t - p) < moe else 0 for p, t in zip(prediction, true) if t <=10]))
    return np.mean([score1,score2, score3])

def accuracy(prediction, true, moe=1):
    return np.mean(
        [1 if t * (1 - moe) <= p <= t * (1 + moe) else 0
         for p, t in zip(prediction, true)]) * 100

def accuracy_(prediction, true, moe=0.1):
    return np.mean([1 if abs(t - p) < moe else 0 for p, t in zip(prediction, true)]) * 100


def MAPE(prediction, true):
    return np.mean(np.abs((true - prediction) / true)) * 100


def MedAPE(prediction, true):
    return np.median(np.abs((true - prediction) / true)) * 100

def feature_importance(model, features) -> pd.DataFrame:
    fi = pd.DataFrame(list(zip(model.feature_importances_, features)), columns = ['score','feature'])
    fi.set_index('feature', inplace=True)
    fi.sort_values('score', inplace=True)
    return fi



class Modeller():
    def __init__(self, data):
        self.path_tocal_folder  = "data"
        self.data = data
        self.TARGET_COLUMN = "pick_diff_seconds"
        self.all_features = self.data.loc[:,self.data.columns != self.TARGET_COLUMN].columns
        
    def data_adapt(self):
        self.data['pick_start_date'] = self.data['pick_start_date'].dt.date
        self.data.drop(columns = 
                       ['pick_end_date', 'pick_diff', 'pick_diff_microseconds','pick_diff_intervals'], 
                       errors = "ignore",
                       inplace = True)
#         self.data['art'] = self.data.art_id.astype(object)
        self.data = pd.get_dummies(self.data.set_index('pick_start_date'))
        self.data.reset_index(inplace = True)
        

    def split_data(self):
        unique_dates = self.data.pick_start_date.dt.date.drop_duplicates().sort_values()
        n_train_days = int(len(unique_dates)*0.5)
        train_dates = unique_dates[:int(n_train_days*0.5)].tolist()
        test_dates = unique_dates[int(n_train_days*0.5):].tolist()
        self.data_adapt()
        self.train = self.data[self.data.pick_start_date.isin(train_dates)]
        self.test = self.data[self.data.pick_start_date.isin(test_dates)]
        self.test.drop(columns = 'pick_start_date',inplace = True)
        self.train.drop(columns = 'pick_start_date',inplace = True)
        self.all_features = self.train.loc[:,self.train.columns != self.TARGET_COLUMN].columns

    def init_regression_model(self):
        self.regression_model = RandomForestRegressor(
            n_jobs=-1,
            n_estimators=300,
#             min_samples_split=2,
#             max_features=0.5
        )

    def fit_model_predict_prices_(self):
        self.init_regression_model()
        self.regression_model.fit(
            X=self.train[self.all_features],
            y=np.log(self.train[self.TARGET_COLUMN ]))
        self.predicted_time = np.exp(
            self.regression_model.predict(
                self.test[self.all_features]
            )
        )

    def scores(self, accuracy_type = 'custom_accuracy'):
        self.accuracy = []
        for moe in [1,1.5,2]:
            predictions = self.predicted_time
            true = self.test[self.TARGET_COLUMN].values
            accuracy_score = globals()[accuracy_type](predictions, true, moe=moe)
            self.accuracy.append((moe, accuracy_score))
            print(f'accuracy at {moe}: {accuracy_score}')
        self.mape = MAPE(predictions, true)
        self.medape = MedAPE(predictions, true)

        print(f'Validation accuracy: {self.accuracy}')
        print(f'Validation MAPE: {self.mape}')
        print(f'Validation MedAPE: {self.medape}')