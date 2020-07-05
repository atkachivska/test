import pandas as pd
import numpy as np
import os
import glob
import datetime
from pyod.models.cblof import CBLOF


class DataHandler():
    def __init__(self):
        self.path_tocal_folder  = "data"
        self.path_local_data = "data2.csv"
        
    @staticmethod
    def str_to_datetime(date_string):
        if pd.isna(date_string) == False:
            y,m,d,t = date_string.split("-")
            try:
                h,mm,s,ms = t.split(".")
            except: 
                h,mm,s,ms = [0,0,0,0]
            return datetime.datetime(year = int(y), 
                                     month = int(m), 
                                     day = int(d), 
                                     hour = int(h), 
                                     minute = int(mm), 
                                     second = int(s),
                                     microsecond = int(ms))
        else:
            return None
        
    @staticmethod
    def to_datetime(df, columns = ['pick_start_date','pick_end_date']):
        for col in columns:
            df[col] = df[col].apply(datetime.datetime.fromisoformat)
        return df
        
    def get_data_filenames(self):
        self.data_path = os.path.join(os.getcwd(), self.path_tocal_folder)
        self.filenames = glob.glob(f'{self.data_path}/*.csv')
        
    def clean_data(self, df, drop = True):
        df.loc[:,"pick_start_date"] = df.pick_start_date.apply(self.str_to_datetime)
        df.loc[:, "pick_end_date"] = df.pick_end_date.apply(self.str_to_datetime)

#         print("pick_start_date na values amount:")
#         print(df.pick_start_date.isna().value_counts())
#         print("pick_end_date na values amount:")
#         print(df.pick_end_date.isna().value_counts())
        
        if drop:
            df.dropna(subset = ["pick_start_date","pick_end_date"], inplace = True)
            df = df[df.pick_start_date.apply(lambda x: x.year > 2000)]
            df = df[df.pick_end_date.apply(lambda x: x.year > 2000)]
        
        return df
    
    @staticmethod
    def feature_generate(df):
        df.loc[:,'pick_diff'] = df[['pick_start_date','pick_end_date']].apply(lambda x: x[1]- x[0] if any(pd.isna(x)==False) else None, axis = 1)
        df.loc[:,"pick_diff_microseconds"] = df.pick_diff.dt.microseconds
        df.loc[:,"pick_diff_seconds"] = df.pick_diff.dt.seconds
        df.loc[:,"pick_diff_seconds"] = df.loc[:,["pick_diff_microseconds","pick_diff_seconds"]].apply(lambda x: x[0]/1e+6+x[1], axis = 1)
        return df
    
    @staticmethod
    def select_df(df):
        df = df[df.pick_diff.dt.days == 0]
        return df
        
    def read_data(self, read = False):
        if read:
            self.get_data_filenames()
            dfs = []
            for filename in self.filenames:
                print(f"filename reading - {filename.split('/')[-1]}")
                df = pd.read_csv(filename, sep = ";")
                df = self.clean_data(df)
                df =  self.feature_generate(df)
                df =  self.select_df(df)
                dfs.append(df)
            self.data = pd.concat([df for df in dfs])
            self.data.drop(columns = 'Unnamed: 11', errors = "ignore", inplace = True)
            self.data.reset_index(inplace = True, drop = True)
            self.data.to_csv(os.path.join(os.getcwd(),"data2.csv"))
        else:
            self.data = pd.read_csv(os.path.join(os.getcwd(), self.path_local_data))
            self.data.drop(columns = "Unnamed: 0", inplace = True)
            self.data.pick_start_date = self.data.pick_start_date.apply(datetime.datetime.fromisoformat)
            self.data.pick_end_date = self.data.pick_end_date.apply(datetime.datetime.fromisoformat)
        print(f"Data row amount: {self.data.shape[0]}")
                      
                      
    def get_ids_amount(self, order_threshold = 1, art_threshold = 50, tour_threshold = 2, delete = True):
        self.data = pd.merge(  self.data, 
                             self.data.los_order_id.value_counts().reset_index(name = "pick_amount_in_order").rename(columns = {"index":"los_order_id"}),
                             on = "los_order_id")


        self.data = pd.merge(  self.data, 
                             self.data.art_id.value_counts().reset_index(name = "art_amount").rename(columns = {"index":"art_id"}),
                             on = "art_id")

        self.data = pd.merge(  self.data, 
                             self.data.tour_id.value_counts().reset_index(name = "tour_amount").rename(columns = {"index":"tour_id"}),
                             on = "tour_id")

        if delete:
            #set -1 value for orders with one pick
            self.data.loc[self.data.pick_amount_in_order ==  order_threshold ,"los_order_id"] = -1
            self.data.loc[self.data.tour_amount <= tour_threshold,"tour_id"] = -1
            self.data.loc[self.data.art_amount <= art_threshold  ,"art_id"] = -1
                      
                      
    def feature_generating(self):
        #delete unique id
        self.get_ids_amount()

#         #Count serial pick number in order
#         self.data.sort_values(["los_order_id","pick_start_date"], inplace = True)
#         self.data.loc[:, "pick_id_serial_number"] = self.data.groupby("los_order_id").los_pick_id.apply(lambda x: 
#                                                                                                     list(range(1, len(x)+1))).sum()

#         self.data.loc[self.data.los_order_id == -1,"pick_id_serial_number"] = -1

        #dencity count
        self.data.loc[:,"volume_weight_product"] = self.data[["volume","weight"]].apply(lambda x: np.round(x[0]*x[1]), axis = 1)

        
    def outliers_detect(self, columns,outliers_fraction = 0.05):
        X = pd.get_dummies(self.data[columns])
        clf = CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=0)
        clf.fit(X)
        scores_pred = clf.decision_function(X) * -1
        y_pred = clf.predict(X)
        self.data['outlier'] = y_pred.tolist()
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)        