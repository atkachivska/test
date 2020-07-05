import pandas as pd
import numpy as np
import os
import glob
import datetime
import matplotlib.pyplot as plt


class DataAnalysis():
    def __init__(self,df):
        self.df = df
        self.target_col = "pick_diff_seconds"
        self.top_values = 10
        
    def get_corr(self, columns = ["pick_diff_seconds","to_pick","weight","volume","tour_id","art_id","box_position_on_cart","box_type"]):
        df = pd.get_dummies(self.df[columns])
        return df.corr()[[self.target_col]].apply(abs).sort_values(by = self.target_col, ascending = False)    

    def hist_plot_groupby(self, group_col, mean = True, median = False):
        if mean:
            print(f"Mean hist plot grouped by: {group_col}")
            data = self.df.groupby(group_col)[self.target_col].mean().sort_index(ascending = True)
            data = pd.concat([data.head(self.top_values), data.tail(self.top_values)]).drop_duplicates()
            return data.plot.bar()
        if median:
            print(f"Median hist plot grouped by: {group_col}")
            data = self.df.groupby(group_col)[self.target_col].median().sort_index(ascending = True)
            data = pd.concat([data.head(self.top_values), data.tail(self.top_values)]).drop_duplicates()
            return data.plot.bar()
        
    @staticmethod  
    def pie_chart_with_groups (dataset, categories, segments, title = ""):
        dataset = dataset[dataset[segments].isna() == False]
        for i in tuple(dataset[segments].unique()):
           # for k in ["New","Regular"]:

            df = dataset[(dataset[segments] == i) & (dataset[segments].isna()==False)]
        # Draw Plot
            fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi= 80)

            data = df[df[categories].isna()==False].groupby(categories).size().reset_index(name = "amount").amount
    #         cats = df.loc[df[categories].isna()==False,categories]
            cats = list(set(df[categories]))
            #explode = [0,0,0,0,0,0.1,0]

            def func(pct, allvals):
                return "{:.1f}% )".format(pct)

            wedges, texts, autotexts = ax.pie(data,radius=1.2, 
                                              autopct=lambda pct: func(pct, data),
                                              textprops=dict(color="w"), 
                                              colors=plt.cm.Dark2.colors,
                                             startangle=230)

            # Decoration
            ax.legend(wedges,cats , title=categories, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),fontsize =17)
            plt.setp(autotexts, size=12, weight=50)
            ax.set_title(title+str(i), size = 15)
            plt.xlabel("amount - " + str(data.sum()),size = 15)
            plt.show()
    @staticmethod    
    def pie_chart (dataset, categories, title):
        #dataset = dataset[dataset[segments].isna() == False]

        df = dataset[dataset[categories].isna()==False]
    # Draw Plot
        fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi= 80)

        data = dataset[dataset[categories].isna()==False].groupby(categories).size().reset_index(name = "amount").sort_values("amount", ascending = False)
        cats = data[categories]
        data = data["amount"]

        #explode = [0,0,0,0,0,0.1,0]

        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}% ({:d} )".format(pct, absolute)

        wedges, texts, autotexts = ax.pie(data,radius=1.1, 
                                          autopct=lambda pct: func(pct, data),
                                          textprops=dict(color="w", fontsize = 40), 
                                          colors=plt.cm.Dark2.colors,
                                         startangle=250)

        # Decoration

        ax.legend(wedges,cats, title=categories, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),fontsize =12)
        plt.setp(autotexts, size=12, weight=50)
        ax.set_title(title, size = 15)
        plt.xlabel("Всего - " + str(data.sum()),size = 15)
        plt.show()        
    @staticmethod
    def compute_histogram_bins(data, desired_bin_size):
        min_val = np.min(data)
        max_val = np.max(data)
        min_boundary = min_val#-1.0 * (min_val % desired_bin_size - min_val)
        max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
        n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
        bins = np.arange(min_boundary, max_boundary, n_bins)
        return bins

    @staticmethod
    def hist_plot(df, column,  desired_bin_size):
        if desired_bin_size != 0:
            b = compute_histogram_bins(df.loc[df[column].isna() ==False,column],1)

            plt.hist(df.loc[df[column].isna() ==False,column], 
                 bins = b)
        else :
                plt.hist(df.loc[df[column].isna() ==False,column])
                plt.ylabel('Amount')
                plt.xticks(rotation='vertical')
                plt.xlabel(column)