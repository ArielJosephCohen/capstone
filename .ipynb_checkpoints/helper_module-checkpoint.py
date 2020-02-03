import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats

def print_number_of_unique(dataframe):
    for col in dataframe.columns:
        print(col,dataframe[col].nunique())

def convert_to_date(column):
    df[column]=pd.to_datetime(df[column])
    
def gender_val(row):
    if row['insured_sex'] == 'MALE':
        return 1
    else:
        return 0

def map_gender(dataframe):
    return dataframe['insured_sex']=df.apply(gender_val,axis=1)
    
def add_timeline(dataframe):
    dataframe['timeline']=dataframe.incident_date-dataframe.policy_bind_date
    num_list.append('timeline')
    for i in range(len(df)):
        df.timeline[i] = int(str(df.timeline[i]).split()[0])
    return timeline
    
def show_high_correlation(dataframe):
    plt.figure(figsize=(15,8))
    plt.tight_layout()
    return sns.heatmap(dataframe[num_list].corr()>=0.7)
    
def remove_correlation(dataframe):
    dataframe.drop(['age','total_claim_amount','vehicle_claim'],axis=1,inplace=True)
    num_list.remove('age')
    num_list.remove('total_claim_amount')
    num_list.remove('vehicle_claim')
    
def assign_df_split():
    df_num = df[num_list]
    df_cat = df[cat_list]
    
def convert_cal_df(df_list):
    for dfr in df_list:
        for col in dfr.columns:
            df_cat[col]=dfr[col]
    return df_cat

def drop_cat(lst):
    return df_cat.drop(lst,axis=1,inplace=True)

def scale_and_transform(dataframe):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for col in dataframe.columns:
        dataframe[col] = scaler.fit_transform(dataframe[[col]])
    for col in dataframe.columns:
        dataframe[col]=list(stats.boxcox(abs(dataframe[col]+0.5)))[0]
    return dataframe

def merge_numerical_and_categorical(dataframe1,dataframe2):
    df_atg = dataframe1.copy()
    for col in dataframe2.columns:
        df_atg[col]=dataframe2[col]
    return df_atg

def filter_outliers(dataframe):
    dataframe = dataframe[(np.abs(stats.zscore(dataframe[num_list])) <= 2.5).all(axis=1)]
    print(dataframe.shape)
    return dataframe