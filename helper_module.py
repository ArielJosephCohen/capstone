import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats

def assign_random_seed(number):
    """
    Input a number for purposes of a random seed later to ensure all tests can be reproduced
    """
    seed = number

def print_number_of_unique(dataframe):
    """
    Input a data frame to see the amount of unique values for each feature
    """
    for col in dataframe.columns:
        print(col,dataframe[col].nunique())

def convert_to_date(column):
    """
    Input a column from a data frame to convert that column to a date-time format
    """
    df[column]=pd.to_datetime(df[column])

def plot_claim_vs_premium(dataframe,fig=(11,6)):
    """
    Input a data frame to see a scatter plot comparing premiums and claim amounts with the ability to customize figure size
    """
    plt.figure(figsize=fig)
    plt.tight_layout()
    return sns.scatterplot(dataframe.policy_annual_premium,dataframe.total_claim_amount,hue=dataframe.insured_sex)

def plot_fraud_proportion(dataframe):
    """
    Input a data frame and see the breakdown of the percent of fraudulent claims across all broad regions surveyed
    """
    print('All')
    dataframe.fraud_reported.value_counts().plot(kind='pie')
    plt.tight_layout()
    plt.show()
    print('Illinois')
    dataframe[dataframe.policy_state=='IL'].fraud_reported.value_counts().plot(kind='pie')
    plt.tight_layout()
    plt.show()
    print('Indiana')
    dataframe[dataframe.policy_state=='IN'].fraud_reported.value_counts().plot(kind='pie')
    plt.tight_layout()
    plt.show()
    print('Ohio')
    dataframe[dataframe.policy_state=='OH'].fraud_reported.value_counts().plot(kind='pie')
    plt.tight_layout()
    plt.show()
    
def show_fraud_ages(dataframe):
    """
    Input a data frame and see the breakdown of fraud committed by age
    """
    return dataframe[dataframe.fraud_reported=='Y'].age.plot(kind='hist')

def show_months_before_fraud(dataframe):
    """
    Input a data frame and see the histogram of fraud committed by month

    """
    dataframe[dataframe.fraud_reported=='Y'].months_as_customer.plot(kind='hist')
    
def show_incident_and_collision_type(dataframe,fig=(13,6)):
    """
    Input a data frame and see the breakdown of incident and collision types with the ability to customize figure size
    """
    incident = pd.crosstab(dataframe['incident_city'], dataframe['incident_type'])
    colors = plt.cm.Blues(np.linspace(0, 1, 5))
    incident.div(incident.sum(1).astype(float), axis = 0).plot(kind = 'bar',
                                                           stacked = False,
                                                           figsize = fig,
                                                           color = colors)

    plt.title('Incident Type vs Collision Type', fontsize = 20)
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def show_occupation_of_insured(dataframe,fig=(13,6)):
    """
    Input a data frame and see the breakdown of occupation of the insured people in this study with the ability to customize figure size
    """
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = fig

    sns.countplot(dataframe['insured_occupation'], palette = 'PuRd')
    plt.tight_layout()
    plt.title('Different Types of Occupation of Insured Customers', fontsize = 20)
    plt.xticks(rotation = 90)
    plt.show()
    
def show_hobbies_of_insured(dataframe,fig=(13,6)):
    """
    Input a data frame and see the breakdown of hobbies of the insured people in this study with the ability to customize figure size
    """
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = fig

    sns.countplot(dataframe['insured_hobbies'], palette = 'PuRd')
    plt.tight_layout()
    plt.title('Different Types of Hobbies of Insured Customers', fontsize = 20)
    plt.xticks(rotation = 90)
    plt.show()
    
def show_types_of_incidents(dataframe,fig=(13,6)):
    """
    Input a data frame and see the breakdown of types of incidents with the ability to customize figure size
    """
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = fig

    sns.countplot(dataframe['incident_type'], palette = 'spring')
    plt.tight_layout()
    plt.title('Different Types of Incidents', fontsize = 20)
    plt.show()
    
def show_gender_fraud_breakdown(dataframe,fig=(4,4)):
    """
    Input a data frame and see the breakdown of fraud by gender with ability to customize figure size
    """
    plt.figure(figsize=fig)
    plt.tight_layout()
    dataframe[dataframe.fraud_reported=='Y'].insured_sex.value_counts(normalize=True).plot(kind='bar')
    
def create_auto_dict():
    """
    Create a dictionary with a car type assigned to each unique car model that appears
    """
    auto_dict = {'RAM':'Truck','Wrangler':'SUV','Neon':'Sedan','A3':'Sedan','MDX':'SUV','Jetta':'Sedan',
                   'Passat':'Sedan','A5':'Sedan', 'legacy':'Sedan','Pathfinder':'SUV','Malibu':'Sedan',
                   'Camry':'Sedan','Forrester':'SUV','92x':'Sedan','95':'Sedan','E400':'Sedan','F150':'Truck',
                   'Grand Cherokee':'SUV','93':'Sedan','Tahoe':'SUV','Escape':'SUV','Maxima':'Sedan','X5':'SUV',
                   'Ultima':'Sedan','Civic':'Sedan','Highlander':'SUV','Silverado':'Truck','Fusion':'Sedan',
                   'ML350':'SUV','Corolla':'Sedan','TL':'Sedan','CRV':'SUV','Impreza':'Sedan','3 Series':'Sedan',
                   'C300':'Sedan','X6':'SUV','M5':'Sedan','Accord':'Sedan','RSX':'Sedan','Legacy':'Sedan',
                   'Forrestor':'SUV'
                  }
    return auto_dict

def assign_car_type(dataframe):
    """
    Input a data frame to map the car type dictionary to each occurance of a car model
    """
    dataframe.auto_model=dataframe.auto_model.map(lambda x: auto_model_dict[x])
    return dataframe

def create_num_list():
    """
    Create an initial list of numerical features
    """
    num_list = ['months_as_customer','age','policy_deductable','policy_annual_premium','umbrella_limit','capital-gains','capital-loss','incident_hour_of_the_day','number_of_vehicles_involved','witnesses','bodily_injuries','total_claim_amount','injury_claim','property_claim','vehicle_claim','auto_year']
    return num_list

def create_cat_list():
    """
    Create an initial list of categorical features
    """
    cat_list =['policy_state','policy_csl','insured_sex','insured_education_level','insured_occupation','insured_hobbies','insured_relationship','incident_type','collision_type','incident_severity','authorities_contacted','incident_state','incident_city','property_damage','police_report_available','auto_make','auto_model','fraud_reported','policy_bind_year','policy_bind_month']
    return cat_list

def create_cat_list_2():
    """
    Create an initial list of categorical features
    """
    cat_list_2 =['policy_state','policy_csl','insured_sex','insured_education_level','insured_occupation','insured_hobbies','insured_relationship','incident_type','collision_type','incident_severity','authorities_contacted','incident_state','incident_city','property_damage','police_report_available','auto_make','auto_model','policy_bind_year','policy_bind_month']
    return cat_list_2


def assign_df_categorical_split(dataframe):
    """
    Input a data frame to be split into numerical and categorical data frames
    """
    df_num = dataframe[num_list]
    df_cat = dataframe[cat_list]

def add_timeline(dataframe):
    """
    Input a data frame and use date and time information to track days after policy began until accident happened
    """
    dataframe['timeline']=dataframe.incident_date-dataframe.policy_bind_date
    num_list.append('timeline')
    for i in range(len(df)):
        df.timeline[i] = int(str(df.timeline[i]).split()[0])
    return timeline

def plot_numerical_feature_correlation(dataframe,fig=(15,8),threshold=0.7):
    """
    Input a numerical data drame and measure amount of collinearity among features with the ability to define a correlation threshold and figure size of graph
    """
    plt.figure(figsize=fig)
    plt.tight_layout()
    return sns.heatmap(dataframe[num_list].corr()>=threshold)

def remove_correlation(dataframe,lst):
    """
    Input a data frame to remove highly-correlated variables and update the list of variables
    """
    dataframe.drop(['age','total_claim_amount','vehicle_claim'],axis=1,inplace=True)
    lst.remove('age')
    lst.remove('total_claim_amount')
    lst.remove('vehicle_claim')
    lst.append('timeline')
    return dataframe,lst

def drop_cat(lst):
    """
    After adding dummy variables drop a list of the initial columns that are no longer needed
    """
    return df_cat.drop(lst,axis=1,inplace=True)

def reassign_year_and_month(dataframe):
    """
    Input a data frame and create policy_bind_year, policy_bind_month, incident_year, and incident_month to then be covnverted into dummy variables
    """
    dataframe['policy_bind_month']=0
    dataframe['policy_bind_year']=0
    for i in range(len(dataframe)):
        dataframe['policy_bind_month'][i]=int(str(dataframe.policy_bind_date[i]).split()[0][3:5])
        dataframe['policy_bind_year'][i]= int(str(dataframe.policy_bind_date[i]).split()[0][6:10])
    return dataframe

def filter_outliers(dataframe,threshold=2.5):
    """
    Input a data frame and remove outliers using a z-score to pick a number of standard deviations captured with the ability to customize a threshold of standard deviations captured
    """
    dataframe = dataframe[(np.abs(stats.zscore(dataframe[num_list])) <= threshold).all(axis=1)]
    print(dataframe.shape)
    return dataframe

def split_data(dataframe,t_s=0.25):
    """
    Input a data frame, assign X and y for features and target, and split into two groups of data with ability to customize amount of data assigned to test set
    """
    X = dataframe.drop('fraud_reported',axis=1)
    y = dataframe.fraud_reported
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=seed,test_size=t_s)
    print('Do shapes match?')
    print(X_test.shape[0]==y_test.shape[0])
    print(X_train.shape[0]==y_train.shape[0])
    return (X_train, X_test, y_train, y_test)

def show_class_balance(data):
    """
    Input data and see the relative difference in size between the target features of the data    
    """
    plt.tight_layout()
    return(round(data.fraud_reported.value_counts(normalize=True),2).plot(kind='bar',color='limegreen'))

def upsample_data(X_tr,y_tr):
    """
    Input X and y training data from an unbalanced data set to upsample the minority class for a more meaningful model
    """
    from sklearn.utils import resample
    training  = pd.concat([X_tr, y_tr], axis=1)
    true = training[training.fraud_reported==0]
    fraud = training[training.fraud_reported==1]
    fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(true), # match number in majority class
                          random_state=seed) # reproducible results
    upsampled = pd.concat([true, fraud_upsampled])
    plt.tight_layout()
    print(round(upsampled.fraud_reported.value_counts(normalize=True),2).plot(kind='bar',color='limegreen'))
    y_tr = upsampled.fraud_reported
    X_tr = upsampled.drop('fraud_reported', axis=1)
    return (X_tr,y_tr)

def show_categorical_breakdown(dataframe):
    cat_cols = []
    cat_col_vals = []
    for col in cat_list[1:]:
        cat_cols.append(col) 
        cat_col_vals.append(dataframe[col].nunique())
    print (col,df[col].nunique())
    plt.tight_layout()
    plt.figure(figsize=(15,8))
    return plt.barh(cat_cols,cat_col_vals)

def create_encoding(column,subset,dataframe):
    """
    Input a feature from a categorical data frame to encode numerically
    """
    column_dict={}
    dummy_df = dataframe[[f'{column}','fraud_reported']].groupby([f'{column}'], 
    as_index = False).mean().sort_values(by = 'fraud_reported', ascending = False)
    for i in range(len(dummy_df)):
        column_dict[dummy_df.iloc[i][0]]=(1-dummy_df.iloc[i][1])
    subset[column] = dataframe[column].map(lambda x: column_dict[x])
    return subset

def remove_categorical_correlation(dataframe,column_lst):
    """
    Input a list of features to drop after categorical encoding
    """
    dataframe.drop(column_lst,axis=1,inplace=True)
    return dataframe

def combine_data_frames(df1,df2):
    """
    Input two data frames to be merged
    """
    new_df = df1.copy()
    for col in df2.columns:
        new_df[col]=df2[col]
    return new_df
        
def clean_data(dataframe):
    """
    Input a data frame and have all the problematic values filled
    """    
    dataframe=dataframe.replace('?',np.NaN)
    dataframe['collision_type'].fillna(dataframe['collision_type'].mode()[0], inplace = True)
    dataframe['property_damage'].fillna('NO', inplace = True)
    dataframe['police_report_available'].fillna('NO', inplace = True)
    return dataframe
    
def map_binary_dict(dataframe,feature,value_1,value_2):
    """
    Input a binary feature and have value 1 mapped as a 1 and value 2 mapped as a zero
    """
    binary_dict = {value_1:1,value_2:0}
    dataframe[feature]=dataframe[feature].map(lambda x: binary_dict[x])
    return dataframe

def create_timeline(dataframe):
    """
    Input a data frame to establish a timeline from policy bind date to claims
    """
    dataframe.policy_bind_date=pd.to_datetime(dataframe.policy_bind_date)
    dataframe.incident_date=pd.to_datetime(dataframe.incident_date)
    dataframe['timeline']=dataframe.incident_date-dataframe.policy_bind_date
    for i in range(len(dataframe)):
        dataframe.timeline[i] = int(str(dataframe.timeline[i]).split()[0])
    dataframe.timeline=dataframe.timeline.astype(int)
    return dataframe

def quantify_absolute_value(dataframe,feature):
    """
    Input a feature from a data frame to turn to absolute value
    """ 
    dataframe[feature]=np.abs(dataframe[feature])
    return dataframe

def assign_new_dataframe(dataframe,lst,drop_lst=None):
    """
    Input a data frame and create a subset data frame
    """
    lst=lst.remove(drop_lst)
    new_dataframe=dataframe[lst]
    return new_dataframe

def reduce_features(dataframe,seed):
    """
    Input a data frame and reduce un-needed features
    """
    from sklearn.feature_selection import rfe
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold
    X = dataframe.drop('fraud_reported',axis=1)
    y = dataframe.fraud_reported
    rfc = RandomForestClassifier(random_state=seed)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(8), scoring='accuracy',min_features_to_select=12)
    rfecv.fit(X, y)
    support_list = list(rfecv.support_)
    importance = []
    for val, sup in list(zip(X.columns,support_list)):
        if sup == True:
            importance.append(val)
    xydf = pd.concat([X[importance],dataframe.fraud_reported],axis=1)
    return xydf

def filter_outliers(dataframe):
    from scipy import stats
    dataframe = dataframe[(np.abs(stats.zscore(dataframe)) <= 2.5).all(axis=1)]
    return dataframe

def normalize_features(dataframe):
    for col in dataframe.columns:
        dataframe[col]=list(stats.boxcox(abs(dataframe[col]+0.01)))[0]
    return dataframe
    
def scale_data(dataframe):
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for col in dataframe.columns:
        if (dataframe[col]>=1).sum() >0:
            dataframe[col] = scaler.fit_transform(dataframe[[col]])
    return dataframe