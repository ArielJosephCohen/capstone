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
    
def gender_val(row):
    """
    Input a row from a data frame to convert to a numerical metric for gender
    """
    if row['insured_sex'] == 'MALE':
        return 1
    else:
        return 0

def map_gender(dataframe):
    """
    Input a data frame to assign the gender column in said data frame as a numerical feature
    """
    dataframe['insured_sex']=df.apply(gender_val,axis=1)
    return dataframe

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
    cat_list =['policy_bind_date','policy_state','policy_csl','insured_sex','insured_education_level','insured_occupation','insured_hobbies','insured_relationship','incident_date','incident_type','collision_type','incident_severity','authorities_contacted','incident_state','incident_city','property_damage','police_report_available','auto_make','auto_model','fraud_reported']
    return cat_list

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

def remove_correlation(dataframe):
    """
    Input a data frame and remove highly-correlated variables
    """
    dataframe.drop(['age','total_claim_amount','vehicle_claim'],axis=1,inplace=True)
    num_list.remove('age')
    num_list.remove('total_claim_amount')
    num_list.remove('vehicle_claim')
    
def convert_cal_df(df_list):
    """
    Input a list of data frames containing dummy variables and bring them into the categorical dataframe as binary otuputs
    """
    for dfr in df_list:
        for col in dfr.columns:
            df_cat[col]=dfr[col]
    return df_cat

def drop_cat(lst):
    """
    After adding dummy variables drop a list of the initial columns that are no longer needed
    """
    return df_cat.drop(lst,axis=1,inplace=True)


def scale_and_transform(dataframe,extra_number=0.5):
    """
    Input a dataframe and apply min-max scaling followed by feature normalization on its features. Use the extra_number parameter for more customization on normalization.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for col in dataframe.columns:
        dataframe[col] = scaler.fit_transform(dataframe[[col]])
    for col in dataframe.columns:
        dataframe[col]=list(stats.boxcox(abs(dataframe[col]+extra_number)))[0]
    return dataframe

def merge_numerical_and_categorical(dataframe1,dataframe2):
    """
    Innput two data frames to combine after addressing numerical and categorical features separately
    """
    df_atg = dataframe1.copy()
    for col in dataframe2.columns:
        df_atg[col]=dataframe2[col]
    return df_atg

def reassign_year_and_month(dataframe):
    """
    Input a data frame and create policy_bind_year, policy_bind_month, incident_year, and incident_month to then be covnverted into dummy variables
    """
    dataframe['policy_bind_month']=0
    dataframe['policy_bind_year']=0
    dataframe['incident_month']=0
    dataframe['incident_year']=0
    for i in range(len(df_atg)):
        dataframe['policy_bind_month'][i]=int(str(dataframe.incident_date[i]).split()[0][5:7])
        dataframe['policy_bind_year'][i]= int(str(dataframe.incident_date[i]).split()[0][0:4])
        dataframe['incident_month'][i]= int(str(dataframe.policy_bind_date[i]).split()[0][5:7])
        dataframe['incident_year'][i]= int(str(dataframe.policy_bind_date[i]).split()[0][0:4])
    dataframe.drop(['incident_date','policy_bind_date'],axis=1,inplace=True)
    i_month_df=pd.get_dummies(dataframe.incident_month,prefix='in-month',drop_first=True)
    i_year_df=pd.get_dummies(dataframe.incident_year,prefix='in-year',drop_first=True)
    p_month_df=pd.get_dummies(dataframe.policy_bind_month,prefix='pol-month',drop_first=True)
    p_year_df=pd.get_dummies(dataframe.policy_bind_year,prefix='pol-year',drop_first=True)
    moyear_lst=[p_month_df,p_year_df,i_month_df,i_month_df]
    for dfra in moyear_lst:
        for col in dfra.columns:
            dataframe[col]=dataframe[col]
    return dataframes

def filter_outliers(dataframe,threshold=2.5):
    """
    Input a data frame and remove outliers using a z-score to pick a number of standard deviations captured with the ability to customize a threshold of standard deviations captured
    """
    dataframe = dataframe[(np.abs(stats.zscore(dataframe[num_list])) <= threshold).all(axis=1)]
    print(dataframe.shape)
    return dataframe

def assign_fraud_binary(row):
    """
    Input a row and convert pressence of fraud to numerica; feature
    """
    if row['fraud_reported'] == 'Y':
        return 1
    else:
        return 0
    
def map_fraud_binary(dataframe):
    """
    Input a data frame and convert all instances of fraud to numerical binaries
    """
    dataframe['fraud_reported']=dataframe.apply(assign_fraud_binary,axis=1)
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
    return(round(training.fraud_reported.value_counts(normalize=True),2).plot(kind='bar',color='limegreen'))

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