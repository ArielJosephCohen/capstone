# **Introduction**: The goal of this notebook is to comprehensively go through the process of creating machine learning models for predicting potential car insurance fraud. When a car is hit or stolen, the insurance company pays out to the effected parties a certain amount of cash as compensation for their monetary loss. Unfortuneately, insured customers and drivers will at times lie about or fake details in a reported insurance claim to make some easy money. My models attempt to solve this problem in two ways. The first way is to determine what models best predict potential fraud. The second is similar, where I attempt to identify the characterisitcs of a report most indicative of potential fraud. The value I attempt to predict is a binary outcome as to whether a report was fraudulent or not.

## Links: 
GitHub: https://github.com/ArielJosephCohen/capstone
Data: https://www.kaggle.com/roshansharma/insurance-claim
Presentation: https://docs.google.com/presentation/d/1IQdYSxrzyGvMpurhM-i097Btp4ksqL70WLEiM6yc5Sw/edit#slide=id.g35f391192_00

# As I attempt to solve this problem, I will go through the following process: Collect relevant data, explore data, clean data, model data, and finally, draw conclusions.

## My data for the project is from kaggle.com and can be found at [data_link](https://www.kaggle.com/roshansharma/insurance-claim). The first thing I will do after gathering the data will be to assign a random seed for uniform randomness (for this project, I chose my luck number which is 14). My data consisted of 1000 rows and 39 columns, with the final column being the output of wheher or not a claim is fraudulent based on input from all the other features. Below, I have included a snapshot of my data
![Initial Data](/Users/flatironschool/Documents/capstone/images/first_df.png)
## In the data, 247 rows out of 1000 total involved fraud. The predictive features are as follows: months as customer, age, policy number, policy bind date, state where policy signed, policy csl (a policy limit), policy deductible, annual policy premium, umbrella limit (limit of extra insurance), zip code of insured, gender of insured, relationship with policy holder, gains in incident, losses in incident, incident type, collision type if relevant, severity of incident, location of incident, hour of day incident took place, vehicles involved, pressence of property damage, pressence of bodily injuries, witnesses present, pressence of police report, claim amount, pressence of injury claim, pressence of property claim, pressence of vehicle claim, car type, car model, and car year.

## Next, I did some initial data exploring. The following histogram shows the frequency of fraud commited by age.
![Fraud by Age](/Users/flatironschool/Documents/capstone/images/fraud_by_age.png)
## The following histogram shows the amount of fraud commited by hour of the day
![Fraud by Hour of Day](/Users/flatironschool/Documents/capstone/images/fraud_by_hour.png)
## The following visual show the breakdown of different types of claims filed
![Different Type of Reports](/Users/flatironschool/Documents/capstone/images/incident_type.png)
## The following visual shows the breakdown of fraud by gender
![Fraud by Gender](/Users/flatironschool/Documents/capstone/images/fraud_by_gender.png)

## Next, I did my data cleaning. The steps involved included: replacing misleading values, assigning numerical values to binary categorical variables, creating custom features to better explain dates, adding a timeline between policy bind date and incidet, and making capital losses an absolute value.

## Next, I split up the data into numerical and categorical features and did some additional fine-tuning. The following chart shows the amount of unique values for relevant categorical features:
![Unique Values per Categorical Feature](/Users/flatironschool/Documents/capstone/images/cat_unique.png)
## Next, I removed highly correlated numerical features. The following heatmap shows how I determined what to remove from the data:
![Numerical Correlation](/Users/flatironschool/Documents/capstone/images/first_num_Corr.png)
## After viewing this, I removed the age, vehicle_claim, and total_claim_amount features. The follwoing heatmap shows the updated correlation visual:
![(Updates) Numerical Correlation](/Users/flatironschool/Documents/capstone/images/second_num_corr.png)
## Next, I examined each categorical feature, found how often each feature correlated to fraud on average, and used that data to encode a numerical value. Below, I have a piece of code demonstrating this process:
![Encoding Categorical Data (Education Level)](/Users/flatironschool/Documents/capstone/images/fraud_education_encoding.png)
## Next I created a new dataframe with numerical features and numerically-encoded categorical features to level the units. I also removed any new correlation created present in the following heatmap (particularly incident_type):
![Correlation with Categorical Encoding](/Users/flatironschool/Documents/capstone/images/first_Cat_corr.png)
## My next step was to remove features with low contribution to explaining data using Recursive Feature Elimination (RFE). After doing RFE, I reduced my data to include only 21 predictive features (from 33). As my final steps of cleaning, I removed outliers, applied min-max scaling for evening out differences in feature magnitude, and normalized data using Box-Cox transformations

## After this, I am almost ready to model. However, there are still two more steps I must take: splitting data into train and test sets as well as balancing train data. Since fraud occurs rather infrequently, I may get misleading results by training on data that is primarily not consisting fraud. Therefore, balanced the data in my train set so the amount of cases involving fraud would be equal to the cases not involving fraud. The following two visuals show the original balance (by number of occurances) and the new balance (by frequency of occurances):
![Initial Target Feature Balance](/Users/flatironschool/Documents/capstone/images/initial_fraud_balance.png)
![Updateds Target Feature Balance](/Users/flatironschool/Documents/capstone/images/new_fraud_balance.png)

## Next I tried the following variety of machine learning classification models to find the one that best describes and predicts what is going on with my data: Support Vector Machines, K Nearest Neighbors, Logistic Regression, Random Forest, Gaussian Naive Bayes, Stochastic Gradient Descent, Linear SVC, Decision Tree, and XGBoost. I first check my score on training data and the validated on the test data derrived from the splitting of data mentioned above. Below, I will display tables containing my train results followed by test results:
![Train Data Summary](/Users/flatironschool/Documents/capstone/images/train_summary.png)
![Test Data Summary](/Users/flatironschool/Documents/capstone/images/test_summary.png)
## Having found my optimal model, I now used that model to find the features most indicative of potential fraud. The following will be a table of highest to lowest (of top 10) features ccorrelated to potential fraud:
![Fraud Indicators](/Users/flatironschool/Documents/capstone/images/fraud_indicators.png)
## Finally, just for visualization, I have provided a preview of what my decision tree model looks like:
![Decision Tree](/Users/flatironschool/Documents/capstone/images/decision_tree_image.png)

# That about wraps it up!
# Thanks for reading!