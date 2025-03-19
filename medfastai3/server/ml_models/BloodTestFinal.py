#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction using Machine Learning
# 
# 
# - **Pregnancies**: Number of times pregnant
# - **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# - **BloodPressure**: Diastolic blood pressure (mm Hg)
# - **SkinThickness**: Triceps skin fold thickness (mm)
# - **Insulin**: 2-Hour serum insulin (mu U/ml)
# - **BMI**: Body mass index (weight in kg/(height in m)^2)
# - **DiabetesPedigreeFunction**: Diabetes pedigree function
# - **Age**: Age (years)
# - **Outcome**: Class variable (0 or 1)
# 
# **Number of Observation Units: 768**
# 
# **Variable Number: 9**
# 
# **Result; The model created as a result of XGBoost hyperparameter optimization became the model with the lowest Cross Validation Score value. (0.90)**

# # 1) Exploratory Data Analysis

# In[ ]:


#Installation of required libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action = "ignore")



# In[ ]:


#Reading the dataset
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")


# In[ ]:


# The first 5 observation units of the data set were accessed.
df.head()


# In[ ]:


# The size of the data set was examined. It consists of 768 observation units and 9 variables.
df.shape


# In[ ]:


#Feature information
df.info()


# In[ ]:


# Descriptive statistics of the data set accessed.
df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T


# In[ ]:


# The distribution of the Outcome variable was examined.
df["Outcome"].value_counts()*100/len(df)


# In[ ]:


# The classes of the outcome variable were examined.
df.Outcome.value_counts()


# In[ ]:


# The histagram of the Age variable was reached.
df["Age"].hist(edgecolor = "black");


# In[ ]:


print("Max Age: " + str(df["Age"].max()) + " Min Age: " + str(df["Age"].min()))


# In[ ]:


# Histogram and density graphs of all variables were accessed.
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df.Age, bins = 20, ax=ax[0,0])
sns.distplot(df.Pregnancies, bins = 20, ax=ax[0,1])
sns.distplot(df.Glucose, bins = 20, ax=ax[1,0])
sns.distplot(df.BloodPressure, bins = 20, ax=ax[1,1])
sns.distplot(df.SkinThickness, bins = 20, ax=ax[2,0])
sns.distplot(df.Insulin, bins = 20, ax=ax[2,1])
sns.distplot(df.DiabetesPedigreeFunction, bins = 20, ax=ax[3,0])
sns.distplot(df.BMI, bins = 20, ax=ax[3,1])


# In[ ]:


df.groupby("Outcome").agg({"Pregnancies":"mean"})


# In[ ]:


df.groupby("Outcome").agg({"Age":"mean"})


# In[ ]:


df.groupby("Outcome").agg({"Age":"max"})


# In[ ]:


df.groupby("Outcome").agg({"Insulin": "mean"})


# In[ ]:


df.groupby("Outcome").agg({"Insulin": "max"})


# In[ ]:


df.groupby("Outcome").agg({"Glucose": "mean"})


# In[ ]:


df.groupby("Outcome").agg({"Glucose": "max"})


# In[ ]:


df.groupby("Outcome").agg({"BMI": "mean"})


# In[ ]:


# The distribution of the outcome variable in the data was examined and visualized.
f,ax=plt.subplots(1,2,figsize=(18,8))
df['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('Outcome',data=df,ax=ax[1])
ax[1].set_title('Outcome')
plt.show()


# In[ ]:


# Access to the correlation of the data set was provided. What kind of relationship is examined between the variables.
# If the correlation value is> 0, there is a positive correlation. While the value of one variable increases, the value of the other variable also increases.
# Correlation = 0 means no correlation.
# If the correlation is <0, there is a negative correlation. While one variable increases, the other variable decreases.
# When the correlations are examined, there are 2 variables that act as a positive correlation to the Salary dependent variable.
# These variables are Glucose. As these increase, Outcome variable increases.
df.corr()


# In[ ]:


# Correlation matrix graph of the data set
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


# # 2) Data Preprocessing

# ## 2.1) Missing Observation Analysis
# 
# We saw on df.head() that some features contain 0, it doesn't make sense here and this indicates missing value Below we replace 0 value by NaN:

# In[ ]:


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[ ]:


df.head()


# In[ ]:


# Now, we can look at where are missing values
df.isnull().sum()


# In[ ]:


# Have been visualized using the missingno library for the visualization of missing observations.
# Plotting
import missingno as msno
msno.bar(df);


# In[ ]:


# The missing values ​​will be filled with the median values ​​of each variable.
def median_target(var):
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
    return temp


# In[ ]:


# The values to be given for incomplete observations are given the median value of people who are not sick and the median values of people who are sick.
columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]


# In[ ]:


df.head()


# In[ ]:


# Missing values were filled.
df.isnull().sum()


# ## 2.2) Outlier Observation Analysis

# In[ ]:


# In the data set, there were asked whether there were any outlier observations compared to the 25% and 75% quarters.
# It was found to be an outlier observation.
for feature in df:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1- 1.5*IQR
    upper = Q3 + 1.5*IQR

    if df[(df[feature] > upper)].any(axis=None):
        print(feature,"yes")
    else:
        print(feature, "no")


# In[ ]:


# The process of visualizing the Insulin variable with boxplot method was done. We find the outlier observations on the chart.
import seaborn as sns
sns.boxplot(x = df["Insulin"]);


# In[ ]:


#We conduct a stand alone observation review for the Insulin variable
#We suppress contradictory values
Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Insulin"] > upper,"Insulin"] = upper


# In[ ]:


import seaborn as sns
sns.boxplot(x = df["Insulin"]);


# ## 2.3)  Local Outlier Factor (LOF)

# In[ ]:


# We determine outliers between all variables with the LOF method
from sklearn.neighbors import LocalOutlierFactor
lof =LocalOutlierFactor(n_neighbors= 10)
lof.fit_predict(df)


# In[ ]:


df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:30]


# In[ ]:


#We choose the threshold value according to lof scores
threshold = np.sort(df_scores)[7]
threshold


# In[ ]:


#We delete those that are higher than the threshold
outlier = df_scores > threshold
df = df[outlier]


# In[ ]:


# The size of the data set was examined.
df.shape


# # 3) Feature Engineering
# 
# Creating new variables is important for models. But you need to create a logical new variable. For this data set, some new variables were created according to BMI, Insulin and glucose variables.

# In[ ]:


# According to BMI, some ranges were determined and categorical variables were assigned.
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]


# In[ ]:


df.head()


# In[ ]:


# A categorical variable creation process is performed according to the insulin value.
def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


# In[ ]:


# The operation performed was added to the dataframe.
df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))

df.head()


# In[ ]:


# Some intervals were determined according to the glucose variable and these were assigned categorical variables.
NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")
df["NewGlucose"] = NewGlucose
df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]


# In[ ]:


df.head()


# # 4) One Hot Encoding
# 

# In[ ]:


# Here, by making One Hot Encoding transformation, categorical variables were converted into numerical values. It is also protected from the Dummy variable trap.
df = pd.get_dummies(df, columns =["NewBMI","NewInsulinScore", "NewGlucose"], drop_first = True)


# In[ ]:


df.head()


# In[ ]:


categorical_df = df[['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]


# In[ ]:


categorical_df.head()


# In[ ]:


y = df["Outcome"]
X = df.drop(["Outcome",'NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis = 1)
cols = X.columns
index = X.index


# In[ ]:


X.head()


# In[ ]:


# The variables in the data set are an effective factor in increasing the performance of the models by standardization.
# There are multiple standardization methods. These are methods such as" Normalize"," MinMax"," Robust" and "Scale".
from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns = cols, index = index)


# In[ ]:


X.head()


# In[ ]:


X = pd.concat([X,categorical_df], axis = 1)


# In[ ]:


X.head()


# In[ ]:


y.head()


# # 5) Base Models

# In[ ]:


# Validation scores of all base models

models = []
models.append(('LR', LogisticRegression(random_state = 12345)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier(random_state = 12345)))
models.append(('XGB', GradientBoostingClassifier(random_state = 12345)))
models.append(("LightGBM", LGBMClassifier(random_state = 12345)))

# evaluate each model in turn
results = []
names = []


# In[ ]:


for name, model in models:

        kfold = KFold(n_splits = 10, random_state = 12345)
        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# # 6) Model Tuning

# ### 1) Random Forests Tuning

# In[ ]:


rf_params = {"n_estimators" :[100,200,500,1000],
             "max_features": [3,5,7],
             "min_samples_split": [2,5,10,30],
            "max_depth": [3,5,8,None]}


# In[ ]:


rf_model = RandomForestClassifier(random_state = 12345)


# In[ ]:


gs_cv = GridSearchCV(rf_model,
                    rf_params,
                    cv = 10,
                    n_jobs = -1,
                    verbose = 2).fit(X, y)


# In[ ]:


gs_cv.best_params_


# ### 1.1) Final Model Installation

# In[ ]:


rf_tuned = RandomForestClassifier(**gs_cv.best_params_)


# In[ ]:


rf_tuned = rf_tuned.fit(X,y)


# In[ ]:


cross_val_score(rf_tuned, X, y, cv = 10).mean()


# In[ ]:


feature_imp = pd.Series(rf_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Variable Severity Levels")
plt.show()


# ### 2) LightGBM Tuning

# In[ ]:


lgbm = LGBMClassifier(random_state = 12345)


# In[ ]:


lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
              "n_estimators": [500, 1000, 1500],
              "max_depth":[3,5,8]}


# In[ ]:


gs_cv = GridSearchCV(lgbm,
                     lgbm_params,
                     cv = 10,
                     n_jobs = -1,
                     verbose = 2).fit(X, y)


# In[ ]:


gs_cv.best_params_


# ### 2.1) Final Model Installation

# In[ ]:


lgbm_tuned = LGBMClassifier(**gs_cv.best_params_).fit(X,y)


# In[ ]:


cross_val_score(lgbm_tuned, X, y, cv = 10).mean()


# In[ ]:


feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Variable Severity Levels")
plt.show()


# ### 3) XGBoost Tuning

# In[ ]:


xgb = GradientBoostingClassifier(random_state = 12345)


# In[ ]:


xgb_params = {
    "learning_rate": [0.01, 0.1, 0.2, 1],
    "min_samples_split": np.linspace(0.1, 0.5, 10),
    "max_depth":[3,5,8],
    "subsample":[0.5, 0.9, 1.0],
    "n_estimators": [100,1000]}


# In[ ]:


xgb_cv_model  = GridSearchCV(xgb,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X, y)


# In[ ]:


xgb_cv_model.best_params_


# ### 3.1) Final Model Installation

# In[ ]:


xgb_tuned = GradientBoostingClassifier(**xgb_cv_model.best_params_).fit(X,y)


# In[ ]:


cross_val_score(xgb_tuned, X, y, cv = 10).mean()


# In[ ]:


feature_imp = pd.Series(xgb_tuned.feature_importances_,
                        index=X.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Significance Score Of Variables')
plt.ylabel('Variables')
plt.title("Variable Severity Levels")
plt.show()


# # 7) Comparison of Final Models

# In[ ]:


models = []

models.append(('RF', RandomForestClassifier(random_state = 12345, max_depth = 8, max_features = 7, min_samples_split = 2, n_estimators = 500)))
models.append(('XGB', GradientBoostingClassifier(random_state = 12345, learning_rate = 0.1, max_depth = 5, min_samples_split = 0.1, n_estimators = 100, subsample = 1.0)))
models.append(("LightGBM", LGBMClassifier(random_state = 12345, learning_rate = 0.01,  max_depth = 3, n_estimators = 1000)))

# evaluate each model in turn
results = []
names = []


# In[ ]:


for name, model in models:

        kfold = KFold(n_splits = 10, random_state = 12345)
        cv_results = cross_val_score(model, X, y, cv = 10, scoring= "accuracy")
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15,10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# # 8) Reporting
# 
# The aim of this study was to create classification models for the diabetes data set and to predict whether a person is sick by establishing models and to obtain maximum validation scores in the established models. The work done is as follows:
# 
# 1) Diabetes Data Set read.
# 
# 2) With Exploratory Data Analysis; The data set's structural data were checked.
# The types of variables in the dataset were examined. Size information of the dataset was accessed. The 0 values in the data set are missing values. Primarily these 0 values were replaced with NaN values. Descriptive statistics of the data set were examined.
# 
# 3) Data Preprocessing section;
# df for: The NaN values missing observations were filled with the median values of whether each variable was sick or not. The outliers were determined by LOF and dropped. The X variables were standardized with the rubost method..
# 
# 4) During Model Building;
# Logistic Regression, KNN, Random Forests, XGBoost, LightGBM like using machine learning models Cross Validation Score were calculated. Later Random Forests, XGBoost, LightGBM hyperparameter optimizations optimized to increase Cross Validation value.
# 
# 5) Result;
# The model created as a result of XGBoost hyperparameter optimization became the model with the lowest Cross Validation Score value. (0.90)
