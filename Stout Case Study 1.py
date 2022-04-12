#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


# In[16]:


df = pd.read_csv("loans_full_schema.csv")


# In[40]:


df


# In[17]:


plt.figure(figsize=(15,8))

plt.subplot(121)
g = sns.distplot(df["loan_amount"])
g.set_xlabel("", fontsize=12)
g.set_ylabel("Frequency Dist", fontsize=12)
g.set_title("Frequency Distribuition of Loan Amount", fontsize=20)


# In[20]:


df['int_round'] = df['interest_rate'].round(0).astype(int)

plt.figure(figsize = (10,8))
g1 = sns.countplot(x="int_round",data=df, 
                   palette="Set3")
g1.set_xlabel("Int Rate", fontsize=12)
g1.set_ylabel("Count", fontsize=12)
g1.set_title("Interest Rate Normal Distribuition", fontsize=20)


# In[25]:


plt.figure(figsize=(8,5))
sns.boxplot(x='homeownership',y='interest_rate',data=df)
plt.title("Interest Rate by Homeownership")


# In[41]:


plt.figure(figsize=(18,5))
sns.barplot(x='loan_purpose',y='interest_rate',data=df)
plt.title("Interest Rate by Loan Purpose")


# In[27]:


plt.figure(figsize=(8,5))
sns.boxplot(x='verified_income',y='interest_rate',data=df)
plt.title("Interest Rate by Verified Income")


# In[4]:


nan_cols = ['emp_length', 'annual_income_joint', 'debt_to_income_joint', 'debt_to_income', 'months_since_last_delinq', 
                 'months_since_90d_late', 'months_since_last_credit_inquiry', 'num_accounts_120d_past_due']
df[nan_cols] = df[nan_cols].fillna(0)


# In[5]:


columns=['emp_length', 'homeownership', 'annual_income',
       'verified_income', 'debt_to_income', 'annual_income_joint',
       'verification_income_joint', 'debt_to_income_joint', 'delinq_2y',
       'months_since_last_delinq', 'earliest_credit_line',
       'inquiries_last_12m', 'total_credit_lines', 'open_credit_lines',
       'total_credit_limit', 'total_credit_utilized',
       'num_collections_last_12m', 'num_historical_failed_to_pay',
       'months_since_90d_late', 'current_accounts_delinq',
       'total_collection_amount_ever', 'current_installment_accounts',
       'accounts_opened_24m', 'months_since_last_credit_inquiry',
       'num_satisfactory_accounts', 'num_accounts_120d_past_due',
       'num_accounts_30d_past_due', 'num_active_debit_accounts',
       'total_debit_limit', 'num_total_cc_accounts', 'num_open_cc_accounts',
       'num_cc_carrying_balance', 'num_mort_accounts',
       'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt',
        'application_type',
       'interest_rate', 'installment',
       'balance', 'paid_total', 'paid_principal', 'paid_interest',
       'paid_late_fees']


# In[6]:


df = df[columns]
df_final = pd.get_dummies(df)


# In[7]:


X=df_final.drop("interest_rate", axis=1)
y=df_final['interest_rate']


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# In[11]:


def evaluate(model,X_train,X_test,Y_train,Y_test):
    
    model.fit(X_train,Y_train)
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)    
    errors_train = abs(predictions_train - Y_train)
    errors_test = abs(predictions_test - Y_test)
    
    mape_train = 100 * np.mean(errors_train / Y_train)
    mape_test = 100 * np.mean(errors_test / Y_test)
    
    accuracy_train = 100 - mape_train
    accuracy_test = 100 - mape_test
    print('Model Performance')
    
    print('Accuracy(Train Data) = {:0.2f}%.'.format(accuracy_train))
    print('Accuracy(Test Data) = {:0.2f}%.'.format(accuracy_test))
    plt.figure(figsize = (10,10))
    plt.scatter(predictions_train,(predictions_train - Y_train),c='g',s=40,alpha=0.5)
    plt.scatter(predictions_test,(predictions_test - Y_test),c='b',s=40,alpha=0.5)
    plt.hlines(y=0,xmin=0,xmax=30)
    plt.title('residual plot: Blue - test data and Green - train data')
    plt.ylabel('residuals')
    return accuracy_train,accuracy_test


# In[39]:


param_lasso = {'alpha' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}
lasso_reg = Lasso()
lasso_grid = GridSearchCV(lasso_reg, param_lasso, cv = 5)
evaluate(lasso_grid,X_train, X_test, y_train, y_test)


# In[13]:


model = RandomForestRegressor(n_estimators= 10, random_state=42)
evaluate(model,X_train, X_test, y_train, y_test)

