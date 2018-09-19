# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:38:26 2018

@author: USER
"""

import pandas as pd
import numpy as np
adult_df = pd.read_csv(r'D:/Document/Python Scripts/adult_data.csv',header = None,delimiter=' *, *',engine='python')
adult_df.shape
#%%
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num','marital_status', 'occupation', 'relationship','race', 'sex', 'capital_gain', 'capital_loss','hours_per_week', 'native_country', 'income']

adult_df.head()
adult_df.isnull().sum()
#%%
for value in['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']:
 print(value,sum(adult_df[value]=='?'))
 
 #%%
#create a copy of the dataframe
adult_df_rev=pd.DataFrame.copy(adult_df)
adult_df_rev.describe(include = 'all')
#%%
for value in ['workclass','occupation','native_country']:
    adult_df_rev[value].replace(['?'], adult_df_rev[value].mode()[0],inplace=True)
#%%

for value in ['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']:
    print(value,sum(adult_df_rev[value]=='?'))
#%%
colname = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']
colname
#%%
#For preprocessing the data
from sklearn import preprocessing

le={}

for x in colname:
      le[x]=preprocessing.LabelEncoder()

for x in colname:
      adult_df_rev[x]=le[x].fit_transform(adult_df_rev.__getattr__(x))

#%% 
adult_df_rev.head()
#0--> <=50K
#1--> >50K
#-1 is to Exclude the last variable i.e., is Income 
X = adult_df_rev.values[:,:-1]
Y = adult_df_rev.values[:,-1]
#%%
#Not a mandatory step but would normalize the data so that the Model could Predict with Accuracy
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X)
X=scaler.transform(X)
print(X)

""" 
Sometime it creates an object and it would give you an Error, Hence this
is a Precaution that would help in coverting the Non-Number values to Int.
""" 
Y=Y.astype(int)

#%%
from sklearn.model_selection import train_test_split

#Split the data into test and train
"""
Always have the same Sequence of the train and test varilables(X_train,X_test etc. 
that are mentioned on left hand side.
"""
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=10)
#%%
#Below is the script used for Prediction
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())
#fitting training data to the model
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

#%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ", acc)


