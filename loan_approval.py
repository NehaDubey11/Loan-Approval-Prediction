# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 04:17:39 2021

@author: 91639
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

loan_data=pd.read_csv('01Exercise1.csv')

''' we are loading the data in another dataset 
so that the actual data will remain the same'''

loan_prep=loan_data.copy()

''' step-2 identify the missing values'''


print(loan_prep.isnull().sum(axis=0)) #axis is 0 means for rows

loan_prep=loan_prep.dropna()

print(loan_prep.isnull().sum(axis=0)) #axis is 0 means for rows

''' drop the column of gender as it's irreleavent'''

loan_prep=loan_prep.drop(['gender'],axis=1)
print(loan_prep.dtypes)

#step-3 creating dummies variables for categorical data
loan_prep=pd.get_dummies(loan_prep,drop_first=True)

#step-4 data normalisation 
from sklearn.preprocessing import StandardScaler
scaler_=StandardScaler()

loan_prep['income']=scaler_.fit_transform(loan_prep[['income']])
loan_prep['loanamt']=scaler_.fit_transform(loan_prep[['loanamt']])



#create X and Y dataframes

Y=loan_prep[['status_Y']]
X=loan_prep.drop(['status_Y'],axis=1)

#split the x and y datadet into training and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=\
train_test_split(X,Y,test_size=0.3,random_state=1234,stratify=Y)

# Build  logistic regression model


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

lr.fit(x_train,y_train)
y_predict=lr.predict(x_test)



from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_test, y_predict)
score=lr.score(x_test,y_test)

cr=classification_report(y_test, y_predict)

score2=accuracy_score(y_test,y_predict)

with open('lr_pickle','wb') as f:
    pickle.dump(lr,f)
    
with open('lr_pickle','rb') as f:
    mp=pickle.load(f)
    
mp.predict()














































