# -*- coding: utf-8 -*-
"""
Handwriting recognition by Random Forest
"""
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

data=pd.read_csv('mnist.csv')

data.head()

a=data.iloc[3,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)

df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

# Splitting data into train and test

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)
x_train.head()
y_train.head()

# Fitting model

rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)

# Predicting 

pred=rf.predict(x_test)
pred


# Testing

s=y_test.values
count=0

for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1
        
count
len(pred)

#8076/8400.0

#Conclusion: We get 96.14% of accuracy by applying random forest