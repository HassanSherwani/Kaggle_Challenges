# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline

#Getting data
data=pd.read_csv('mnist.csv')

df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

# Splitting into train and test
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)

nn.fit(x_train,y_train)

pred=nn.predict(x_test)

#activation logistic with hidden layer sizes-> 45,90 Gave 92 % accuracy
#activation relu with hidden layer sizes-> 45,90 gave 89 % accuracy 
#Test with different combinatations of learning rate, activation and other hyper parameters and mesure the accuracy

a=y_test.values

a
count=0

for i in range(len(pred)):
    if pred[i]==a[i]:
        count=count+1
        
count

len(pred)

6824/8400.0

# Creating confusion matrix

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Applying confusion matrix
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['Not accurate', 'accurate'])

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)