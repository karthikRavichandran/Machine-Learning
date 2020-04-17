#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
d_f = pd.read_csv("/home/karthik/Downloads/Iris.csv")


# In[93]:


print(d_f["Species"].unique())
case1 = d_f['Species'] == "Iris-setosa"
case2 = d_f['Species'] == "Iris-versicolor"
d_f2 = d_f[case1].append(d_f[case2])
print(d_f2["Species"].unique())
Y = []
for i in d_f2["Species"]:
    if i == "Iris-setosa":
        Y.append(1)
    elif i == "Iris-versicolor":
        Y.append(-1)
print(Y)
Y = np.array(Y) 
print(d_f2.keys())
d_f2.drop(["Species", "Id"], axis = 1)
x1 = np.array(d_f2["SepalLengthCm"])
x2 = np.array(d_f2["SepalWidthCm"])
x3 = np.array(d_f2["PetalLengthCm"])
x4 = np.array(d_f2["PetalWidthCm"])
X = np.vstack([x1,x2,x3,x4])
train_x,test_x,train_y, test_y =  train_test_split(X.T,Y)
train_x.shape
svm = SVC()
svm.fit(train_x, train_y)
y_pred = svm.predict(test_x[0:2,:])
print("y_pred : ", y_pred)
print("test_y : ", test_y)

print(train_x[:,1].shape[0])
# w1 = np.random.random(train_x[:,0].shape[0])
# w2 = np.random.random(train_x[:,1].shape[0])
w1 = 1
w2 = 1
epoch = 1
count = 0
alpha_ = 0.00001
while epoch < 10000:
    count = 0
    y = w1*train_x[:,0] + w2* train_x[:,1]
    y_pred = y*train_y
    lambda_ = 1/epoch
    print("Epoch : ", epoch)
    for val in y_pred:
        if val > 1:
            cost = 0
            w1 = w1 - 2*alpha_*lambda_*w1
            w2 = w2 - 2*alpha_*lambda_*w2
        else:
            cost = 1-y_pred
#             print("Epoch :", epoch, " count : ", count, " Cost :", cost)
            
            w1 = w1 + 2*alpha_*(train_y[count]*train_x[count,0]-lambda_*w1)
            w2 = w2 + 2*alpha_*(train_y[count]*train_x[count,1]-lambda_*w2)
        count = count + 1
#     print(y_pred)
    epoch = epoch + 1


# In[94]:


pred = w1*test_x[:,0] + w2*test_x[:,1]
pred_ = []
for i in pred:
    if i > 0:
        pred_.append(1)
    else:
        pred_.append(-1)
print("pred : ", pred_)
print("test_y :", test_y )

