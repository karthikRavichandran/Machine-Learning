import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def dist_cal_1(Quary, ref):
    dist_cal = []
    for i in range(len(ref)):
        dif = ref[i] - Quary
        sum_sq = 0
        for val in dif:
            sum_sq = sum_sq + math.pow(val,2)
        dist_cal.append(sum_sq)
    return dist_cal

def KNN(ref, train_x, train_y, K=10):
    dist_cal = dist_cal_1(ref, train_x)
    dist_convert = [(index,value) for index, value in enumerate(dist_cal)] 
    dist_convert = sorted(dist_convert, key = lambda x:x[1], reverse = False)
    dist_slice = dist_convert[:K]
    y_pred = []
    for i,val in dist_slice:
        y_pred.append(train_y[i])
    pred, frq = np.unique(y_pred, return_counts = True)
    out_pred_class = sorted(set(zip(pred,frq)), key = lambda x:x[1], reverse = True)
    out_classes = out_pred_class[0][0]
    return out_classes

if __name__ == "__main__":
	file = "/home/karthik/Downloads/Iris.csv"
	d_f = pd.read_csv(file)
	# d_f  = d_f.drop(["Id"], axis = 1)
	# print(d_f)
	Y = d_f["Species"].to_numpy()
	d_f  = d_f.drop(["Id", "Species"], axis = 1)
	X = d_f.to_numpy()
	train_x,test_x, train_y, test_y = train_test_split(X,Y)
	count = 0
	pred_y = []
	for i in test_x:
	    pred = KNN(i, train_x,train_y)
	#     print(pred)
	    pred_y.append(pred)
	#     print(test_y[count])
	    count = count + 1
	print(confusion_matrix(test_y, pred_y))