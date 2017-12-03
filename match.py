# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 10:48:58 2017

@author: Q
"""

import numpy as np
import matplotlib.pyplot as plt
import KNN
def file2matrix(filename):
    data = np.loadtxt(filename,usecols=(0,1,2))
    label = np.loadtxt(filename,dtype=np.str,usecols=(3,))
    return data,label
def str2num(label):
    name = []
    label_ = label.copy()
    for i in range(len(label_)):
        if label_[i] in name:
            label_[i] = name.index(label_[i])
        else: 
            name.append(label_[i])
            label_[i] = name.index(label_[i])
    return label_
def autoNorm(data):
    col_num = data.shape[1]
    for i in range(col_num):
        col = data[:,i]
        maxx = col.max()
        minn = col.min()
        for j in range(len(col)):
            col[j] = (col[j]-minn)/(maxx - minn)
        col = col.reshape(len(col),1)
        if i == 0:
            return_data = col
        else:
            return_data = np.hstack((return_data,col))
    return return_data
            
        
if __name__=="__main__":
    data,label = file2matrix('datingTestSet.txt')
    label_num = str2num(label)
    group = autoNorm(data)
    test = np.array([0.5,0.5,0.5])
    test_label = KNN.kNN(test,group,label,270)
    print(test_label)
    
    
#    fig = plt.figure(0)
#    ax = fig.add_subplot(111)
#    ax.scatter(data[:,1],data[:,2],c=label_num)
#    fig.show()