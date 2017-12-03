# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 11:06:08 2017

@author: Q
"""

import numpy as np
import matplotlib.pyplot as plt
import KNN
import os
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

        
        
    
    
    
    
    
if __name__ == "__main__":
    trainFilename = os.listdir("trainingDigits")
    for i in range(len(trainFilename)):
        trainFilename[i] = trainFilename[i].split('.')[0]
        trainFilename[i] = trainFilename[i].split('_')[0]
    print(trainFilename)