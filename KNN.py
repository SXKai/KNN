# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:26:47 2017

@author: Q
"""

import numpy as np
import collections

class KnnError(Exception):pass

def createDate():
    group = np.array([[1,1],[0.9,0.9],[0,0],[0.1,0.1]])
    label = np.array(['A','A','B','B'])
    return group,label

def kNN(data,group,label,k):
    datasize = group.shape[0]
    col_one = np.ones((datasize,1))
    data_max = np.kron(col_one,data)
    distance = np.linalg.norm(data_max-group,axis=1)
    maxlabel = []
    llabel = None;
    count = 0;
    for i in range(k):
        indice = np.argmax(distance)
        distance[indice] = -1
        maxlabel.append(label[indice])
    countlabel = collections.Counter(maxlabel)
    for key,value in countlabel.items():
        if count < value:
            llabel = key
    if llabel == None:
        raise KnnError("knn is error!")
    return llabel

if __name__ == "__main__": 
    group,label = createDate()
    b = np.array([0.9,0.5])
    print(kNN(b,group,label,3))