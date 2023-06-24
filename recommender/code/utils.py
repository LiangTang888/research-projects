#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from datetime import datetime
from datetime import timedelta
import pandas as pd
from math import sqrt
import os, copy, math, time, h5py, json
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
from nltk.corpus import stopwords
#from svmutil import svm_read_problem

#
def output_evaluate( real_ary, pred_ary ):
    # print(real_ary,pred_ary)
    real_ary = np.array(real_ary);
    pred_ary = np.array(pred_ary);
    assert len(real_ary)==len(pred_ary),"len(real_ary)!=len(pred_ary)"
    assert len(real_ary[0])==len(pred_ary[0]),"len(real_ary[0])!=len(pred_ary[0])"
    D = len(real_ary); k = len(real_ary[0])
    ############  delta2_aspect #################
    delta2_aspect = 1.0*np.sum((real_ary-pred_ary)**2)/(D*k)
    ############ p_aspect #################
    sigma = 0.0
    for i in range(len(real_ary)):
        tmp = corrcoef( real_ary[i], pred_ary[i] )
        if np.isnan(tmp): sigma += 0
        else: sigma += tmp
    p_aspect = sigma / D
    ############ p_review #################
    sigma = 0.0
    for i in range(len(real_ary[0])):
        tmp = corrcoef(real_ary[:,i],pred_ary[:,i])
        if np.isnan(tmp): sigma += 0
        else: sigma += tmp
    p_review = 1.0*sigma/k
    ############ MAP  #################
    print("delta2_aspect = %.4f."%(delta2_aspect)),
    print("p_aspect = %.4f."%(p_aspect)),
    print("p_review = %.4f."%(p_review))

# 
def corrcoef(x,y):
    n=len(x)
    #sum
    sum1 = sum(x)
    sum2 = sum(y)
    sumofxy = multipl(x,y)
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num = sumofxy - (float(sum1)*float(sum2)/n)
    a = (sumofx2-float(sum1**2)/n)
    b = (sumofy2-float(sum2**2)/n)
    if a<1e-8 or b < 1e-8: return 0
    den = sqrt( a * b )
    return num / den

def multipl(a,b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab

def my_normalization( data_ary, size=(-1,1), axis=0 ):
    # axis = 0 
    if axis == 1:
        data_ary = np.matrix(data_ary).T
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=size)
        ans = min_max_scaler.fit_transform(ans)
        ans = np.matrix(ans).T
        ans = np.array(ans)
    else:
        ans = preprocessing.scale(data_ary)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=size)
        ans = min_max_scaler.fit_transform(ans)
    return ans

def one_hot( data_ary, one_hot_len):
	# data_ary = array([1,2,3,5,6,7,9])
	# one_hot_len: one_hot最长列
    max_num = np.max(data_ary);
    ans = np.zeros((len(data_ary),one_hot_len),dtype=np.float64)
    for i in range(len(data_ary)):
    	ans[ i, int(data_ary[i]) ] = 1.0
    return ans

def re_onehot( data_ary ):
	# data_ary = array([[0,0,0,1.0],[1.0,0,0,0],...])
    ans = np.zeros((len(data_ary),),dtype=np.float64)
    for i in range(len(data_ary)):
    	for j in range(len(data_ary[i,:])):
        	if data_ary[i,j] == 1.0:
        		ans[i] = 1.0*j;
        		break;
    return ans

def write2H5(h5DumpFile,data):
    # if not os.path.exists(h5DumpFile): os.makedir(h5DumpFile)
    with h5py.File(h5DumpFile, "w") as f:
        f.create_dataset("data", data=data, dtype=np.float64)

def readH5(h5DumpFile):
    feat = [];
    with h5py.File(h5DumpFile, "r") as f:
        feat.append(f['data'][:])
    feat = np.concatenate(feat, 1)
    print('readH5 Feature.shape=', feat.shape)
    return feat.astype(np.float64)

def store_json(file_name,data):
    with open(file_name, 'w') as json_file:
        json_file.write(json.dumps(data,indent=2))

def load_json(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data

def moveFileto( sourceDir, targetDir ): shutil.copy( sourceDir, targetDir )

def removeDir(dirPath):
    if not os.path.isdir(dirPath): return
    files = os.listdir(dirPath)
    try:
        for file in files:
            filePath = os.path.join(dirPath, file)
            if os.path.isfile(filePath):
                os.remove(filePath)
            elif os.path.isdir(filePath):
                removeDir(filePath)
        os.rmdir(dirPath)
    except Exception(e): print(e)

