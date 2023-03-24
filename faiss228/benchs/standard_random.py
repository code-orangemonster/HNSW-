# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import numpy as np
sys.path.append('/home/wanghongya/bab/faiss150/benchs')
sys.path.append('/home/wanghongya/bab/faiss150/python')

import faiss
import util

from datasets import load_sift1M
from datasets import load_bigann
from datasets import load_deep
from datasets import load_gist
from datasets import load_glove
from datasets import load_sun
from datasets import load_random

k = int(sys.argv[1])
m = int(sys.argv[2])
efCon = int(sys.argv[3])


print("load data")

# xb, xq, xt, gt = load_sift1M()
# xb, xq, xt, gt = load_bigann()
# xb, xq, xt, gt = load_deep()
# xb, xq, xt, gt = load_gist()
# xb, xq, xt, gt = load_glove()
# xb, xq, xt, gt = load_sun()
xb, xq, xt, gt = load_random()


print(xq.shape)

'''
# 将数据（xq ，xb ，xt）存储到a中保存做相同的shuffle
a=xq

a = np.append(a, xb, axis=0)

a = np.append(a, xt, axis=0)


a = np.ascontiguousarray(a)

print(a.shape)

## shuffle数据集
def datasets_shuffle(datas):
    datas=datas.T
    np.random.shuffle(datas)
    datas=datas.T
    return datas

## 对 a shuffle
a=datasets_shuffle(a)
xb = np.ascontiguousarray(a)
print(a.shape)


## shuflle后的结果划分
xq=a[0:10000,:]
xb=a[10000:1010000,:]
xt=a[1010000:,:]

xb = np.ascontiguousarray(xb)
xq = np.ascontiguousarray(xq)
xt = np.ascontiguousarray(xt)

print(xq.shape)
print(xb.shape)
print(xt.shape)
'''
'''
def datasets_shuffle(xb,xq,xt):
    # 数据用a暂存
    a=xb
    a = np.append(a, xq, axis=0)
    a = np.append(a, xt, axis=0)
    # 转置之后，利用np.random的行shuffle
    a=a.T
    np.random.shuffle(a)
    # 转置回原来的样子
    a=a.T
    # 获取三者行数，数据分割
    nq=xq.shape[0]
    nb=xb.shape[0]
    nt=xt.shape[0]
    # 数据分割
    xb=a[0:nb,:]
    xq=a[nb:nb+nq,:]
    xt=a[nb+nq:,:]
    # 数据原顺序返回
    return xb,xq,xt



xb,xq,xt=datasets_shuffle(xb,xq,xt)
'''


print("load data")

nq, d = xq.shape



def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))

print("Testing HNSW Flat")

index = faiss.IndexHNSWFlat(d, m)

# training is not needed

# this is the default, higher is more accurate and slower to
# construct
index.hnsw.efConstruction = efCon

print("add")
# to see progress
index.verbose = True
index.add(xb)   
index.search_mode = 0
print("search")
for efSearch in 100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000,25000,30000,35000,40000,50000,60000,70000,80000,90000,100000:
    index.hnsw.efSearch = efSearch
    evaluate(index)
    print("\n")

