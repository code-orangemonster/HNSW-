import time
import sys
import numpy as np
sys.path.append('/home/wanghongya/bab/faiss150/python')
sys.path.append('/home/wanghongya/bab/faiss150/benchs')
import faiss
import util
import os
import random

from datasets import load_random
from datasets import load_glove
from datasets import load_audio
from datasets import load_sift1M
from datasets import load_deep
from datasets import load_gist
from datasets import load_imageNet

k = int(sys.argv[1])
m = int(sys.argv[2])
efcon = int(sys.argv[3])
r1 = float(sys.argv[4])
r2 = float(sys.argv[5])

nb1 = int(sys.argv[6])
nb2 = int(sys.argv[7])

todo = sys.argv[2:]

print("load data")
# xb, xq, xt, gt = load_deep()

xb, xq, xt, gt = load_sift1M()

nq, d = xq.shape
n=xb.shape[0]

print("Testing HNSW Flat")


# 生成len=xb.shape[0] 的 list 打乱顺序 
ls = [i for i in range(xb.shape[0])]
random.shuffle(ls)

def createIndex(xb): 
    index = faiss.IndexHNSWFlat(xb.shape[1], m)
    index.hnsw.efConstruction = efcon
    index.verbose = True
    index.hnsw.search_bounded_queue = False
    index.hnsw.search_mode = 0
    index.add(xb)
    return index

# 指向下次取数据的起始位置
cur = 0

def getSubscript(length):
    length *= 1000000
    print("length：",length)
    ss_list = []
    global cur
    for i in range(cur,cur+length):
        ss_list.append(ls[i])
    cur += length 
    return ss_list


indexList = []
indexScript = []
# 索引最大结点数
mm = xb.shape[0]//(3*1000000)
mn = 1

# 创建k个随机结点索引
for i in range(3):
    # 生成随机数,表示该索引的数据量
    # 最小值为1M ，最大值为mm 
    indexSize = random.randint(mn, mm)
    # 获取对应元素的下标列表 ，return list
    ss_list = getSubscript(indexSize)
    # 将每个索引对应的数据下标存入indexScript中
    indexScript.append(ss_list)
    # 根据下标从xb中拿数据。创建子图索引
    sub_xb = xb[ss_list]
    print(sub_xb.shape)
    # 构建子索引放入indexList
    sub_index = createIndex(sub_xb)
    indexList.append(sub_index)

# 热点参数
ratios = np.empty((1, 2), dtype=np.float32)
nb_nbors_per_level = np.empty((1, 2), dtype=np.int32)
print(ratios.shape)
# ratios.fill(0.0001)
# nb_nbors_per_level.fill(0)
ratios[0][0]=r1
ratios[0][1]=r2

nb_nbors_per_level[0][0]=nb1
nb_nbors_per_level[0][1]=nb2

print(ratios)
print(nb_nbors_per_level)



index = createIndex(xb) 

# 三类
idx1 = np.array(indexScript[0])
idx2 = np.array(indexScript[1])
idx3 = np.array(indexScript[2])
print(idx1.shape)
print(ratios.shape)


# 传入子图索引寻找热点
index.enhence_index_with_subIndex_hubs(n,2,faiss.swig_ptr(ratios),0,0,
    faiss.swig_ptr(nb_nbors_per_level),indexList[0],indexList[1],
    indexList[2],
    faiss.swig_ptr(idx1),
    faiss.swig_ptr(idx2),
    faiss.swig_ptr(idx3))








# 增强索引
# 2，1 表示使用聚类方法选择/随机分类，添加热点
# 0，0 表示使用全局方法选择，添加热点


# 数据收集
ef= []
r = []
t = []  

def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)
    D = np.empty((xq.shape[0], k), dtype=np.float32)
    I = np.empty((xq.shape[0], k), dtype=np.int64)
    t0 = time.time()
    index.search_with_hot_hubs_enhence(xq.shape[0], faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0])
    t1 = time.time()
    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    ef.append(index.hnsw.efSearch)
    r.append(float(format(recall_at_1, '.4f')))
    t.append(float(format((t1-t0)*1000.0/nq, '.4f')))
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))
    return recall_at_1

# 搜索
for efSearch in 50,100 ,200,300,400,500,600,700,800,900,1000,2000,3000,4000 ,5000 ,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000,25000,30000,35000,40000,50000:
    index.hnsw.efSearch = efSearch
    if(evaluate(index) == 1):
        break
print(ef)
print(r)
print(t)
    