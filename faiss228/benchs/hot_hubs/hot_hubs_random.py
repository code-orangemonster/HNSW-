import time
import sys
import numpy as np
sys.path.append('/home/wanghongya/bab/faiss150/python')
sys.path.append('/home/wanghongya/bab/faiss150/benchs')
import faiss
import util
import os
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
# xb, xq, xt, gt = load_sift1M()                            
xb, xq, xt, gt = load_audio()
# xb, xq, xt, gt = load_imageNet()
# xb, xq, xt, gt = load_random()
# xb, xq, xt, gt = load_glove()
# xb, xq, xt, gt = load_gist()
# xb, xq, xt, gt = load_deep()

nq, d = xq.shape
n=xb.shape[0]

print("Testing HNSW Flat")


# 数据聚类
kmeans = faiss.Kmeans(d, 10)

print("training")
kmeans.train(xb)
print("training done.")

# 每个点对应的类id，该点到质心的距离
D1,I1 = kmeans.assign(xb)

print(len(I1))

# faiss.swig_ptr(D)

# 将质心存为fvecs文件
# fvecs_write("/home/wanghongya/bab/kmeans/1.fvecs",kmeans.centroids)
# 读取质心
# centroids = fvecs_read("/home/wanghongya/bab/kmeans/1.fvecs")


# 创建标准索引
index = faiss.IndexHNSWFlat(d, m)
index.hnsw.efConstruction = efcon
index.verbose = True
index.hnsw.search_bounded_queue = False
index.hnsw.search_mode = 0
index.add(xb)

# 热点参数
ratios = np.empty((1, 2), dtype=np.float32)
nb_nbors_per_level = np.empty((1, 2), dtype=np.int32)
# ratios.fill(0.0001)
# nb_nbors_per_level.fill(0)
ratios[0][0]=r1
ratios[0][1]=r2
# ratios[0][1]=0.001

nb_nbors_per_level[0][0]=nb1
nb_nbors_per_level[0][1]=nb2
# nb_nbors_per_level[0][1]=12
print(ratios)
print(nb_nbors_per_level)


# 增强索引
# 2，1 表示使用聚类方法选择，添加热点
# 0，0 表示使用全局方法选择，添加热点
index.combine_index_with_hot_hubs_enhence(n,2,faiss.swig_ptr(ratios),0,0,
    faiss.swig_ptr(nb_nbors_per_level),faiss.swig_ptr(I1))

# 数据收集
ef= []
r = []
t = []  
failed_queries = []

def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)
    D = np.empty((xq.shape[0], k), dtype=np.float32)
    I = np.empty((xq.shape[0], k), dtype=np.int64)
    t0 = time.time()
    # index.search_with_hot_hubs_enhence(xq.shape[0], faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0])
    index.search_with_hot_hubs_enhence_random(xq.shape[0], faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0])
    t1 = time.time()
    t1 = time.time()
    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    for i in range(len(I)):
        if I[i][0]!=gt[i][0]:
            failed_queries.append(i)
    ef.append(index.hnsw.efSearch)
    r.append(float(format(recall_at_1, '.4f')))
    t.append(float(format((t1-t0)*1000.0/nq, '.4f')))
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))
    return recall_at_1

# 搜索
# for efSearch in 50 ,100:# ,200,300,400,500,600,700,800,900,1000,2000,3000,4000 ,5000 ,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000,25000,30000,35000,40000,50000:
#     index.hnsw.efSearch = efSearch
#     if(evaluate(index) == 1):
#         break
index.hnsw.efSearch = 200
evaluate(index)
print(ef)
print(r)
print(t)
print(len(failed_queries))
print(failed_queries)

    