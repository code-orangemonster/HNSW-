import time
import sys
import numpy as np
sys.path.append('/home/wanghongya/bab/faiss150/python')
import faiss
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
# xb, xq, xt, gt = load_audio()
# xb, xq, xt, gt = load_imageNet()
# xb, xq, xt, gt = load_random()
xb, xq, xt, gt = load_glove()
# xb, xq, xt, gt = load_gist()
# xb, xq, xt, gt = load_deep()

nq, d = xq.shape
n=xb.shape[0]

print("Testing HNSW Flat")

ls = []
for i in range(len(gt)):
    ls.append(gt[i][0])

I1 = np.array(ls,dtype=np.int64)


index = faiss.IndexHNSWFlat(d, m)
index.hnsw.efConstruction = efcon
index.verbose = True
index.hnsw.search_bounded_queue = False
index.hnsw.search_mode = 0
index.add(xb)

ratios = np.empty((1, 2), dtype=np.float32)
nb_nbors_per_level = np.empty((1, 2), dtype=np.int32)

ratios[0][0]=r1
ratios[0][1]=r2


nb_nbors_per_level[0][0]=nb1
nb_nbors_per_level[0][1]=nb2

print(ratios)
print(nb_nbors_per_level)
index.hnsw.efSearch = 10000
# 添加热点前
index.write_index_to_CSV()
index.out_indegrees()
D = np.empty((xq.shape[0], k), dtype=np.float32)
I = np.empty((xq.shape[0], k), dtype=np.int64)
D2 = np.empty((xq.shape[0], k), dtype=np.float32)
I2 = np.empty((xq.shape[0], k), dtype=np.int64)
index.search_with_ndis(xq.shape[0], faiss.swig_ptr(xq), k, 
        faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0],faiss.swig_ptr(I1))

# 添加热点
index.combine_index_with_hot_hubs_enhence(n,2,faiss.swig_ptr(ratios),0,0,
   faiss.swig_ptr(nb_nbors_per_level),faiss.swig_ptr(I1))
# index.search_from_hubs_to_nbs()
# 新的入度分布
index.out_indegrees2()
index.out_HubsIDs_and_its_reverse_nbs()


ef= []
r = []
t = []  
failed_queries = []
failed_queries2 = []

def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)
    D = np.empty((xq.shape[0], k), dtype=np.float32)
    I = np.empty((xq.shape[0], k), dtype=np.int64)
    D2 = np.empty((xq.shape[0], k), dtype=np.float32)
    I2 = np.empty((xq.shape[0], k), dtype=np.int64)
    t0 = time.time()

    index.search_with_ndis(xq.shape[0], faiss.swig_ptr(xq), k, 
        faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0],faiss.swig_ptr(I1))
    t1 = time.time()
    
    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    for i in range(len(I)):
        if I[i][0]!=gt[i][0]:
            failed_queries.append(i)
    for i in range(len(I2)):
        if I2[i][0]!=gt[i][0]:
            failed_queries2.append(i)
    ef.append(index.hnsw.efSearch)
    r.append(float(format(recall_at_1, '.4f')))
    t.append(float(format((t1-t0)*1000.0/nq, '.4f')))
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))
    return recall_at_1

# for efSearch in 50 ,100:# ,200,300,400,500,600,700,800,900,1000,2000,3000,4000 ,5000 ,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000,25000,30000,35000,40000,50000:
#     index.hnsw.efSearch = efSearch
#     if(evaluate(index) == 1):
#         break

index.hnsw.efSearch = 10000
evaluate(index)

index.level1_hubs()
index.indegreesDistr()

print(ef)
print(r)
print(t)

print(len(failed_queries))
print(failed_queries)

# print(len(failed_queries2))
# print(failed_queries2)

gt1 = []
for v in failed_queries:
    gt1.append(gt[v][0])
print(gt1)