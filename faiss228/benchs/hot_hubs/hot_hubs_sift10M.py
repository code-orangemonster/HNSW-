import time
import sys
import numpy as np
sys.path.append('/home/wanghongya/bab/faiss150/python')
sys.path.append('/home/wanghongya/bab/faiss150/benchs')
import faiss
import util
import os


from datasets import load_glove
from datasets import load_audio
from datasets import load_sift1M
from datasets import load_bigann

k = int(sys.argv[1])
m = int(sys.argv[2])
efcon = int(sys.argv[3])
hot_ratio = float(sys.argv[4])
print("-------------阿弥陀佛--------------------")
print("k={}--m={}--efcon={}--hot_ratio={}".format(k,m,efcon,hot_ratio))
print("-------------阿弥陀佛--------------------")

print("load data")
xb, xq, xt, gt = load_bigann()
# xb, xq, xt, gt = load_audio()

nq, d = xq.shape

print("Testing HNSW Flat")

index1 = faiss.IndexHNSWFlat(d, m)
index1.hnsw.efConstruction = efcon
index1.hnsw.search_mode = 0
print("add")
# to see progress
index1.verbose = True
index1.add(xb)
index1.ratio=hot_ratio
index1.hnsw.search_bounded_queue = False


index = faiss.IndexHNSWFlat(d, m)
index.hnsw.efConstruction = 0
index.verbose = True
index.hnsw.search_bounded_queue = False
index.hnsw.search_mode = 0
index.ratio = hot_ratio
## 创建index的storage，
index.add_with_hot_hubs(xb.shape[0],faiss.swig_ptr(xb))

n=xq.shape[0]

index.combine_index_with_hot_hubs(index,index1,xb.shape[0])


def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)
    D = np.empty((n, k), dtype=np.float32)
    I = np.empty((n, k), dtype=np.int64)
    t0 = time.time()
    index.search_with_hot_hubs(xq.shape[0], faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0])
    t1 = time.time()
    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))
    return recall_at_1

for efSearch in 100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000:
    index.hnsw.efSearch = efSearch
    if evaluate(index)==1:
        break

    
