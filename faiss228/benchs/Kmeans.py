# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:06:09 2021

@author: Sean
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/wanghongya/bab/faiss150/python')
sys.path.append('/home/wanghongya/bab/faiss150/benchs')
import heapq
import _thread
import faiss
import struct
import util

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def bvecs_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

###############################################################
# 读取fbin文件
def read_base_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=(nvecs-990000000) * dim, dtype=np.float32, 
                          offset=990000000 * 4 * dim)
    return arr.reshape(nvecs-990000000, dim)

def read_fbin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def read_i8bin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int8, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def read_u8bin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.uint8, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def read_i32bin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def write_ibin(filename, vecs):
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('int8').flatten().tofile(f)

def write_u8bin(filename, vecs):
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('uint8').flatten().tofile(f)



from datasets import load_random
from datasets import load_glove
from datasets import load_audio
from datasets import load_sift1M
from datasets import load_deep
from datasets import load_gist
from datasets import load_imageNet

xb, xq, xt, gt = load_sift1M()

n, d = xb.shape
kmeans = faiss.Kmeans(d, 100)#聚类中心个数   10M用15000，1M用1000
print("training")
kmeans.train(xb)
print("training done.")

D = np.empty((xq.shape[0], 1), dtype=np.float32)
I = np.empty((xq.shape[0], 1), dtype=np.int64)
D,I = kmeans.assign(xb)

# 将I传过去 

# faiss.swig_ptr(D)
print(D[0:200])

# 将质心存为fvecs文件
fvecs_write("/home/wanghongya/bab/kmeans/1.fvecs",kmeans.centroids)
# 读取质心
cent = fvecs_read("/home/wanghongya/bab/kmeans/1.fvecs")


print(type(cent),type(D))
print("cent:",len(cent))
print("cent:",cent)
