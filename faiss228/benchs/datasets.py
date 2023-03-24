# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
from __future__ import print_function
import sys
import time
import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    print(a.shape)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def bvecs_read(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def load_sift1M():
    dataset_basedir =  '/home/wanghongya/dataset/' 
    print("Loading sift1M...", end='', file=sys.stderr)
    xt = fvecs_read(dataset_basedir +"sift1M/sift_learn.fvecs")
    xb = fvecs_read(dataset_basedir +"sift1M/sift_base.fvecs")
    xq = fvecs_read(dataset_basedir +"sift1M/sift_query.fvecs")
    gt = ivecs_read(dataset_basedir +"sift1M/sift_groundtruth.ivecs")
    print("done", file=sys.stderr)
    xt
    return xb, xq, xt, gt

def load_sift10M():
    dataset_basedir =  '/home/wanghongya/sift1B/' 
    print("Loading sift10M...", end='', file=sys.stderr)
    
    xt = bvecs_read("/home/wanghongya/sift1B/learn.bvecs")
    xb = fvecs_read("/home/wanghongya/sift1B/sift10M.fvecs")
    xq = bvecs_read("/home/wanghongya/sift1B/queries.bvecs")
    gt = ivecs_read("/home/wanghongya/sift1B/gnd/idx_10M.ivecs")

    print("done", file=sys.stderr)

    return xb, xq, xt, gt

def load_gist():
    dataset_basedir =  '/home/bab/faiss-1.7.1/data/gist/' 
    print("Loading sift1M...", end='', file=sys.stderr)
    xt = fvecs_read(dataset_basedir + "gist_learn.fvecs")
    xb = fvecs_read(dataset_basedir + "gist_base.fvecs")
    xq = fvecs_read(dataset_basedir + "gist_query.fvecs")
    gt = ivecs_read(dataset_basedir + "gist_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt

def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls
'''

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import sys
import time
import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def ivecs_mmap(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')

def bvecs_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]
    
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

simdir = '/home/wanghongya/'

def load_sift1M():
    print("Loading sift1M...", end='', file=sys.stderr)
    # basedir = '/home/wanghongya/dataset/sift1M/'
    basedir = '/home/wanghongya/bab/sift1M/'
    xt = fvecs_read(basedir + "sift_base.fvecs")
    xb = fvecs_read(basedir + "sift_base.fvecs")
    xq = fvecs_read(basedir + "sift_query.fvecs")
    gt = ivecs_read(basedir + "sift_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt

def load_sift10K():
    print("Loading sift10K...", end='', file=sys.stderr)
    basedir = simdir + '1M/sift10K/'
    basedir = '/mnt/d/github/sift10K/'
    xt = fvecs_read(basedir + "siftsmall_learn.fvecs")
    xb = fvecs_read(basedir + "siftsmall_base.fvecs")
    xq = fvecs_read(basedir + "siftsmall_query.fvecs")
    gt = ivecs_read(basedir + "siftsmall_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt

def load_bigann():
    # print("Loading bigann...", end='', file=sys.stderr)
    # basedir = '/mnt/d/github/sift1B/'
    basedir = simdir + 'sift1B/'

    dbsize = 10
    xb = bvecs_mmap(basedir + '1milliard.p1.siftbin')
    xq = bvecs_mmap(basedir + 'queries.bvecs')
    xt = bvecs_mmap(basedir + 'learn.bvecs')
    # trim xb to correct size
    xb = sanitize(xb[:dbsize * 1000 * 1000])
    xt = sanitize(xt[:250000])
    xq = sanitize(xq)
    gt = ivecs_read(basedir + 'gnd/idx_%dM.ivecs' % dbsize)

    return xb, xq, xt, gt


def load_deep():
    # simdir = '/home/wanghongya/'
    print("load deep.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/home/wanghongya/deep1b/'
    
    dbsize = 10
    xb = mmap_fvecs(basedir + 'deep1B_base.fvecs')
    xq = mmap_fvecs(basedir + 'deep1B_queries.fvecs')
    xt = mmap_fvecs(basedir + 'deep1B_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:dbsize * 1000 * 1000])
    xt = sanitize(xt[:500000])
    xq = sanitize(xq[:10000])

    gt = ivecs_read('/home/wanghongya/dataset/deep10M_groundtruth.ivecs')
    # gt = ivecs_read(basedir + 'deep%dM_groundtruth.ivecs' % dbsize)

  
    return xb, xq, xt, gt


# gist 1M
def load_gist():
    # simdir = '/home/wanghongya/'
    print("load gist.....")
    # basedir = simdir + 'deep1b/'
    basedir = '/home/wanghongya/dataset/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'gist_base.fvecs')
    xq = mmap_fvecs(basedir + 'gist_query.fvecs')
    xt = mmap_fvecs(basedir + 'gist_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:dbsize * 1000 * 1000])
    xt = sanitize(xt[:100000])
    xq = sanitize(xq[:1000])
    gt = ivecs_read(basedir + 'gist_groundtruth.ivecs')

    return xb, xq, xt, gt


# glove 1M
def load_glove():
    # simdir = '/home/wanghongya/'
    # basedir = simdir + 'deep1b/'

    basedir = '/home/wanghongya/dataset/glove/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'glove_base.fvecs')
    xq = mmap_fvecs(basedir + 'glove_query.fvecs')
    xt = mmap_fvecs(basedir + 'glove_query.fvecs')
    # trim xb to correct size
    print(xb.shape[0])
    xb = sanitize(xb[:xb.shape[0]+1])
    xt = sanitize(xt[:1000])
    xq = sanitize(xq[:10000])
    gt = ivecs_read(basedir + 'glove_groundtruth.ivecs')
    print(xq.shape[0])
    return xb, xq, xt, gt
    
def load_audio():
    print("Loading audio...", end='', file=sys.stderr)
    basedir = '/home/wanghongya/bab/audio/'
    xt = fvecs_read(basedir + "audio_base.fvecs")
    xb = fvecs_read(basedir + "audio_base.fvecs")
    xq = fvecs_read(basedir + "audio_query.fvecs")
    gt = ivecs_read(basedir + "audio_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_sun():
    print("Loading sun...", end='', file=sys.stderr)
    basedir = '/home/wanghongya/dataset/sun/'
    xt = fvecs_read(basedir + "sun_base.fvecs")
    xb = fvecs_read(basedir + "sun_base.fvecs")
    xq = fvecs_read(basedir + "sun_query.fvecs")
    gt = ivecs_read(basedir + "sun_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_random():
    print("Loading random...", end='', file=sys.stderr)
    basedir = '/home/wanghongya/dataset/random/'
    xt = fvecs_read(basedir + "random_base.fvecs")
    xb = fvecs_read(basedir + "random_base.fvecs")
    xq = fvecs_read(basedir + "random_query.fvecs")
    gt = ivecs_read(basedir + "random_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_imageNet():
    print("Loading sun...", end='', file=sys.stderr)
    basedir = '/home/wanghongya/dataset/imageNet/'
    xt = fvecs_read(basedir + "imageNet_base.fvecs")
    xb = fvecs_read(basedir + "imageNet_base.fvecs")
    xq = fvecs_read(basedir + "imageNet_query.fvecs")
    gt = ivecs_read(basedir + "imageNet_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls
