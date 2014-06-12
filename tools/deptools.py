# tools for analysing the dependence in sequences
import numpy as np
import lzma
import bz2
import pandas as pd
import scipy.stats

def entropy_rate(x, method='lzma'):
    if method=='lzma':
        return(float(len(lzma.compress(x, format=lzma.FORMAT_ALONE, 
            preset=9))) / float(len(x)))
    elif method=='bz2':
        return(float(len(bz2.compress(x))) / float(len(x)))
    
def block_shuffle(x, blocksize):
    nblock = len(x)//blocksize
    # random number of starting values to truncate
    # (the rest will truncate from the end)
    offset = np.random.randint(0,len(x)%blocksize+1)
    block_index = np.arange(nblock)
    np.random.shuffle(block_index)
    block_starting_indices = np.resize(block_index, (blocksize,nblock)
        ).transpose()*blocksize+offset
    block_middle_indices = np.resize(np.arange(blocksize), (nblock,blocksize))
    shuffle_indices = np.hstack(block_starting_indices+block_middle_indices)
    return(x[shuffle_indices])

def discretise(x, k):
    # discretise the sequence x into k blocks
    if type(x) in [pd.core.series.Series, pd.Series]:
        rx = x.rank(method='max')
    else:
        rx = scipy.stats.rankdata(x, method='max')
    n = len(rx)
    return((rx*k) // (n+1))
    
if __name__=='__main__':
    # tests of entopy_rate
    print('\ntests of entropy_rate')
    x = bytearray(np.random.randint(0,2,size=10000).astype(np.uint32))
    print(x[:12])
    for method in ['lzma','bz2']:
        print(method, entropy_rate(x, method=method))
    # tests of block_shuffle
    print('\ntests of block_shuffle')
    x = 1+np.arange(12)
    for blocksize in [2,2,3,3,4,4,5,5,5]:
        print(blocksize, block_shuffle(x, blocksize))
    # tests of discretise
    print('\ntests of discretise')
    x = 1+np.arange(12)
    np.random.shuffle(x)
    print('x:', x)
    for k in [1,2,3,4,12]:
        print(k, discretise(x, k))
    # tests of discretise + entropy_rate
    print('\ntests of discretise + entropy_rate')
    x = np.random.randint(0,2^8,size=100000).astype(np.uint32)
    xb = bytearray(x)
    print('x[0]: ',x[0],'xb[:4]: ',xb[:4],'list(xb)[:4]',list(xb)[:4])
    for method in ['lzma','bz2']:
        for discretisation_level in [2**j for j in [1,2,3,5,8]]:
            print(method,discretisation_level,entropy_rate(bytearray(discretise(x,discretisation_level))))