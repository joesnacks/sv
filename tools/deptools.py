# tools for analysing the dependence in sequences
import numpy as np
import lzma
import bz2
import pandas as pd
import scipy.stats
from statsmodels.tsa.arima_process import arma_generate_sample
import os
import matplotlib.pyplot as plt

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
    rx = rx.astype(np.uint32)
    n = len(rx)
    if k>2**16:
        print('warning: in deptools.discretise: discretisation specifies ' +  
        '%.0f > 2**16 states, but converted to 32 bit integer; ' % k +
        'integer overflow')
    return(((rx*k) // (n+1)).astype(np.uint32))

def dependogram(x, blocksizes=None, nshuffles=100, nbins=256, plot=False, method='lzma'):
    # test stat for dependence at or beyond m lags against m
    # x is a scalar time series
    # output is a named array of p-values; plot output of simulated distribution
    # of the test stat under the null is optional
    # method can be 'lzma' or 'bz2'
    shuffled_entropies = {}
    if blocksizes is None:
        kmax = np.int(np.log10(len(x)/3))
        blocksizes = [10**b for b in range(0,kmax,1+kmax//6)]
        print('using default block sizes %s' % repr(blocksizes))
    for blocksize in blocksizes:
        print(' doing block size %s' % repr(blocksize))
        shuffled_entropies[blocksize] = []
        for i in range(nshuffles):
            xs = block_shuffle(discretise(x, nbins), blocksize)
            shuffled_entropies[blocksize].append(entropy_rate(bytearray(xs), method=method))
    print('shuffled entropy rates:')
    for blocksize in blocksizes:
        print('blocksize: %.0f mean: %.4f sd: %.6f' % (blocksize,
        np.mean(shuffled_entropies[blocksize]),
        np.std(shuffled_entropies[blocksize])))
    # print('x[0]: ',x[0],'xb[:4]: ',bytearray(x)[:4],'list(xb)[:4]',
    #     list(bytearray(x))[:4])
    unshuffled_entropy_rate = entropy_rate(bytearray(
        discretise(x, nbins)), method=method)
    print('unshuffled entropy rate %.4f' % unshuffled_entropy_rate)
    teststat = pd.DataFrame(shuffled_entropies)-unshuffled_entropy_rate
    rejections = (teststat>0).sum() / teststat.shape[0] #rejection frequencies
    pvalues = 1-rejections
    if plot:
        print('pvalues:\n',pvalues)
        teststat.boxplot()
        plt.title('test stat for dependence at or beyond m lags against m')
        plt.show()
    return(pvalues)

if __name__=='__main__':
    # tests of entopy_rate
    print('\ntests of entropy_rate')
    print(' byte sequence (256 states)')
    x = bytes(os.urandom(10000))
    print(x[:10])
    for method in ['lzma','bz2']:
        print(method, entropy_rate(x, method=method))
    print(' byte sequence, 8 bit encoding [1 byte per byte; 1 bit per bit]')
    x = bytearray(np.random.randint(0,2**8,size=10000).astype(np.uint8))
    print(x[:12])
    for method in ['lzma','bz2']:
        print(method, entropy_rate(x, method=method))
    print(' byte sequence, 16 bit encoding ' + 
    '[.5 byte per byte; 8 bits per 16 bits]')
    x = bytearray(np.random.randint(0,2**8,size=10000).astype(np.uint16))
    print(x[:12])
    for method in ['lzma','bz2']:
        print(method, entropy_rate(x, method=method))
    print(' byte sequence, 32 bit encoding ' + 
    '[1 byte per 4 bytes; 8 bits per 32 bits]')
    x = bytearray(np.random.randint(0,2**8,size=10000).astype(np.uint32))
    print(x[:12])
    for method in ['lzma','bz2']:
        print(method, entropy_rate(x, method=method))
    print(' 2**4 states, 32 bit encoding [4 bits per 32 bits]')
    x = bytearray(np.random.randint(0,2**4,size=10000).astype(np.uint32))
    print(x[:12])
    for method in ['lzma','bz2']:
        print(method, entropy_rate(x, method=method))
    print(' 2**2 states, 32 bit encoding [2 bits per 32 bits]')
    x = bytearray(np.random.randint(0,2**2,size=10000).astype(np.uint32))
    print(x[:12])
    for method in ['lzma','bz2']:
        print(method, entropy_rate(x, method=method))
    print(' 1 bit sequence (2 states), 32 bit encoding [1 bit per 32 bits]')
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
    # print('\ntests of discretise + entropy_rate')
    # x = np.random.randint(0,2**8,size=100000).astype(np.uint32)
    # xb = bytearray(x)
    # print('x[:4]: ',x[:4],'xb[:4]: ',xb[:4],'list(xb)[:4]',list(xb)[:4])
    # for method in ['lzma','bz2']:
    #     print(method,'undiscretised',entropy_rate(bytearray(x)))
    #     for discretisation_level in [2**j for j in [1,2,3,5,8,17]]:
    #         xd = discretise(x,discretisation_level)
    #         print(method,'discretised',discretisation_level,
    #         entropy_rate(bytearray(xd), method=method))
    # tests of dependogram
    print('\ntests of dependogram')
    x = arma_generate_sample([1,.5],[1],nsample=10000).astype(np.float64)
    xb = bytearray(x)
    print('x[:5]',x[:5])
    # print(dependogram(x, blocksizes=[1,2,3,100], nshuffles=100, nbins=4, plot=True))
    # dependogram(x, nshuffles=100, nbins=8, plot=True)
    # dependogram(x, nshuffles=100, nbins=6, plot=True)
    dependogram(x, nshuffles=100, nbins=2, plot=False, method='lzma')
    dependogram(x, nshuffles=100, nbins=2, plot=True, method='bz2')