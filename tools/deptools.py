# tools for analysing the dependence in sequences
import numpy as np
import lzma
import bz2

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
    
if __name__=='__main__':
    # tests of entopy_rate
    x = bytearray(np.random.randint(0,2,size=10000).astype(np.uint32))
    print(x[:12])
    for method in ['lzma','bz2']:
        print(method, entropy_rate(x, method=method))
    # tests of block_shuffle
    x = 1+np.arange(12)
    for blocksize in [2,2,3,3,4,4,5,5,5]:
        print(blocksize, block_shuffle(x, blocksize))
    