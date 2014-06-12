# example application to financial data
from data.findata import *
from tools.deptools import *
import os
import pylab

if __name__=='__main__':
    print('running in directory:', os.getcwd())
    print('loading daily prices')
    spd = load_spday()
    price = spd.set_index('date')['PX_LAST']
    returns = np.log(price).diff()
    print('head of returns data:',returns.head())
    print('shape of returns data:',returns.shape)
    print(returns.dtype)
    print(type(returns.values))
    print(returns.values.dtype)
    print('estimating entropy rate of discretised sequences')
    for k in [2**j for j in [1,2,3,5,8]]:
        print('lzma',k,entropy_rate(bytearray(discretise(returns.values, k))))
    print('lzma entropy rate estimates of shuffled sequences in 256 bins')
    nshuffles = 100
    shuffled_entropies = {}
    blocksizes = [1,10,100,1000,5000,10000,15000]
    for blocksize in blocksizes:
        print(' doing block size %s' % repr(blocksize))
        shuffled_entropies[blocksize] = []
        for i in range(nshuffles):
            x = block_shuffle(discretise(returns.values, 256), blocksize)
            shuffled_entropies[blocksize].append(entropy_rate(bytearray(x)))
    for blocksize in blocksizes:
        print('blocksize: %.0f mean: %.4f sd: %.6f' % (blocksize,
        np.mean(shuffled_entropies[blocksize]), 
        np.std(shuffled_entropies[blocksize])))
    print('x[0]: ',x[0],'xb[:4]: ',bytearray(x)[:4],'list(xb)[:4]',
        list(bytearray(x))[:4])
    unshuffled_entropy_rate = entropy_rate(bytearray(
        discretise(returns.values, 256)))
    print('unshuffled entropy rate', unshuffled_entropy_rate)
#     print('... making boxplot; compare to unshuffled entropy rate %.4f' % 
#         unshuffled_entropy_rate)
#     pd.DataFrame(shuffled_entropies).boxplot()
#     pylab.title('entropy rate estimates of shuffled sequences by block size')
    teststat = pd.DataFrame(shuffled_entropies)-unshuffled_entropy_rate
    print((teststat>0).sum() / teststat.shape[0]) #rejection frequencies
    teststat.boxplot()
    pylab.title('test stat for dependence at or beyond m lags against m')

