# example application to financial data
import os
os.chdir('c:/users/gsher/dropbox/git/sv')
import pylab
from data.findata import *
from tools.deptools import *


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
    dependogram(returns.values, blocksizes=[1,10,100,1000,5000,10000], method='lzma', plot=True)
    # dependogram(returns.values, blocksizes=[1,10,100,1000,5000], method='bz2', plot=True)

