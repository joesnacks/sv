# example application to financial data
from data.findata import *
from tools.deptools import *
import os

if __name__=='__main__':
    print(os.getcwd())
    spd = load_spday()
    price = spd.set_index('date')['PX_LAST']
    returns = np.log(price).diff()
    print(returns.head())
    print(returns.dtype)
    print(type(returns.values))
    print(returns.values.dtype)
    for k in [2**j for j in [1,2,3,5,8]]:
        print('lzma',k,entropy_rate(bytearray(discretise(returns.values, k))))



