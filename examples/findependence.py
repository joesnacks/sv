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
    print(entropy_rate(returns.values))
    # need some discretisation...


