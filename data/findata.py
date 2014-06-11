# functions for reading in financial data
# read data from .csv.gz format
# --------------------------------------------
import os
import pandas as pd
import gzip

def data_loader(name_string):
    datafile_path = os.path.join('data',name_string)
    with gzip.open(datafile_path, 'r') as f:
        return(pd.read_csv(f))

def load_spday():
    return(data_loader('SP_H_19271230-20130228.csv.gz'))

def load_spminute():
    return(data_loader('SP_ID1_20120824-20130301.csv.gz'))

def load_sptick():
    return(data_loader('SP_TICK_20120824-20130301.csv.gz'))

def load_psiday():
    return(data_loader('PSI20_H_19921231-20130228.csv.gz'))

def load_psiminute():
    return(data_loader('PSI20_ID1_20120824-20130301.csv.gz'))

def load_psitick():
    return(data_loader('PSI20_TICK_20120824-20130301.csv.gz'))