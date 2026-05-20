import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from logging_code import setup_logging
logger=setup_logging('missing_data')

def  handling_data(x_train,x_test):
    logger.info(f'total columns in x_train={x_train.shape}')
    logger.info(f'total columns in x_test={x_test.shape}')
    logger.info('Before handling data')
    logger.info(f'total no of null values in x_trian {x_train.isnull().sum()}')
    logger.info(f'total no of null values in x_trian {x_test.isnull().sum()}')
    x_train['TotalCharges']=x_train['TotalCharges'].ffill()
    x_test['TotalCharges']=x_test['TotalCharges'].ffill()
    logger.info('After handling data')
    logger.info(f'total no of null values in x_trian {x_train.isnull().sum()}')
    logger.info(f'total no of null values in x_trian {x_test.isnull().sum()}')
    x_train['TotalCharges'] = x_train['TotalCharges'].ffill()
    logger.info(f'total columns in x_train={x_train.shape}')
    logger.info(f'total columns in x_test={x_test.shape}')

    return x_train,x_test