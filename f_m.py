import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from logging_code import setup_logging
logger=setup_logging('f_m')
import sys
from sklearn.feature_selection import mutual_info_classif



def feature_engineering(x_train,x_test,y_train):
    try:
        logger.info(f'total columns in x_train={x_train.shape}')
        logger.info(f'total columns in x_test={x_test.shape}')
        logger.info(f'x_train.columns {x_train.columns}')
        mi=mutual_info_classif(x_train,y_train,random_state=42)
        mi_scores=pd.Series(mi,index=x_train.columns)
        good_columns = mi_scores[mi_scores > 0.03].index
        x_train=x_train[good_columns]
        x_test=x_test[good_columns]
        logger.info(f'after performing feature engineering x_train_columns= {x_train.columns}')
        logger.info(f'x_test_columns={x_test.columns}')
        logger.info(f'total columns in x_train={x_train.shape}')
        logger.info(f'total columns in x_test={x_test.shape}')
        return x_train,x_test,y_train




    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error type{error_type},error msg {error_msg} ,error_line {error_line}')