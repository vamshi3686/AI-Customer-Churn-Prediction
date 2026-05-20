import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from logging_code import setup_logging
logger=setup_logging('label_encoder')
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder


def one_hot_encoder(x_train,x_test):
    try:
        logger.info(f'total columns in x_train={x_train.shape}')
        logger.info(f'total columns in x_test={x_test.shape}')
        logger.info(f'before one hot encoding x_train column names :{x_train.columns}')
        logger.info(f'before one hot encoding x_test column names :{x_test.columns}')
        one=OneHotEncoder(drop='first')
        one.fit(x_train[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod','sim_column']])
        values_train=one.transform(x_train[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies',
       'PaperlessBilling', 'PaymentMethod','sim_column']]).toarray()
        values_test = one.transform(x_test[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                              'TechSupport', 'StreamingTV', 'StreamingMovies',
                                              'PaperlessBilling', 'PaymentMethod','sim_column']]).toarray()
        t1=pd.DataFrame(values_train)
        t2=pd.DataFrame(values_test)
        t1.columns=one.get_feature_names_out()
        t2.columns=one.get_feature_names_out()
        x_train.reset_index(drop=True,inplace=True)
        x_test.reset_index(drop=True,inplace=True)
        t1.reset_index(drop=True,inplace=True)
        t2.reset_index(drop=True,inplace=True)
        x_train=pd.concat([x_train,t1],axis=1)
        x_test=pd.concat([x_test,t2],axis=1)
        x_train=x_train.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                              'TechSupport', 'StreamingTV', 'StreamingMovies',
                                              'PaperlessBilling', 'PaymentMethod','sim_column'],axis=1)
        x_test=x_test.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                              'TechSupport', 'StreamingTV', 'StreamingMovies',
                                              'PaperlessBilling', 'PaymentMethod','sim_column'],axis=1)

        ordinal=OrdinalEncoder()
        ordinal.fit(x_train[['Contract']])
        results_x_train=ordinal.transform(x_train[['Contract']])
        results_x_test=ordinal.transform(x_test[['Contract']])
        t3=pd.DataFrame(results_x_train)
        t4=pd.DataFrame(results_x_test)
        t3.columns=ordinal.get_feature_names_out()+'_od'
        t4.columns=ordinal.get_feature_names_out()+'_od'
        t3.reset_index(drop=True, inplace=True)
        t4.reset_index(drop=True, inplace=True)
        x_train = pd.concat([x_train, t3], axis=1)
        x_test = pd.concat([x_test, t4], axis=1)
        x_train=x_train.drop(['Contract'],axis=1)
        x_test=x_test.drop(['Contract'],axis=1)
        logger.info(f'---------------------------------------------------------------------')
        logger.info(f'after one hot encoding x_train column names :{x_train.columns}')
        logger.info(f'after one hot encoding x_test column names :{x_test.columns}')
        logger.info(f'total columns in x_train={x_train.shape}')
        logger.info(f'total columns in x_test={x_test.shape}')
        return x_train,x_test





    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error_type:{error_type},error_msg:{error_msg},error_line:{error_line}')

