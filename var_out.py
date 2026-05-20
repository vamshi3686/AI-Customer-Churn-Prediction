import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from logging_code import setup_logging
logger=setup_logging('var_out')
import sys
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from scipy.stats import boxcox

scaler = StandardScaler()


def variable_outliers(x_train, x_test):
    try:
        logger.info(f'before column names:{x_train.columns}')

        # ✅ Transformations
        x_train['TotalCharges_var'] = np.sqrt(x_train['TotalCharges'])
        x_test['TotalCharges_var'] = np.sqrt(x_test['TotalCharges'])
        x_train.drop(['TotalCharges'], axis=1, inplace=True)
        x_test.drop(['TotalCharges'], axis=1, inplace=True)

        x_train['MonthlyCharges_boxcox'], _ = boxcox(x_train['MonthlyCharges'])
        x_test['MonthlyCharges_boxcox'], _ = boxcox(x_test['MonthlyCharges'])
        x_train.drop(['MonthlyCharges'], axis=1, inplace=True)
        x_test.drop(['MonthlyCharges'], axis=1, inplace=True)

        # ⚠️ Fix here (important)
        x_train['tenure_scaled'] = scaler.fit_transform(x_train[['tenure']])
        x_test['tenure_scaled'] = scaler.transform(x_test[['tenure']])

        x_train.drop(['tenure'], axis=1, inplace=True)
        x_test.drop(['tenure'], axis=1, inplace=True)

        logger.info(f'after transformation columns:{x_train.columns}')


        cols_to_trim = ['TotalCharges_var', 'MonthlyCharges_boxcox', 'tenure_scaled']

        for col in cols_to_trim:
            Q1 = x_train[col].quantile(0.25)
            Q3 = x_train[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            # Train
            x_train[col + '_trim'] = np.where(
                x_train[col] > upper_limit, upper_limit,
                np.where(x_train[col] < lower_limit, lower_limit, x_train[col])
            )

            # Test (use SAME limits)
            x_test[col + '_trim'] = np.where(
                x_test[col] > upper_limit, upper_limit,
                np.where(x_test[col] < lower_limit, lower_limit, x_test[col])
            )

            # Drop original transformed column
            x_train.drop([col], axis=1, inplace=True)
            x_test.drop([col], axis=1, inplace=True)

        # =====================================

        logger.info(f"After Train Column Name : {x_train.columns}")
        logger.info(f"After Test Column Name : {x_test.columns}")

        return x_train, x_test

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error type {error_type}, error msg {error_msg}, error_line {error_line}')