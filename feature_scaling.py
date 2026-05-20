import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import seaborn as sns
import logging
from logging_code import setup_logging
logger = setup_logging("feature_scaling")
import sys
from sklearn.preprocessing import StandardScaler # z_score
from all_models import common
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
def fs(X_train , y_train , X_test , y_test):
    try:
        logger.info(f"Training data independent size : {X_train.shape}")
        logger.info(f"Training data depdepent size : {y_train.shape}")
        logger.info(f"Testing data independent size : {X_test.shape}")
        logger.info(f"Testing data dependent size : {y_test.shape}")
        logger.info(f"before : {X_train.head(1)}")

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)

        with open('standar_scaler.pkl','wb') as f:
            pickle.dump(sc,f)

        logger.info(f"{X_train_sc}")
        common(X_train_sc,y_train,X_test_sc,y_test)
        '''
        best_model.fit(X_train_sc,y_train)
        logger.info(f'Test accuracy:{accuracy_score(y_test,best_model.predict(X_test_sc))}')


        #finding best parameters for the best model
        parameter_grid = {
            "penalty": ["l1", "l2"],
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "saga"],
            "max_iter": [200, 500],
            "class_weight": [None, "balanced"]
        }
        lr=best_model
        grid=GridSearchCV(estimator=lr,param_grid=parameter_grid,scoring='roc_auc',cv=5,n_jobs=-1)
        grid.fit(X_train_sc, y_train)
        logger.info(f'Best Parameters:, {grid.best_params_}')
        logger.info(f'Best AUC:, {grid.best_score_}')'''






        logger.info(f'best parameters for logistic regression : C: 10, class_weight: None, max_iter: 200, penalty: l2, solver: saga')
        reg = LogisticRegression(
            C=10,
            class_weight=None,
            max_iter=200,
            penalty='l2',
            solver='saga'
        )
        reg.fit(X_train_sc,y_train) # Training completed
        logger.info(f"Test Accuracy : {accuracy_score(y_test,reg.predict(X_test_sc))}")
        logger.info(f"Test Confusion Matrix : {confusion_matrix(y_test,reg.predict(X_test_sc))}")
        logger.info(f"Classification report : {classification_report(y_test,reg.predict(X_test_sc))}")

        with open('Model.pkl','wb') as t:
            pickle.dump(reg,t)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")