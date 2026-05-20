import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
from logging_code import setup_logging
logger = setup_logging("all_models")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve

def train_models(x_train, y_train):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "LR": LogisticRegression(),
        "DT": DecisionTreeClassifier(criterion='entropy'),
        "RF": RandomForestClassifier(n_estimators=5, criterion='entropy'),
        "ADA": AdaBoostClassifier(estimator=LogisticRegression(), n_estimators=5),
        "GB": GradientBoostingClassifier(n_estimators=5),
        "XGB": XGBClassifier(n_estimators=5, use_label_encoder=False, eval_metric='logloss')
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        logger.info(f"{name} trained successfully")

    return trained_models


def evaluate_models(models, x_test, y_test):
    scores = {}
    roc_data = {}

    for name, model in models.items():
        y_prob = model.predict_proba(x_test)[:, 1]

        auc_score = roc_auc_score(y_test, y_prob)
        scores[name] = auc_score

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)

        logger.info(f"{name} AUC Score: {auc_score}")

    #  Best model
    best_model_name = max(scores, key=scores.get)
    best_score = scores[best_model_name]
    best_model = models[best_model_name]

    logger.info(f"\n BEST MODEL: {best_model_name}")
    logger.info(f"BEST AUC: {best_score}")

    return best_model, scores, roc_data


def plot_roc(roc_data):
    plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], "k--")

    for name, (fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, label=name)

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("All Models ROC Curve")
    plt.legend()

    plt.savefig("roc_curve.png")
    plt.close()


def common(X_train, y_train, X_test, y_test):
    try:
        logger.info("Training all models...")

        models = train_models(X_train, y_train)

        logger.info("Evaluating models...")

        best_model, scores, roc_data = evaluate_models(models, X_test, y_test)

        logger.info("Plotting ROC curves...")

        plot_roc(roc_data)

        return best_model

    except Exception as e:
        logger.error(f"Error: {e}")