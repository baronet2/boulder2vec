from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import torch

def analyze_model(model, df):
    y_true = df['Status'].values
    y_pred = model.predict(df)
    y_pred_binary = np.round(y_pred)

    results = {
        "Accuracy": accuracy_score(y_true, y_pred_binary),
        "Log Loss": log_loss(y_true, y_pred),
        "Brier Score": brier_score_loss(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred_binary),
        "ROC Score": roc_auc_score(y_true, y_pred)
    }
    return results

if __name__ == '__main__':
    import pandas as pd
    import torch
    from preprocessing import create_split
    from pmf import PMF
    from lr import LogReg
    import pickle

    data = pd.read_csv('data/men_data.csv')
    REPLACEMENT_LEVELS = [500,1000]
    SEED = 42
    K_FOLDS = 5

    train, test = create_split(data, SEED)

    pmf_model = torch.load('models/pmf/model_1_500.pth')
    pmf_model.eval()

    print("PMF Model:")
    print(f"Train: {analyze_model(pmf_model, train)}")
    print(f"Test: {analyze_model(pmf_model, test)}")

    with open('models/lr/model_500.pkl', 'rb') as f:
        lr_model = pickle.load(f)

    print("LogReg Model:")
    print(f"Train: {analyze_model(lr_model, train)}")
    print(f"Test: {analyze_model(lr_model, test)}")
