import numpy as np
import torchtext #; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

SEED = 42
K_FOLDS = 5

class LogReg():
    def __init__(self, df, replacement_level):
        self.climber_vocab = build_vocab_from_iterator([df['Name'].values], min_freq=replacement_level, specials=['other'])
        self.climber_vocab.set_default_index(self.climber_vocab['other'])

        X = self.create_X(df)
        y = df['Status'].values

        self.lr = LogisticRegression(max_iter=1000)

    def create_X(self, df):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        return encoder.fit_transform(np.array([self.climber_vocab[name] for name in df['Name']]).reshape(-1,1))

    def predict(self, df):
        return self.lr.predict_proba(self.create_X(df))[:, 1]

    def fit(self, X, y): #separate out the fit 
        self.lr.fit(X, y)

def train_model(df, replacement_level): # warning about df, replacement_levle redefinition 
    model = LogReg(df, replacement_level)
    X_train = model.create_X(df)
    y_train = df['Status'].values
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    import pandas as pd
    import pickle
    from preprocessing import create_split
    df = pd.read_csv('data/men_data.csv')
    REPLACEMENT_LEVELS = [500,1000]

    train, test = create_split(df, SEED)

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    for replacement_level in REPLACEMENT_LEVELS:
        #  Set Seed Here
        print(f'Commenced Training of LogReg with replacement_level: {replacement_level}')

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train)):
            train_df = train.iloc[train_idx]
            val_df = train.iloc[val_idx]

            # Train model for this fold 
            model = train_model(train_df, replacement_level)

            # Save the model and validation indices for this fold
            print(f"Saving LogReg Model with replacement_level: {replacement_level}, fold {fold + 1}")
            with open(f"models/lr/model_{replacement_level}_fold_{fold+1}.pkl", 'wb') as f:
                pickle.dump({
                    'model': model,
                    'val_indices': val_idx
                }, f)

        print(f'Completed Training of LogReg with replacement_level: {replacement_level}')