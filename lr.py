import numpy as np
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

class LogReg():
    def __init__(self, df, replacement_level):
        self.climber_vocab = build_vocab_from_iterator([df['Name'].values], min_freq=replacement_level, specials=['other'])
        self.climber_vocab.set_default_index(self.climber_vocab['other'])

        X = self.create_X(df)
        y = df['Status'].values

        self.lr = LogisticRegression(max_iter=1000, random_state=SEED).fit(X, y)

    def create_X(self, df):
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        return encoder.fit_transform(np.array([self.climber_vocab[name] for name in df['Name']]).reshape(-1,1))

    def predict(self, df):
        return self.lr.predict_proba(self.create_X(df))[:, 1]


if __name__ == '__main__':
    import pandas as pd
    import pickle
    from sklearn.model_selection import KFold

    SEED = 42
    K_FOLDS = 5
    REPLACEMENT_LEVELS = [25, 50, 100, 250, 500, 1000]

    df = pd.read_csv('data/men_data.csv')

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        train = df.iloc[train_idx]
        print(fold, train.shape, train['Name'].iloc[0])

        for replacement_level in REPLACEMENT_LEVELS:
            print(f"Training LR model for fold {fold }, replacement level {replacement_level}")
            model = LogReg(train, replacement_level)
            with open(f"models/lr/model_rl_{replacement_level}_fold_{fold}.pkl", 'wb') as f:
                pickle.dump(model, f)