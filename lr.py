import numpy as np
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

SEED = 42
NUM_EPOCHS = 100

# Create PMF Model
class LogReg():
    def __init__(self, df, replacement_level):
        self.climber_vocab = build_vocab_from_iterator([df['Name'].values], min_freq=replacement_level, specials=['other'])
        self.climber_vocab.set_default_index(self.climber_vocab['other'])

        X = self.create_X(df)

        print(X.shape)
        y = df['Status'].values

        self.lr = LogisticRegression(max_iter=1000).fit(X, y)

    def create_X(self, df):
        encoder = OneHotEncoder(drop='first', sparse_output=False)

        return encoder.fit_transform(np.array([self.climber_vocab[name] for name in df['Name']]).reshape(-1,1))

    def predict(self, df):
        return self.lr.predict_proba(self.create_X(df))[:, 1]

if __name__ == '__main__':
    import pandas as pd
    import pickle
    from preprocessing import create_split
    df = pd.read_csv('data/men_data.csv')
    REPLACEMENT_LEVELS = [500,1000]
    SEED = 42

    train, test = create_split(df, SEED)

    for replacement_level in REPLACEMENT_LEVELS:
        #  Set Seed Here
        print(f'Commenced Training of LogReg with replacement_level: {replacement_level}')
        model = LogReg(train, replacement_level) # Creates and Trains Model in Init
        with open(f"models/lr/model_{replacement_level}.pkl",'wb') as f:
            pickle.dump(model, f)