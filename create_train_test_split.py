import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read in Data
df_M = pd.read_csv('data/men_data_1000.csv')
df_W = pd.read_csv('data/women_data_1000.csv')

SEED = 42

def create_split(df, seed):
    df['stratify_group'] = df[['Competition', 'Level', 'problem']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=seed, stratify=df['stratify_group'])

    train_df = train_df.drop(columns=['stratify_group'])
    test_df = test_df.drop(columns=['stratify_group'])

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# Create the train/test splits
train_M, test_M = create_split(df_M, SEED)
train_W, test_W = create_split(df_W, SEED)

# Write train/test M+W to
train_M.to_csv('./data/split/train_M.csv', index=False)
test_M.to_csv('./data/split/test_M.csv', index=False)

train_W.to_csv('./data/split/train_W.csv', index=False)
test_W.to_csv('./data/split/test_W.csv', index=False)