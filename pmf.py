import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext 
from torchtext.vocab import build_vocab_from_iterator
import random 
import os 

# Create PMF Model
class PMF(nn.Module):
    def __init__(self, df, replacement_level, num_factors):
        super(PMF, self).__init__()
        self.climber_vocab = build_vocab_from_iterator([df['Name'].values], min_freq=replacement_level, specials=['other'])
        self.climber_vocab.set_default_index(self.climber_vocab['other'])
        self.problem_vocab = build_vocab_from_iterator([df['Problem_ID'].values], min_freq=10, specials=['Problem'])
        self.problem_vocab.set_default_index(self.problem_vocab['Problem'])
        self.climber_embedding = nn.Embedding(len(self.climber_vocab), num_factors)
        self.problem_embedding = nn.Embedding(len(self.problem_vocab), num_factors)

    def forward(self, climber_names, problem_ids):
        climber_indices = torch.tensor([self.climber_vocab[name] for name in climber_names])
        problem_indices = torch.tensor([self.problem_vocab[problem] for problem in problem_ids])

        climber_vector = self.climber_embedding(climber_indices)
        problem_vector = self.problem_embedding(problem_indices)

        dot_product = (climber_vector * problem_vector).sum(dim=1)
        outputs = torch.sigmoid(dot_product)
        return outputs

    def predict(self, df):
        self.eval()
        with torch.no_grad():
            predictions = self(df['Name'].values, df['Problem_ID'].values)
        return predictions

# Setting seed function
def set_seed(seed=42):
    '''
    Modified the function here: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy 
    Sets seeds for numpy, pytorch, python.random and PYTHONHASHSEED.
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # uncomment if ur using gpu / set cuda 
    os.environ["PYTHONHASHSEED"] = str(seed) # for setting seed in hash operations in libraries 

# Training function
def train_model(model, df, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(df['Name'].values, df['Problem_ID'].values)
        loss = criterion(predictions, torch.tensor(df['Status'].values, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import KFold

    torch.device("mps" if torch.backends.mps.is_available() and os.environ.get("USE_MPS") else "cpu")
    
    SEED = 42
    K_FOLDS = 5
    REPLACEMENT_LEVELS = [25, 50, 100, 250, 500, 1000]
    LATENT_FACTORS = [1, 2, 3, 4, 5, 6]
    NUM_EPOCHS = 1000

    df = pd.read_csv('data/men_data.csv')

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        train = df.iloc[train_idx]

        for num_factors in LATENT_FACTORS:
            for replacement_level in REPLACEMENT_LEVELS:
                print(f"Training PMF model for fold {fold}, replacement level {replacement_level}, dimension {num_factors}")
                set_seed(SEED)
                model = PMF(train, replacement_level, num_factors)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.1)

                train_model(model, train, criterion, optimizer, NUM_EPOCHS)
                torch.save(model, f"models/pmf/model_rl_{replacement_level}_d_{num_factors}_fold_{fold}.pth")

    for num_factors in LATENT_FACTORS:
        for replacement_level in REPLACEMENT_LEVELS:
            print(f"Training PMF model for full data, replacement level {replacement_level}, dimension {num_factors}")
            set_seed(SEED)
            model = PMF(df, replacement_level, num_factors)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.1)

            train_model(model, df, criterion, optimizer, NUM_EPOCHS)
            torch.save(model, f"models/pmf/model_rl_{replacement_level}_d_{num_factors}_full_data.pth")