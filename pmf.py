import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext #; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import KFold
import random 
import os 
import pandas as pd
from preprocessing import create_split

# Create PMF Model
class PMF(nn.Module):
    def __init__(self, df, replacement_level, num_factors):
        super(PMF, self).__init__()
        self.climber_vocab = build_vocab_from_iterator([df['Name'].values], min_freq=replacement_level, specials=['other'])
        self.climber_vocab.set_default_index(self.climber_vocab['other'])
        self.problem_vocab = build_vocab_from_iterator([df['Problem_ID'].values], min_freq=num_factors, specials=['Problem'])
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
    #torch.cuda.manual_seed(seed)  # uncomment if ur using gpu, im using mac so idc

    os.environ["PYTHONHASHSEED"] = str(seed) # for setting seed in hash operations in libraries 
    print(f"Seed set as {seed}")

# Training function
def train_model(model, df, criterion, optimizer, num_epochs): # warning about df redefinition 
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(df['Name'].values, df['Problem_ID'].values)
        loss = criterion(predictions, torch.tensor(df['Status'].values, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    return model 


if __name__ == '__main__':
    SEED = 42
    NUM_EPOCHS = 5
    K_FOLDS = 5
    REPLACEMENT_LEVELS = [500,1000]
    LATENT_FACTORS = 2
    set_seed(SEED)

    df = pd.read_csv('data/men_data.csv')

    train, test = create_split(df, SEED)

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    for num_factors in np.arange(1, LATENT_FACTORS+1):
        for replacement_level in REPLACEMENT_LEVELS:
            print(f'Commenced Training of PMF with Latent Factors: {num_factors} \t replacement_level: {replacement_level}')
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train)):
                print(f"Fold {fold + 1}/{K_FOLDS}")
                train_fold = train.iloc[train_idx]
                val_fold = train.iloc[val_idx]

                model = PMF(train_fold, replacement_level, num_factors)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.1)

                # Train the model
                trained_model = train_model(model, train_fold, criterion, optimizer, NUM_EPOCHS)

                # Save the model for this fold
                print(f"Saving PMF model for replacement level {replacement_level}, latent factor {num_factors}, fold {fold + 1}")
                torch.save({'model_state_dict': trained_model.state_dict(),'val_indices': val_idx}, f"models/pmf/model_{num_factors}_{replacement_level}_fold_{fold+1}.pth")
            print(f"Completed training for Latent Factors: {num_factors}, replacement_level: {replacement_level}")
