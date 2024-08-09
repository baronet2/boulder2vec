import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import build_vocab_from_iterator

SEED = 42
NUM_EPOCHS = 100

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
    from preprocessing import create_split
    df = pd.read_csv('data/men_data.csv')
    REPLACEMENT_LEVELS = [500,1000]
    SEED = 42
    LATENT_FACTORS = 2

    train, test = create_split(df, SEED)

    for num_factors in np.arange(1, LATENT_FACTORS+1):
        for replacement_level in REPLACEMENT_LEVELS:
            #  Set Seed Here
            print(f'Commenced Training of PMF with Latent Factors: {num_factors} \t replacement_level: {replacement_level}')

            model = PMF(train, replacement_level, num_factors)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.1)

            train_model(model, train, criterion, optimizer, NUM_EPOCHS)
            torch.save(model, f"./models/model_{num_factors}_{replacement_level}.pth")