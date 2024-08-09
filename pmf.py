import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import build_vocab_from_iterator # Make sure to use Torch 2.3.0, Torchtext 0.18.0 => Add to README LAter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Static Variables
SEED = 42
NUM_EPOCHS = 10

# Create Stratified Train/Test Splits
def create_split(df, seed):
    train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=seed, stratify=df['Problem_ID'])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# Create PMF Model
class PMF(nn.Module):
    def __init__(self, df, replacement_level, num_factors):
        super(PMF, self).__init__()
        self.climber_vocab = build_vocab_from_iterator(df['Name'], min_freq=replacement_level, specials=['other'])
        self.climber_vocab.set_default_index(self.climber_vocab['other'])
        self.problem_vocab = build_vocab_from_iterator(df['Problem_ID'], min_freq=num_factors, specials=['Problem'])
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

# Training function
def train_model(model, train, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        # names, problems = torch.tensor(train['Name'].values, dtype=torch.long), torch.tensor(train['Problem_ID'].values, dtype=torch.long)
        predictions = model(train['Name'].values, train['Problem_ID'].values)
        # predictions = model(train['Name'].values, train['Problem_ID'].values)
        loss = criterion(predictions, torch.tensor(train['Status'].values, dtype=torch.float32))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Full Pipeline
def train_pmf(df, replacement_levels, latent_factors):
    train, test = create_split(df, SEED)

    for num_factors in np.arange(1, latent_factors):
        for replacement_level in replacement_levels:
            print(f'Commenced Training of PMF with Latent Factors: {num_factors} \t replacement_level: {replacement_level}')

            model = PMF(train, replacement_level, num_factors)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.1)

            train_model(model, train, criterion, optimizer, NUM_EPOCHS)
            torch.save(model, f"./models/model_{num_factors}_{replacement_level}.pth")

# Prediction function
def predict(model, climber_indices, problem_indices):
    model.eval()
    with torch.no_grad():
        predictions = model(climber_indices, problem_indices)
    return predictions


# Import model analysis libraries
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# def analyze_model(model, test):
#     test_values   = torch.tensor(test['Status'].values, dtype=torch.float32)
#     test_climbers = torch.tensor(test['Climber_code'].values, dtype=torch.long)
#     test_problems = torch.tensor(test['Stratification_code'].values, dtype=torch.long)

#     test_predictions = predict(model, test_climbers, test_problems)
#     test_predictions_binary = np.round(test_predictions.numpy())

#     accuracy = accuracy_score(test_values.numpy(), test_predictions_binary)
#     log_loss = log_loss(test_values.numpy(), test_predictions)
#     brier_score = brier_score_loss(test_values.numpy(), test_predictions)
#     f1_score = f1_score(test_values.numpy(), test_predictions_binary)
#     roc_score = roc_auc_score(test_values.numpy(), test_predictions)

#     print(f'Test Accuracy: {accuracy}')
#     print(f'Log Loss: {log_loss}')
#     print(f'Brier Score: {brier_score}')
#     print(f'F1 Score: {f1_score}')
#     print(f'ROC Score: {roc_score}')

# # Extract and map embeddings to names
# def extract_embeddings(model, encoder, embedding_layer, column_name):
#     embeddings = embedding_layer.weight.data.cpu().numpy()
#     names = encoder.inverse_transform(np.arange(len(embeddings)))
#     df = pd.DataFrame(embeddings, index=names)
#     df.index.name = column_name
#     return df









# Preprocess for training
# def preprocess(df, replacement_level):
#     # Create TrainTest Splits
#     train, test = create_split(df, STRATIFICATION, SEED)

#     # Create Replacement Level => Vocab
#     # climber_counts = train['Name'].value_counts()
#     # train[f'Climber_{replacement_level}'] = np.where(train['Name'].map(climber_counts) >= replacement_level, train['Name'], 'Other')
#     # train[f'Climber_{replacement_level}'] = pd.Categorical(train[f'Climber_{replacement_level}'], categories=['Other'] + [x for x in train[f'Climber_{replacement_level}'].unique() if x != 'Other'])

#     # Convert to Climber, Stratification into Category
#     train['Climber_code'] = train[f'Climber_{replacement_level}'].astype('category').cat.codes
#     train['Stratification_code'] = train['Stratification'].astype('category').cat.codes

#     # Convert to Tensor => Vocab
#     train_y   = torch.tensor(train['Status'].values, dtype=torch.float32)
#     train_climbers = torch.tensor(train['Climber_code'].values, dtype=torch.long)
#     train_problems = torch.tensor(train['Stratification_code'].values, dtype=torch.long)

#     return train, test, train_y, train_climbers, train_problems