import matplotlib.pyplot as plt
import seaborn as sns
import plotnine
from plotnine import *

def create_correlation_matrix(df, pca=False):
    prefix = 'PC' if pca else 'weight'

    rows = [row for row in df.columns if row.startswith(prefix)]
    cols = ['coefs', 'size', 'success', 'height']
    df_corr = df.corr().loc[rows, cols]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, cbar=False)
    ax.set_xticklabels(["LR Coef", "# Climbs", "P(Success)", "Height"])

    plt.tight_layout()
    filename = 'figs/fig4_climber_pc_correlation_matrix.png'

    plt.savefig(filename)
    plt.close()

def create_scatter(df, variable, raw=False):
    if raw:
        return (
            df
            .pipe(ggplot, aes(x = 'weight_1', y = 'weight_2', color = variable)) +
            geom_point() +
            labs(x = 'Embedding 1', y = 'Embedding 2', title = 'Climber Embeddings PCA Analysis') +
            theme_bw()
        )
    else:
        return (
            df
            .pipe(ggplot, aes(x = 'PC1', y = 'PC2', color = variable)) +
            geom_point() +
            labs(x = 'Principal Component 1', y = 'Principal Component 2', title = 'Climber Embeddings PCA Analysis') +
            theme_bw()
        )

if __name__=="__main__":
    import pandas as pd
    import torch
    import pickle
    from pmf import PMF
    from lr import LogReg
    from climber_embeddings import create_climbers_df
    from problem_embeddings import create_problems_df

    REPLACEMENT_LEVEL = 100
    LATENT_FACTORS = 2

    df = pd.read_csv('data/men_data.csv')
    heights = pd.read_csv('data/climbers_heights.csv', index_col=0)
    df = df.merge(heights, on='Name', how='left')

    with open(f"models/lr/model_rl_{REPLACEMENT_LEVEL}_full_data.pkl", 'rb') as f:
            lr_model = pickle.load(f)

    pmf_model = torch.load(f"models/pmf/model_rl_{REPLACEMENT_LEVEL}_d_{LATENT_FACTORS}_full_data.pth")
    pmf_model.eval()

    climbers_df = create_climbers_df(df, lr_model, pmf_model)
    problems_df = create_problems_df(df, pmf_model)


    ### Figure 4 Climber PC Correlation Graph
    create_correlation_matrix(climbers_df, pca=True)
    # create_correlation_matrix(climbers_df, pca=False)

    ### Figure 5 Climber Raw Embeddings Scatter (Color = Success)
    fig = create_scatter(climbers_df, "success", False)
    fig.save('figs/fig5_climber_pc_scatter_success.png')
    fig = create_scatter(climbers_df, "success", True)
    fig.save('figs/fig5_climber_raw_scatter_success.png')

    ### Figure 6 Climber Raw Embeddings Scatter (Color = Height?)
    fig = create_scatter(climbers_df, "height", False)
    fig.save('figs/fig6_climber_pc_scatter_height.png')
    fig = create_scatter(climbers_df, "height", True)
    fig.save('figs/fig6_climber_raw_scatter_height.png')

    ### Figure 7 Problem Principal Component Scatter (Color = ProbType x Color = Success)
    fig = create_scatter(problems_df, "success", False)
    fig.save('figs/fig7_problem_pc_scatter_success.png')
    fig = create_scatter(problems_df, "success", True)
    fig.save('figs/fig7_problem_raw_scatter_success.png')