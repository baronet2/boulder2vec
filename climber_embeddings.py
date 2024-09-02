import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import plotnine
from plotnine import *
plotnine.options.figure_size = (18,9)


def create_climbers_df(df, lr_model, pmf_model):
    climbers_df = (
        df
        .groupby('Name')
        .agg({'Status': 'mean', 'Problem_ID': 'size', 'Height': 'mean'})
        .rename(columns = {'Status': "success", 'Problem_ID': 'size', 'Height': 'height'})
    )

    lr_climber_names = lr_model.climber_vocab.get_itos()[1:]
    lr_climbers = pd.DataFrame(
        {"coefs": lr_model.lr.coef_.flatten().tolist()[1:len(lr_climber_names) + 1]},
        index = lr_climber_names
    )

    pmf_climber_names = pmf_model.climber_vocab.get_itos()[1:]
    pmf_embeddings = pmf_model.climber_embedding.weight[1:].detach().numpy()
    embeddings_df = pd.DataFrame(
        pmf_embeddings,
        index = pmf_climber_names,
        columns = [f"weight_{i+1}" for i in range(pmf_embeddings.shape[1])]
    )

    climbers_df = pd.concat([climbers_df.loc[pmf_climber_names], lr_climbers.loc[pmf_climber_names], embeddings_df], axis = 1)

    ### Create PCs
    components = min(pmf_embeddings.shape)
    pca = PCA(n_components = components)
    pcs = pca.fit_transform(pmf_embeddings)
    for pc in range(components):
        climbers_df[f'PC{pc+1}'] = pcs[:, pc]

    return climbers_df


def create_correlation_matrices(dfs, pca=False):
    for replacement_level in REPLACEMENT_LEVELS:

        fig, axs = plt.subplots(nrows=1, ncols=len(LATENT_FACTORS), figsize=(28, 8))

        for num_factors in LATENT_FACTORS:
            df = dfs[f'{replacement_level}_{num_factors}']

            prefix = 'PC' if pca else 'weight'
            rows = [row for row in df.columns if row.startswith(prefix)]
            cols = ['coefs', 'size', 'success', 'height']
            df_corr = df.corr().loc[rows, cols]
            sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=axs[num_factors-1])

        plt.suptitle(f'Correlation Matrices at RL: {replacement_level}', fontsize=16)
        plt.tight_layout()

        filename = f'results/climber_embeddings/{"corr_pca" if pca else "corr_raw"}/{replacement_level}.png'
        plt.savefig(filename)
        plt.close(fig)


def create_pc_scatter_plot(climbers_df, color_variable = 'success'):
    return (
        climbers_df
        .pipe(ggplot, aes(x = 'PC1', y = 'PC2', color = color_variable)) +
        geom_point() +
        labs(x = 'Principal Component 1', y = 'Principal Component 2', title = 'Climber Embeddings PCA Analysis') +
        theme_bw()
    )


if __name__ == '__main__':
    import torch
    import pickle
    from pmf import PMF
    from lr import LogReg

    REPLACEMENT_LEVELS = [25, 50, 100, 250, 500, 1000]
    LATENT_FACTORS = [1, 2, 3, 4]

    df = pd.read_csv('data/men_data.csv')
    ### Add heights from collected
    heights = pd.read_csv('data/climbers_heights.csv', index_col=1)
    df = df.merge(heights, on='Name', how='left')

    climber_dfs = {}
    for replacement_level in REPLACEMENT_LEVELS:
        with open(f"models/lr/model_rl_{replacement_level}_full_data.pkl", 'rb') as f:
            lr_model = pickle.load(f)

        for num_factors in LATENT_FACTORS:
            pmf_model = torch.load(f"models/pmf/model_rl_{replacement_level}_d_{num_factors}_full_data.pth")
            pmf_model.eval()

            climbers_df = create_climbers_df(df, lr_model, pmf_model)
            climber_dfs[f'{replacement_level}_{num_factors}'] = climbers_df

            if num_factors > 1:
                for column in ['success', 'height']:
                    p = create_pc_scatter_plot(climbers_df, column)
                    p.save(f'results/climber_embeddings/scatter_plot/{column}/PCA_{replacement_level}_{num_factors}.png')

    create_correlation_matrices(climber_dfs)
    create_correlation_matrices(climber_dfs, pca=True)
