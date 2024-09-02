import pandas as pd
from sklearn.decomposition import PCA

import plotnine
from plotnine import *
plotnine.options.figure_size = (18,9)


def create_problems_df(df, pmf_model):
    problems_df = (
        df
        .groupby('Problem_ID')
        .agg({'Status': 'mean', 'Problem_ID': 'size', 'Problem': 'first', 'Year': 'first'})
        .rename(columns = {'Status': "success", 'Problem_ID': 'size'})
        .assign(category = lambda d: d['Problem'].str[:-1])
    )

    pmf_problem_names = pmf_model.problem_vocab.get_itos()[1:]
    pmf_embeddings = pmf_model.problem_embedding.weight[1:].detach().numpy()
    embeddings_df = pd.DataFrame(
        pmf_embeddings,
        index = pmf_problem_names,
        columns = [f"weight_{i+1}" for i in range(pmf_embeddings.shape[1])]
    )

    problems_df = pd.concat([problems_df.loc[pmf_problem_names], embeddings_df], axis = 1)

    ### Create PCs
    components = min(pmf_embeddings.shape)
    pca = PCA(n_components = components)
    pcs = pca.fit_transform(pmf_embeddings)
    for pc in range(components):
        problems_df[f'PC{pc+1}'] = pcs[:, pc]

    return problems_df


def create_pc_scatter_plot(problems_df, variable = 'success'):
    return (
        problems_df
        .pipe(ggplot, aes(x = 'PC1', y = 'PC2', color = variable)) +
        geom_point() +
        labs(x = 'Principal Component 1', y = 'Principal Component 2', title = 'Problem Embeddings PCA Analysis') +
        theme_bw()
    )


if __name__ == '__main__':
    import torch
    from pmf import PMF

    REPLACEMENT_LEVELS = [25, 50, 100, 250, 500, 1000]
    LATENT_FACTORS = [1, 2, 3, 4]

    df = pd.read_csv('data/men_data.csv')

    problem_dfs = {}
    for replacement_level in REPLACEMENT_LEVELS:
        for num_factors in LATENT_FACTORS:
            pmf_model = torch.load(f"models/pmf/model_rl_{replacement_level}_d_{num_factors}_full_data.pth")
            pmf_model.eval()

            problems_df = create_problems_df(df, pmf_model)
            problem_dfs[f'{replacement_level}_{num_factors}'] = problems_df

            if num_factors > 1:
                for column in ['success', 'category', 'Year']:
                    p = create_pc_scatter_plot(problems_df, column)
                    p.save(f'results/problem_embeddings/scatter_plot/{column}/PCA_{replacement_level}_{num_factors}')
