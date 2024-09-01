
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def get_accuracy_helper(df, model):
    y_true = df['Status'].values
    y_pred = model.predict(df)
    y_pred_binary = np.round(y_pred)
    return accuracy_score(y_true, y_pred_binary)

def create_climbers_df():
    climbers = {}

    grouped_df = df.groupby('Name')
    size_by_name = grouped_df.size()
    success_by_name = grouped_df['Status'].mean()

    for replacement_level in REPLACEMENT_LEVELS:

        ### Handle the LR Model
        with open(f"models/lr/model_rl_{replacement_level}_full_data.pkl", 'rb') as f:
            lr_model = pickle.load(f)

        lr_climber_names = lr_model.climber_vocab.get_itos()[1:]
        lr_coefs = lr_model.lr.coef_.flatten()[1:-3] ### may adjust after other edits

        lr_climbers = pd.DataFrame({
            "coefs": lr_coefs,
        }, index=lr_climber_names)

        for num_factors in LATENT_FACTORS:

            ### Handle the PMF Model
            pmf_model = torch.load(f"models/pmf/model_rl_{replacement_level}_d_{num_factors}_full_data.pth")
            pmf_model.eval()

            pmf_climber_names = pmf_model.climber_vocab.get_itos()[1:]
            weights = pmf_model.climber_embedding.weight.data.numpy()[1:]

            ### Create Metrics
            pmf_climbers = pd.DataFrame({
                "weights": list(weights),
                "pmf_accuracy": grouped_df.apply(lambda group: get_accuracy_helper(group, pmf_model)).reindex(pmf_climber_names),
                "size": size_by_name.reindex(pmf_climber_names),
                "success": success_by_name.reindex(pmf_climber_names),
            }, index=pmf_climber_names)

            weight_columns = pd.DataFrame(pmf_climbers['weights'].tolist(), index=pmf_climbers.index)
            weight_columns.columns = [f'weight_{i+1}' for i in range(weight_columns.shape[1])]
            pmf_climbers = pd.concat([pmf_climbers.drop(columns=['weights']), weight_columns], axis=1)

            ### Merge LR and PMF
            lr_pmf_climbers = pd.merge(lr_climbers, pmf_climbers, left_index=True, right_index=True, how='outer').dropna()

            ### Create PCs
            embeddings = weight_columns.values
            pca = PCA(n_components=min(embeddings.shape))
            pcs = pca.fit_transform(embeddings)
            for pc in range(pcs.shape[1]):
                lr_pmf_climbers[f'PC{pc+1}'] = pcs[:, pc]

            climbers[f'{replacement_level}_{num_factors}'] = lr_pmf_climbers

    return climbers

def create_correlation_matrices(df, path, raw=True):
    for replacement_level in REPLACEMENT_LEVELS:

        fig, axs = plt.subplots(nrows=1, ncols=len(LATENT_FACTORS), figsize=(28, 8))

        for num_factors in LATENT_FACTORS:
            df_athletes = df[f'{replacement_level}_{num_factors}']

            prefix = 'weight' if raw else 'PC'
            rows = [row for row in df_athletes.columns if row.startswith(prefix)]
            cols = ['coefs','pmf_accuracy','size', 'success']
            df_corr = df_athletes.corr().loc[rows, cols]
            sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=axs[num_factors-1])

        plt.suptitle(f'Correlation Matrices at RL: {replacement_level}', fontsize=16)
        plt.tight_layout()

        filename = f'figs/{path}/{"corr_raw" if raw else "corr_pc"}/CMatrix_{replacement_level}_{"raw" if raw else "pc"}'
        plt.savefig(filename)
        plt.close(fig)

def create_pc_figures(df, variable, path, datatype):
    for replacement_level in REPLACEMENT_LEVELS:

        fig, axs = plt.subplots(nrows=1, ncols=len(LATENT_FACTORS[1:]), figsize=(28, 8))

        for i, num_factors in enumerate(LATENT_FACTORS[1:]):
            df_climbers = df[f'{replacement_level}_{num_factors}']

            if datatype == 'discrete':
                unique_categories = df_climbers[variable].unique()
                colormap = plt.get_cmap('tab10')
                category_colors = {category: colormap(i) for i, category in enumerate(unique_categories)}

                colors = df_climbers[variable].map(category_colors)

            elif datatype == 'continuous':
                norm = plt.Normalize(df_climbers[variable].min(), df_climbers[variable].max())
                colormap = plt.get_cmap('viridis')

                colors = colormap(norm(df_climbers[variable]))

            axs[i].scatter(df_climbers['PC1'], df_climbers['PC2'], alpha=0.5, c=colors)

            if datatype == 'discrete':
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colormap(i), markersize=10) for i in range(len(unique_categories))]
                axs[i].legend(handles, unique_categories, title=variable, loc="best")
            elif datatype == 'continuous':
                cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=axs[i])
                cbar.set_label(variable)

            axs[i].set_title(f'PCA for {num_factors} Factors')
            axs[i].set_xlabel('Principal Component 1')
            axs[i].set_ylabel('Principal Component 2')

        plt.suptitle(f'PCA at RL: {replacement_level}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        filename = f'figs/{path}/PCA/{variable}/PCA_{replacement_level}_{variable}'
        plt.savefig(filename)
        plt.close(fig)

def create_problems_df():
    problems = {}

    grouped_df = df.groupby('Problem_ID')
    size_by_problem_id = grouped_df.size()
    success_by_problem_id = grouped_df['Status'].mean()

    for replacement_level in REPLACEMENT_LEVELS:
        for num_factors in LATENT_FACTORS:

            ### Handle the PMF model
            pmf_model = torch.load(f"models/pmf/model_rl_{replacement_level}_d_{num_factors}_full_data.pth")
            pmf_model.eval()

            pmf_problem_ids = pmf_model.problem_vocab.get_itos()[1:]
            weights = pmf_model.problem_embedding.weight.data.numpy()[1:]

            ### Create Metrics
            pmf_problems = pd.DataFrame({
                "weights": list(weights),
                "pmf_accuracy": grouped_df.apply(lambda group: get_accuracy_helper(group, pmf_model)).reindex(pmf_problem_ids),
                "size": size_by_problem_id.reindex(pmf_problem_ids),
                "success": success_by_problem_id.reindex(pmf_problem_ids),
                "year": [float(problem.split('_')[0]) for problem in pmf_problem_ids],
                "round": [problem.split('_')[-2] for problem in pmf_problem_ids],
                "category": [problem.split('_')[-1][:-1] for problem in pmf_problem_ids],
            }, index=pmf_problem_ids)

            weight_columns = pd.DataFrame(pmf_problems['weights'].tolist(), index=pmf_problems.index)
            weight_columns.columns = [f'weight_{i+1}' for i in range(weight_columns.shape[1])]
            pmf_problems = pd.concat([pmf_problems.drop(columns=['weights']), weight_columns], axis=1)

            ### Create PCs
            embeddings = weight_columns.values
            pca = PCA(n_components=min(embeddings.shape))
            pcs = pca.fit_transform(embeddings)
            for pc in range(pcs.shape[1]):
                pmf_problems[f'PC{pc+1}'] = pcs[:, pc]

            problems[f'{replacement_level}_{num_factors}'] = pmf_problems

    return problems

def create_geom_smooth(df, path):
    for replacement_level in REPLACEMENT_LEVELS:

        fig, axs = plt.subplots(nrows=1, ncols=len(LATENT_FACTORS), figsize=(48, 8))

        for i, num_factors in enumerate(LATENT_FACTORS):
            df_climbers = df[f'{replacement_level}_{num_factors}']

            sns.regplot(x='year', y=f'PC{i+1}', data=df_climbers, lowess=True, ax=axs[i])

            axs[i].set_title(f'GeomPlot for PC{1}')
            axs[i].set_xlabel('Year')
            axs[i].set_ylabel('Embedding')

        plt.suptitle(f'GeomSmooth at RL: {replacement_level}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        filename = f'figs/{path}/geom/geom_{replacement_level}.png'
        plt.savefig(filename)
        plt.close(fig)

if __name__ == '__main__':
    import torch
    import pickle
    import numpy as np
    import pandas as pd

    from pmf import PMF
    from lr import LogReg

    SEED = 42
    REPLACEMENT_LEVELS = [25, 50, 100, 250, 500, 1000]
    LATENT_FACTORS = [1, 2, 3, 4]

    df = pd.read_csv('data/men_data.csv')

    ### Create Climber / Problem DataFrames
    climbers = create_climbers_df()
    problems = create_problems_df()

    # Create Climber Correlation Matrices
    create_correlation_matrices(climbers, 'climbers')
    create_correlation_matrices(climbers, 'climbers', raw=False)

    # Create PCA
    create_pc_figures(climbers, 'size', 'climbers', 'continuous')
    create_pc_figures(climbers, 'success', 'climbers', 'continuous')

    create_pc_figures(problems, 'category', 'problems', 'discrete')
    create_pc_figures(problems, 'round', 'problems', 'discrete')
    create_pc_figures(problems, 'success', 'problems', 'continuous')
    create_pc_figures(problems, 'year', 'problems', 'continuous')