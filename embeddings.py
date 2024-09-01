
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def get_accuracy(df, model, names, mask):
    accuracies = []
    for name in reversed(names):
        if mask:
            df_cut = df.loc[df.Name == name]
        else:
            df_cut = df.loc[df.Problem_ID == name]

        y_true = df_cut['Status'].values
        y_pred = model.predict(df_cut)
        y_pred_binary = np.round(y_pred)

        accuracies.append(accuracy_score(y_true, y_pred_binary))
    return accuracies

def get_counts(df, names, mask):
    counts = []
    for name in names:
        if mask:
            df_cut = df.loc[df.Name == name]
        else:
            df_cut = df.loc[df.Problem_ID == name]
        counts.append(df_cut.shape[0])
    return counts

def get_success(df, names, mask):
    success = []
    for name in names:
        if mask:
            df_cut = df.loc[df.Name == name]
        else:
            df_cut = df.loc[df.Problem_ID == name]
        success.append(df_cut['Status'].mean())
    return success

def create_climbers_df():
    climbers = {}

    for replacement_level in REPLACEMENT_LEVELS:
        with open(f"models/lr/model_rl_{replacement_level}_full_data.pkl", 'rb') as f:
            lr_model = pickle.load(f)

        ### Handle Climbers (LR)
        lr_climber_names = lr_model.climber_vocab.get_itos()[1:]
        lr_climbers = pd.DataFrame({
            "coefs": lr_model.lr.coef_.flatten().tolist()[1:-3],
        }, index=lr_climber_names)

        for num_factors in LATENT_FACTORS:

            ### Handle Climbers (PMF)
            pmf_model = torch.load(f"models/pmf/model_rl_{replacement_level}_d_{num_factors}_full_data.pth")
            pmf_model.eval()

            pmf_climber_names = pmf_model.climber_vocab.get_itos()[1:]
            pmf_climbers = pd.DataFrame({
                "weights": pmf_model.climber_embedding.weight.data.numpy().tolist()[1:],
                "pmf_accuracy": get_accuracy(df, pmf_model, pmf_climber_names, True),
                "size": get_counts(df, pmf_climber_names, True),
                "success": get_success(df, pmf_climber_names, True),
                # "height": ,
                # "weight": ,
                # "bmi"
            }, index=pmf_climber_names)

            weights = pmf_climbers['weights'].apply(pd.Series)
            pmf_climbers = pd.concat([pmf_climbers[['pmf_accuracy','size', 'success']], weights], axis=1)
            pmf_climbers.columns = ['pmf_accuracy', 'size', 'success'] + [f'weight_{i+1}' for i in range(weights.shape[1])]

            ### Merge Climbers
            lr_pmf_climbers = pd.merge(lr_climbers, pmf_climbers, left_index=True, right_index=True, how='outer')
            lr_pmf_climbers.dropna(inplace=True)

            ### Create PCs
            embeddings = lr_pmf_climbers[[row for row in lr_pmf_climbers.columns if row.startswith('w')]].values
            components = min(embeddings.shape[0], embeddings.shape[1])
            pca = PCA(n_components=components)
            pcs = pca.fit_transform(embeddings)

            for pc in range(components):
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

    for replacement_level in REPLACEMENT_LEVELS:
        for num_factors in LATENT_FACTORS:

            ### Handle Climbers (PMF)
            pmf_model = torch.load(f"models/pmf/model_rl_{replacement_level}_d_{num_factors}_full_data.pth")
            pmf_model.eval()

            pmf_problem_ids = pmf_model.problem_vocab.get_itos()[1:]
            pmf_problems = pd.DataFrame({
                "weights": pmf_model.problem_embedding.weight.data.numpy().tolist()[1:],
                "pmf_accuracy": get_accuracy(df, pmf_model, pmf_problem_ids, False),
                "size": get_counts(df, pmf_problem_ids, False),
                "success": get_success(df, pmf_problem_ids, False),
                "year": [float(problem.split('_')[0]) for problem in pmf_problem_ids],
                "round": [problem.split('_')[-2] for problem in pmf_problem_ids],
                "category": [problem.split('_')[-1][:-1] for problem in pmf_problem_ids],
            }, index=pmf_problem_ids)

            weights = pmf_problems['weights'].apply(pd.Series)
            pmf_problems = pd.concat([pmf_problems[['pmf_accuracy', 'size', 'success', 'year', "round", "category"]], weights], axis=1)
            pmf_problems.columns = ['pmf_accuracy','size', 'success', 'year', "round", "category"] + [f'weight_{i+1}' for i in range(weights.shape[1])]

            ### Merge Climbers
            c_probs = pmf_problems

            # ### Create PCs
            embeddings = c_probs[[row for row in c_probs.columns if row.startswith('w')]].values
            components = min(embeddings.shape[0], embeddings.shape[1])
            pca = PCA(n_components=components)
            pcs = pca.fit_transform(embeddings)

            for pc in range(components):
                c_probs[f'PC{pc+1}'] = pcs[:, pc]

            problems[f'{replacement_level}_{num_factors}'] = c_probs

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