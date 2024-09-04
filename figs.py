import matplotlib.pyplot as plt
import seaborn as sns
import plotnine
from plotnine import *
from matplotlib.colors import LinearSegmentedColormap


def create_correlation_matrix(df):
    rows = [row for row in df.columns if row.startswith('PC')]
    cols = ['coefs', 'size', 'success', 'height']
    df_corr = df.corr().loc[rows, cols]

    plt.figure(figsize=(10, 4))
    ax = sns.heatmap(
        df_corr,
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        cbar=False,
        annot_kws={"size": 16}
    )

    ax.set_xticklabels(["LR Coef", "# Climbs", "P(Success)", "Height"], fontsize=16)
    ax.set_yticklabels(['PC1', 'PC2'], fontsize=16)

    plt.tight_layout()
    filename = 'figs/climber_pc_correlation_matrix.png'

    plt.savefig(filename)

def get_cmap():
    return LinearSegmentedColormap.from_list("gradient", ["#ff914d", "#B42D2D", "#8c52ff", "#1f4b8e"])

def create_scatter(df, variable, figsize, position, raw=False):
    cmap = get_cmap()

    if pd.api.types.is_numeric_dtype(df[variable]):
        color_scale = scale_color_gradientn(
            colors=[cmap(i/100) for i in range(101)],
            limits=(0, 1)
        )
        legend_guide = None
    else:
        color_scale = None
        legend_guide = guides(color=guide_legend(override_aes={'shape': 's', 'size': 10, 'stroke': 0}))

    if variable == "success":
        df['P(Success)'] = df[variable]
        variable = 'P(Success)'
    elif variable == "category":
        df['Problem Type'] = df[variable]
        variable = 'Problem Type'

    if raw:
        plot = (
            ggplot(df, aes(x='weight_1', y='weight_2', color=variable)) +
            geom_point(size=4) +
            (color_scale if color_scale else None) +
            labs(x='Embedding 1', y='Embedding 2') +
            theme_bw() +
            theme(
                figure_size=figsize,
                axis_title_x=element_text(size=25),
                axis_title_y=element_text(size=25),
                axis_text=element_text(size=25),
                legend_position=position,
                legend_title=element_text(size=25),
                legend_text=element_text(size=12)
            ) +
            (legend_guide if legend_guide else None)
        )
    else:
        plot = (
            ggplot(df, aes(x='PC1', y='PC2', color=variable)) +
            geom_point(size=4) +
            (color_scale if color_scale else None) +
            labs(x='Principal Component 1', y='Principal Component 2') +
            theme_bw() +
            theme(
                figure_size=figsize,
                axis_title_x=element_text(size=25),
                axis_title_y=element_text(size=25),
                axis_text=element_text(size=25),
                legend_position=position,
                legend_title=element_text(size=25),
                legend_text=element_text(size=12)
            ) +
            (legend_guide if legend_guide else None)
        )
    return plot

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
    create_correlation_matrix(climbers_df)

    ### Figure 5 Climber Raw Embeddings Scatter (Color = Success)
    fig = create_scatter(climbers_df, "success", (18, 10), (0.975, 0.05), False)
    fig.save('figs/climber_pc_scatter_success.png')

    ### Figure 6 Problem Principal Component Scatter (Color = ProbType x Color = Success)
    fig = create_scatter(problems_df, "category", (10, 8), (0.975, 0.95), False)
    fig.save('figs/problem_pc_category_success.png')
    fig = create_scatter(problems_df, "success", (10, 8), (0.975, 0.95), False)
    fig.save('figs/problem_pc_scatter_success.png')