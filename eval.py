import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from plotnine import * 


def get_data_for_plot(metrics_long):
    return (
        metrics_long
        .groupby(['model_type', 'replacement_level', 'num_factors', 'dataset', 'metric'])
        ['value'].agg(['mean', 'std', 'size'])
        .reset_index()
        .assign(
            se = lambda d: d['std'] / d['size'].apply(np.sqrt),
            mean_err_min = lambda d: d['mean'] - 1.96 * d.se,
            mean_err_max = lambda d: d['mean'] + 1.96 * d.se,
            rl = lambda d: pd.Categorical(d['replacement_level']).rename_categories(lambda x: f"$N = {x}$"),
            metric = lambda d: pd.Categorical(d['metric'], ordered = True, categories = ['Accuracy', 'F1 Score', 'ROC Score', 'Log Loss', 'Brier Score'])
        )
    )


def evaluate_model(model, df):
    y_true = df['Status'].values
    y_pred = model.predict(df)
    y_pred_binary = np.round(y_pred)

    results = {
        "Accuracy": accuracy_score(y_true, y_pred_binary),
        "Log Loss": log_loss(y_true, y_pred),
        "Brier Score": brier_score_loss(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred_binary),
        "ROC Score": roc_auc_score(y_true, y_pred)
    }
    return results


def produce_results_plot(plot_data):
    return (
        plot_data
        .pipe(ggplot, aes(x = "num_factors", y = "mean", color = "dataset", linetype = "model_type")) +
        # Add horizontal lines for logistic regression
        # geom_hline(aes(yintercept = "mean", color = "dataset"), data = plot_data.query("model_type == 'lr'")) +
        # Add lines and points for PMF
        geom_line() +
        geom_errorbar(aes(ymin = "mean_err_min", ymax = "mean_err_max"), width = 0.15, size = 0.75) +
        geom_point(size = 2, data = plot_data.query("model_type == 'pmf'")) +
        facet_grid("metric ~ rl", scales = "free_y") +
        labs(x = "# of PMF Latent Factors ($d$)", y = "Metric Value", color = "Dataset", linetype = "Model Type", size = '48pt') +
        scale_color_manual(
            values=["#4E79A7", "#E15759"],  # Viridis colors
            labels=["Test Set", "Training Set"]
            ) +
        scale_linetype_manual(
            values={"lr": "solid", "pmf": "dashed"},
            labels={"lr": "Log Reg", "pmf": "PMF"}
        ) +
        theme_bw() +
        theme(
            axis_title=element_text(size=22),
            axis_text=element_text(size=14),
            legend_title=element_text(size=22),
            legend_text=element_text(size=18),
            strip_text=element_text(size=16),
            legend_key_spacing_y = 15,
            legend_spacing = 60
        )
    )


if __name__ == '__main__':
    import torch
    from pmf import PMF
    from lr import LogReg
    import pickle

    SEED = 42
    K_FOLDS = 5
    REPLACEMENT_LEVELS = [25, 50, 100, 250, 500, 1000]
    LATENT_FACTORS = [1, 2, 3, 4, 5]

    df = pd.read_csv('data/men_data.csv')

    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]

        for replacement_level in REPLACEMENT_LEVELS:
            with open(f"models/lr/model_rl_{replacement_level}_fold_{fold}.pkl", 'rb') as f:
                lr_model = pickle.load(f)
            for num_factors in LATENT_FACTORS:
                results.append({
                    'model_type': 'lr',
                    'replacement_level': replacement_level,
                    'num_factors': num_factors,
                    'fold': fold,
                    'train_results': evaluate_model(lr_model, train),
                    'test_results': evaluate_model(lr_model, test)
                })

            for num_factors in LATENT_FACTORS:
                pmf_model = torch.load(f"models/pmf/model_rl_{replacement_level}_d_{num_factors}_fold_{fold}.pth")
                pmf_model.eval()
                results.append({
                    'model_type': 'pmf',
                    'replacement_level': replacement_level,
                    'num_factors': num_factors,
                    'fold': fold,
                    'train_results': evaluate_model(pmf_model, train),
                    'test_results': evaluate_model(pmf_model, test)
                })

    results = pd.DataFrame(results)

    metrics = pd.concat([
        pd.concat(
            [
                results[['model_type', 'replacement_level', 'num_factors', 'fold']],
                results['train_results'].apply(pd.Series).assign(dataset = "train")
            ], axis = 1
        ),
        pd.concat(
            [
                results[['model_type', 'replacement_level', 'num_factors', 'fold']],
                results['test_results'].apply(pd.Series).assign(dataset = "test")
            ], axis = 1
        )
    ])

    metrics_long = pd.melt(
        metrics,
        id_vars = ['model_type', 'replacement_level', 'num_factors', 'fold', 'dataset'],
        value_vars = ['Accuracy', 'Log Loss', 'Brier Score', 'F1 Score', 'ROC Score'],
        var_name = 'metric'
    )

    plot_data = get_data_for_plot(metrics_long)

    import plotnine 
    plotnine.options.figure_size = (18,9)
    p = produce_results_plot(plot_data)
    p.save("results/performance_evaluation/results.png")
