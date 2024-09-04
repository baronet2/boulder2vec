import plotnine
from plotnine import * 

def climbers_above_replacement_level(df, color_fill, filename):
    attempt_counts = df['Name'].value_counts()
    climbers_above = [sum(attempt_counts > level) for level in REPLACEMENT_LEVELS]
    plot_df = pd.DataFrame({
        'Replacement Level': REPLACEMENT_LEVELS,
        'Climbers Above': climbers_above})
    plot_df['Replacement Level'] = plot_df['Replacement Level'].astype('category')

    plotnine.options.figure_size = (14,6)
    
    p = (
    ggplot(plot_df, aes(x='Replacement Level', y='Climbers Above'))
    + geom_bar(stat='identity', fill=color_fill, alpha=0.7, width = 0.9)
    + geom_text(aes(label='Climbers Above'), va='bottom', size=10)
    + labs(x='Replacement Level $N$ (# of Problems Attempted)', 
           y='# Climbers Above Replacement Level')
    + theme_bw()
    + theme(axis_title=element_text(size=16), 
            axis_text=element_text(size=12))
    + scale_x_discrete())
    p.save(f'results/eda/{filename}.png', dpi=1000)
    print(f"Saved file {filename}.png")

    return p


def height_histogram(df, color_fill, filename):
    plotnine.options.figure_size = (12,6)

    p = (ggplot(df, aes(x='Height'))
    + geom_histogram(binwidth=2, fill=color_fill, alpha = 0.7)
    + labs(x='Height (cm)', y='# of Climbers')
    + theme_bw()
    + scale_x_continuous(breaks=range(160, 191, 2))
    + scale_y_continuous(breaks=range(0, 15, 3))
    + theme(axis_title=element_text(size=24),
            axis_text=element_text(size=18)))

    p.save(f'results/eda/{filename}.png', dpi=1000)
    print(f"Saved file {filename}.png")

    return p

def problem_attempts(df, color_fill, filename):
    problem_id_counts = df['Problem_ID'].value_counts().reset_index()
    problem_id_counts.columns = ['Problem_ID', 'Frequency']

    plotnine.options.figure_size = (16,6)
    p = (
    ggplot(problem_id_counts, aes(x='Frequency'))
    + geom_histogram(binwidth=5, fill=color_fill, alpha=0.8)
    + labs(x='# of Climbers Attempted', 
           y='# of Problems')
    + theme_bw()
    + theme(
        axis_title=element_text(size=24),
        axis_text=element_text(size=18))
    + scale_x_continuous(breaks=range(0, 167, 10))
    + scale_y_continuous(breaks=range(0, 900, 60)))

    p.save(f'results/eda/{filename}.png', dpi=1000)
    print(f"Saved file {filename}.png")

    return p

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/men_data.csv')
    heights = pd.read_csv('data/climbers_heights.csv', index_col = 0)
    REPLACEMENT_LEVELS = [25, 50, 100, 250, 500, 1000]

    climbers_above_replacement_level(df,'#00abff', 'climbers_above_replacement_level')
    height_histogram(heights,'#00abff', 'height_histogram')
    problem_attempts(df, '#00abff', 'problem_id_frequency_histogram')