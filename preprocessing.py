import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_split(df, seed):
    train_df, test_df = train_test_split(df, train_size=0.8, test_size=0.2, random_state=seed, stratify=df['Problem_ID'])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == '__main__':
    df_raw = pd.read_excel("https://github.com/DavidBreuer/ifsc-analysis/raw/main/ifsc_Boulder.xlsx")

    df = (
        df_raw
        # Get rid of columns we don't need
        .drop(columns=['Unique', 'Discipline', 'Number', 'Group'])
        .drop(columns=df_raw.filter(regex='Route|Run').columns)
        # Capitalize climber names consistently
        .assign(Name=lambda x: x['Name'].str.title())
        # "Unpivot" so it's one row per climber-problem
        # I'm treating tops and zones as separate problems even though there's obviously a correlation
        .melt(id_vars=['Year', 'Competition', 'Gender', 'Level', 'Name', 'Country'],
            value_vars=['Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'Zone1', 'Zone2', 'Zone3', 'Zone4', 'Zone5'],
            var_name='Problem', value_name='Attempts')
        .dropna(subset=['Attempts'])
        .replace([-np.inf, np.inf], np.nan)
    )

    # Keep completed boulders
    grouped = df.groupby(['Competition', 'Level', 'Problem', 'Gender'])
    df = grouped.filter(lambda x: x['Attempts'].notnull().any())

    # Max attempts passed through
    df = df.assign(Max_attempts=lambda x: grouped['Attempts'].transform('max'))

    # Survival model features
    df = df.assign(
        Status=lambda x: x['Attempts'].notna(),
        Time=lambda x: np.where(x['Attempts'].isna(), x['Max_attempts'], x['Attempts'])
    )

    # Failure set to max attempts
    df['Attempts'] = np.where(df['Attempts'].isna(), df['Max_attempts'], df['Attempts'])
    df[['Attempts', 'Max_attempts', 'Status', 'Time']] = df[['Attempts', 'Max_attempts', 'Status', 'Time']].astype(int)

    # Make Problem Category
    df['Problem_category'] = np.where(df.Problem.str.startswith('Top').values, 'Top', 'Zone')

    # Handle 
    dfm = df[df.Gender=="M"].copy()
    dfw = df[df.Gender=="W"].copy()

    STRATIFICATION = ['Year', 'Competition', 'Level', 'Problem']
    dfm['Problem_ID'] = dfm[STRATIFICATION].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    dfw['Problem_ID'] = dfw[STRATIFICATION].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    dfm.to_csv('./data/men_data.csv', index=False)
    dfw.to_csv('./data/women_data.csv', index=False)