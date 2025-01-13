import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os

def preprocess_games(df: pd.DataFrame):

    df = df.replace({"True": True, "False": False})
    df.update(df.apply(pd.to_numeric, errors='coerce'))
    df[['Genre 1', 'Genre 2']] = df['Genres'].str.split(',', n=2, expand=True)[[0, 1]]
    df['Genre 1'] = df['Genre 1'].str.strip()
    df['Genre 2'] = df['Genre 2'].str.strip()

    # Drop useless columns
    drop_labels = ['AppID', 'About the game', 'Full audio languages', 'Reviews', 'Header image', 'Website','Support url','Support email','Metacritic url', 'User score','Score rank','Notes','Average playtime forever','Average playtime two weeks','Median playtime two weeks','Tags','Screenshots','Movies']
    df.drop(axis= "columns", columns= drop_labels, inplace = True)

    # Extract the most important languages
    lang_eng = df['Supported languages'].astype(str).str.contains("English")
    lang_sp = df['Supported languages'].astype(str).str.contains("Spanish")
    lang_de = df['Supported languages'].astype(str).str.contains("German")
    lang_fr = df['Supported languages'].astype(str).str.contains("French")

    df['Is English'] = lang_eng
    df['Is Spanish'] = lang_sp
    df['Is German'] = lang_de
    df['Is French'] = lang_fr

    # Delete "Supported languages" column
    df.drop(axis= "columns", columns= ['Supported languages'], inplace = True)

    # Extract the most important languages
    cat_single = df['Categories'].astype(str).str.contains("Single-player")
    cat_multi = df['Categories'].astype(str).str.contains("Multi-player")

    df['Is Single-player'] = cat_single
    df['Is Multi-player'] = cat_multi

    # Delete "Categories" and "Genres" columns
    df.drop(axis= "columns", columns= ['Categories'], inplace = True)
    df.drop(axis= "columns", columns= ['Genres'], inplace = True)

    # Convert the "Estimated owners" column to int - average of the range
    df['Estimated owners (avg)'] = df['Estimated owners'].apply(convert_estimated_owners)

    df.drop(axis='columns', columns='Estimated owners', inplace=True)
    df.head()

    to_drop = ['Name', 'Release date', 'Developers', 'Publishers', 'Genre 2']
    df.drop(axis='columns', columns=to_drop, inplace=True)

    df.dropna(inplace=True)
    df['Genre 1_encoded'] = pd.factorize(df['Genre 1'])[0] + 1
    df.drop(axis='columns', columns='Genre 1', inplace=True)

    df = df.nsmallest(81543, keep='first', columns='Estimated owners (avg)')
    df.to_csv('output.csv', index=False)
    return df

def convert_estimated_owners(owners_range):
    lower, upper = owners_range.split(' - ')
    return (int(lower) + int(upper) /2)

def create_model_input_table(games: pd.DataFrame) -> pd.DataFrame:
    return games

def log_heatmap_to_wandb(df: pd.DataFrame):
    """
    Creates a heatmap from a DataFrame, logs it to Weights & Biases (WandB), and sends it for visualization.

    Args:
        df (pd.DataFrame): The DataFrame to visualize as a heatmap.
    """
    # Initialize the WandB run
    run = wandb.init(
        project="gamersAI",
        name="Heatmap_Logging",
        group=os.environ.get("WANDB_RUN_GROUP"),
        config={} 
    )

    # Create a correlation heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.title("Correlation Heatmap")

    # Log the heatmap
    wandb.log({"heatmap": wandb.Image(plt)})
    plt.close()

    # Additional metadata
    to_log = {
        "columns_logged": list(df.columns),
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
    }
    run.log(to_log)

    # Finish the WandB run
    run.finish()
    print("Heatmap and metadata logged to WandB.")