import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.datasets import load_iris

def preprocess_games(df: pd.DataFrame):
    print('Start test autogluon')
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    # Podziel dane na dane treningowe i testowe
    train_data = data.sample(frac=0.8, random_state=42)  # 80% jako treningowe
    test_data = data.drop(train_data.index)              # 20% jako testowe

    # Definiuj predyktor i trenuj model
    predictor = TabularPredictor(label='target').fit(train_data)
    # Przewiduj na danych testowych
    predictions = predictor.predict(test_data)

    # Wyświetl kilka przykładowych wyników
    print("Przykładowe przewidywania:")
    print(predictions.head())    
    print('End test autogluon')

    # Ocena wyników
    performance = predictor.evaluate(test_data)
    print("Ocena modelu:", performance)

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

    return df

def convert_estimated_owners(owners_range):
    lower, upper = owners_range.split(' - ')
    return (int(lower) + int(upper) /2)

def create_model_input_table(games: pd.DataFrame) -> pd.DataFrame:
    return games