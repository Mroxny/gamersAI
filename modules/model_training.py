import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_model(df: pd.DataFrame):
    # Define X and y sets
    X = df.drop('Estimated owners (avg)', axis='columns')
    y = df['Estimated owners (avg)']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    # Initialize and train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model, X_test, y_test