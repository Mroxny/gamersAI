from modules import prep_data
from modules import model_training
from modules import model_prediction


def main():
    # Step 1: Prepeare data
    data_file = "data/games.csv"
    df = prep_data.prepare_data_frame(data_file)

    # Step 2: Train the model
    model, X_test, y_test = model_training.train_model(df)

    # Step 3: Prediction and Results
    prediction = model_prediction.predict(model, X_test)
    model_prediction.regression_results(y_test, prediction)

if __name__ == "__main__":
    main()