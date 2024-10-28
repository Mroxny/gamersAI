from modules import prep_data
from modules import model_training

def main():
    # Step 1: Prepeare data
    data_file = "data/games.csv"
    df = prep_data.prepare_data_frame(data_file)

    # Step 2: Train the model
    model, X_test, y_test = model_training.train_model(df)

if __name__ == "__main__":
    main()