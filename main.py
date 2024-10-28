from modules import prep_data

def main():
    data_file = "data/games.csv"
    df = prep_data.prepare_data_frame(data_file)

    # Test (safe to delete)
    print(df)

if __name__ == "__main__":
    main()