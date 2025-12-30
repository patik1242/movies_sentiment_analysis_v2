from add_features import add_features
from load_and_clean_data import load_and_clean_data
from train_with_grid_and_custom_features import train_with_grid_and_custom_features

def main():
    #Wczytanie i wstępne czyszczenie danych
    clean_training_pre = load_and_clean_data()

    #Feature engineering
    clean_training = add_features(clean_training_pre)

    #Trening modelu 
    train_with_grid_and_custom_features(clean_training_pre, clean_training)

if __name__ == "__main__":
    main()