from load_and_clean_data import load_and_clean_data
from compare_representation import comparing_representations

def main():
    #Wczytanie i wstępne czyszczenie danych
    clean_training = load_and_clean_data()

    #Porównanie
    comparing_representations(clean_training)

if __name__ == "__main__":
    main()