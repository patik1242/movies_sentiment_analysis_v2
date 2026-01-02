import pandas as pd
from preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data():
    """
    Wczytanie, analiza i czyszczenie datasetu
    """

    training_dataset = pd.read_csv("data/IMDB_Dataset.csv")

    print('==BEFORE CLEANING - DATASET ANALYSIS==')
    print("Number of duplicates: ", training_dataset.duplicated().sum())
    print('Number of null values: \n', training_dataset.isnull().sum())
    print("Number of empty rows: ", training_dataset.isnull().all(axis=1).sum())
    print("Number of rows: ", training_dataset.shape[0])
    print("Number of columns: ", training_dataset.shape[1])
    print("Number of classes: ", training_dataset.nunique())


    print('\n==REVIEW ANALYSIS==')
    print("Type of reviews: ", training_dataset["review"].apply(type).value_counts())
    print("Minimum number of chars: ", training_dataset["review"].str.len().min())
    print("Maximum number of chars: ", training_dataset["review"].str.len().max())
    print("Mean of numbers of chars: ", training_dataset["review"].str.len().mean())
    print("Median of numbers of chars: ", training_dataset["review"].str.len().median())
    print("Missing values: ", training_dataset["review"].isna().sum())


    print('\n==SENTIMENT ANALYSIS==')
    print("Type of sentiment: ", training_dataset["sentiment"].apply(type).value_counts())
    print("Unique values: ", training_dataset["sentiment"].unique())
    print("Missing values: ", training_dataset["sentiment"].isna().sum())
    print("Empty strings: ",(training_dataset["sentiment"]=="").sum()) 
    print("Balance of classes: ",training_dataset["sentiment"].value_counts())
    print("Balance of classes (%): ",(training_dataset["sentiment"].value_counts(normalize=True)).mul(100).round(2))



    clean_training = training_dataset.drop_duplicates()
    clean_training = clean_training.reset_index(drop=True)
    clean_training["review"] = clean_training["review"].apply(preprocess)
    clean_training = clean_training[clean_training["review"].str.strip()!=""]
    clean_training["sentiment"] = clean_training["sentiment"].map({
        "negative":0,
        "positive":1
    })


    print("==AFTER CLEANING - DATASET ANALYSIS==")
    print("Number of duplicates: ", clean_training.duplicated().sum())
    print('Number of null values: \n', clean_training.isnull().sum())
    print("Number of rows: ", clean_training.shape[0])
    print("Number of columns: ", clean_training.shape[1])
    print("Number of deleted rows: ", training_dataset.shape[0] - clean_training.shape[0])
    print(f"Number of kept rows (%): {(clean_training.shape[0]/training_dataset.shape[0])*100:.2f}")


    print('\n==REVIEW ANALYSIS==')
    print("Minimum number of chars: ", clean_training["review"].str.len().min())
    print("Maximum number of chars: ", clean_training["review"].str.len().max())
    print("Mean of numbers of chars: ", clean_training["review"].str.len().mean())


    print('\n==SENTIMENT ANALYSIS==')
    print("Balance of classes: ",clean_training["sentiment"].value_counts())
    print("Balance of classes (%): ",(clean_training["sentiment"].value_counts(normalize=True)).mul(100).round(2))
    print("Number of NaN values: ",clean_training["sentiment"].isna().sum())

    return clean_training

def training_data(clean_training):
    
    texts = clean_training["review"].copy()

    #Własne cechy numeryczne
    custom_features=["pos", "neg","pos_ratio","neg_ratio","negated_pos_count", 
                        "negated_neg_count", "exclamation_count", 
                    "digit_count", "question_count", "negation_count", 
                    "intensifier_count", "contrast_count", "pos_end", "neg_end"]
    
    X_custom = clean_training[custom_features]

    #etykiety klas sentymentu
    y = clean_training["sentiment"]

    #Podział danych
    (X_text_train, X_text_test, X_custom_train, X_custom_test,y_train, y_test) = train_test_split(
        texts, X_custom, y, test_size=0.2, random_state=42, stratify=y)
    
    #Standaryzacja cech numerycznych
    scaler = StandardScaler()

    X_custom_train = scaler.fit_transform(X_custom_train)
    X_custom_test = scaler.transform(X_custom_test)

    return (X_text_train, X_text_test, 
            X_custom_train, X_custom_test, 
            y_train, y_test)