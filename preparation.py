import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dictionaries_and_extracting_features import extract_features

def training_data(clean_training):
    
    texts = clean_training["review"].copy()
    #etykiety klas sentymentu
    y = clean_training["sentiment"]

    #Podział danych
    (X_text_train, X_text_test,y_train, y_test) = train_test_split(
        texts,y, test_size=0.2, random_state=42, stratify=y)
    
    X_custom_train = X_text_train.apply(extract_features).apply(pd.Series)
    X_custom_test = X_text_test.apply(extract_features).apply(pd.Series)

    #Standaryzacja cech numerycznych
    scaler = StandardScaler()


    X_custom_train = pd.DataFrame(
        scaler.fit_transform(X_custom_train), 
        columns=X_custom_train.columns,
        index = X_custom_train.index
    )
    X_custom_test = pd.DataFrame(
        scaler.transform(X_custom_test), 
        columns=X_custom_test.columns, 
        index = X_custom_test.index
    )

    return (X_text_train, X_text_test, 
            X_custom_train, X_custom_test, 
            y_train, y_test)