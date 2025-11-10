import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame):
    """Підготовка даних: кодування цільової змінної та ознак."""
    
    # 1. Обробка цільової змінної (NObeyesdad)
    # Зміна назв для бінарної класифікації (спрощення)
    df['Obese_Target'] = df['NObeyesdad'].apply(
        lambda x: 1 if x in ['Obesity Type I', 'Obesity Type II', 'Obesity Type III'] else 0
    )
    
    # 2. Обробка категоріальних ознак (Gender, FAVC)
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['FAVC'] = df['FAVC'].map({'no': 0, 'yes': 1})
    
    # Вибір ознак та цільової змінної
    X = df[['Gender', 'Age', 'Height', 'Weight', 'FAVC']]
    y = df['Obese_Target']
    
    # Повертаємо ознаки, цільову змінну та оригінальний датафрейм (з timestamp)
    return X, y, df

def split_data(df: pd.DataFrame, test_size: float = 0.1):
    """Розділяє дані 90:10, але не на X_train/X_test, а на train/new_input."""
    
    # Встановлюємо довільний порядок, щоб останні 10% були "нові вхідні"
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_index = int(len(df_shuffled) * (1 - test_size))
    
    train_df = df_shuffled.iloc[:split_index].copy()
    new_input_df = df_shuffled.iloc[split_index:].copy()
    
    return train_df, new_input_df