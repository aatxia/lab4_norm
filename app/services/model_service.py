import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from app.services.data_preprocessing import preprocess_data

MODEL_PATH = "models/obesity_classifier.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

def load_data_from_db(db):
    """Читає всі записи ознак з БД."""
    from app.database.crud import get_all_features
    features = get_all_features(db)
    
    # Конвертуємо список об'єктів SQLAlchemy у DataFrame
    data = [{
        'id': f.id, 
        'Gender': f.Gender, 
        'Age': f.Age, 
        'Height': f.Height, 
        'Weight': f.Weight, 
        'FAVC': f.FAVC,
        'NObeyesdad': f.NObeyesdad,
        'timestamp': f.timestamp
    } for f in features]
    
    return pd.DataFrame(data)

def train_and_save_model(df_train):
    X_train, y_train, _ = preprocess_data(df_train)

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        max_features="sqrt",
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/obesity_classifier.joblib")
    return model

def load_model():
    """Завантажує модель з диска."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_single_input(model, data: dict) -> str:
    """Робить передбачення для одного вхідного запису (для /predict)."""
    
    # Конвертуємо вхідні дані у DataFrame для моделі
    df = pd.DataFrame([data])
    
    # Обробка категоріальних ознак
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['FAVC'] = df['FAVC'].map({'no': 0, 'yes': 1})
    
    # Вибір ознак
    X = df[['Gender', 'Age', 'Height', 'Weight', 'FAVC']]
    
    # Передбачення
    prediction_raw = model.predict(X)[0]
    
    # Конвертація результату в зрозумілий рядок
    return "Obese" if prediction_raw == 1 else "Not Obese"