from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd
import os

from app.database.connection import get_db
from app.database.crud import get_all_features, create_prediction, bulk_insert_features
from app.services.model_service import load_data_from_db, train_and_save_model
from app.services.data_preprocessing import split_data, preprocess_data
from sklearn.metrics import accuracy_score

router = APIRouter()

# Визначаємо назву файлу (використовуємо ту, яка викликала помилку, але перевіряємо шлях)
DATA_FILE_NAME = "ObesityDataSet_raw_and_data_sinthetic.csv" 
# Якщо ви використовуєте 'cleaned.csv', змініть цей рядок

@router.post("/train-model")
def train_model_endpoint(db: Session = Depends(get_db)):
    
    # Якщо БД порожня, завантажуємо дані з файлу
    if not get_all_features(db):
        
        # 1. Формування шляху
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Шлях до кореня проєкту (lab4): піднімаємося на два рівні
        project_root_dir = os.path.join(current_dir, "..", "..")
        # Фінальний шлях до файлу: lab4/data/ObesityDataSet_raw_and_data_sinthetic.csv
        file_path = os.path.join(project_root_dir, "data", DATA_FILE_NAME)
        
        # Виводимо абсолютний шлях до консолі Uvicorn для відлагодження
        print(f"\nDEBUG: Calculated absolute file path: {file_path}")
        
        # 2. Перевірка існування файлу
        if not os.path.exists(file_path):
            print(f"ERROR: File not found at the calculated path: {file_path}")
            raise HTTPException(
                status_code=500, 
                detail=f"Data file not found. Ensure '{DATA_FILE_NAME}' is in the 'data/' directory."
            )
        
        # 3. Читання та вставка даних
        try:
            df_raw = pd.read_csv(file_path)
            
            # Перейменовуємо, якщо потрібно (якщо колонка існує)
            if 'NObeyesdad' in df_raw.columns:
                df_raw = df_raw.rename(columns={'NObeyesdad': 'NObeyesdad'}) 
                
            bulk_insert_features(db, df_raw)
            print("INFO: Successfully inserted initial data into 'features' table.")

        except Exception as e:
            # Ловимо інші помилки, наприклад, помилки читання CSV або вставки в БД
            raise HTTPException(status_code=500, detail=f"Failed to load initial data or insert into DB: {e}")
    
    # --- Подальша логіка тренування ---
    
    # 1. Читає всі записи з таблиці ознак
    df_features = load_data_from_db(db)
    
    # 2. Ділить їх на train та new_input (90:10)
    df_train, df_new_input = split_data(df_features, test_size=0.1)
    
    if df_train.empty:
        # Ця помилка може виникнути, якщо дані були вставлені, але їх потім видалили, або виникла помилка в CRUD
        raise HTTPException(status_code=400, detail="Not enough data to train. Check database connection or CRUD operations.")
    
    # 3. Тренує модель та 4. Зберігає її
    model = train_and_save_model(df_train)
    
    # 5. Додає передбачення на train-даних у таблицю predictions
    X_train, y_train, _ = preprocess_data(df_train)
    predictions_raw = model.predict(X_train)
    
    # Декодування передбачень для логування
    # Припускаємо, що 1 = Obese, 0 = Not Obese (на основі препроцесингу)
    predicted_values = ["Obese" if p == 1 else "Not Obese" for p in predictions_raw]
    
    # Логування
    for index, (pred_val, feat_id) in enumerate(zip(predicted_values, df_train['id'])):
        create_prediction(
            db=db, 
            feature_id=feat_id, 
            predicted_value=pred_val, 
            source="train"
        )
        
    # Оцінка якості (для звіту)
    accuracy = accuracy_score(y_train, predictions_raw)
    
    return {
        "message": "Model trained and saved successfully.",
        "train_size": len(df_train),
        "new_input_size": len(df_new_input),
        "train_accuracy": round(accuracy, 4),
        "model_path": "models/obesity_classifier.joblib",
        "log_entries_count": len(df_train)
    }