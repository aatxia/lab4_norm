from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
import os

from app.database.connection import get_db
from app.database.crud import get_all_features, create_prediction, bulk_insert_features
from app.services.model_service import load_data_from_db, train_and_save_model
from app.services.data_preprocessing import split_data, preprocess_data
from sklearn.metrics import accuracy_score

router = APIRouter()

DATA_FILE_NAME = "ObesityDataSet_raw_and_data_sinthetic.csv"


@router.post("/train-model")
def train_model_endpoint(db: Session = Depends(get_db)):

    # Якщо в таблиці features порожньо — імпортуємо CSV
    if not get_all_features(db):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..", "..")
        file_path = os.path.join(project_root, "data", DATA_FILE_NAME)

        print(f"\nDEBUG: Using CSV at: {file_path}")

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=500,
                detail=f"Data file not found. Ensure '{DATA_FILE_NAME}' is in /data."
            )

        try:
            df = pd.read_csv(file_path)

            # Перевіряємо колонки
            required_cols = ["Age", "Height", "Weight", "FAVC", "NObeyesdad", "Gender_Male"]
            for col in required_cols:
                if col not in df.columns:
                    raise Exception(f"CSV missing column: {col}")

            # One-hot → текст
            df["Gender"] = df["Gender_Male"].apply(lambda x: "Male" if x == 1 else "Female")

            # Потрібні колонки
            df_db = df[["Gender", "Age", "Height", "Weight", "FAVC", "NObeyesdad"]]

            bulk_insert_features(db, df_db)
            print("INFO: Inserted CSV data into DB.")

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to import CSV data: {e}"
            )

    # === TRAINING ===

    df_features = load_data_from_db(db)
    df_train, df_new_input = split_data(df_features, test_size=0.1)

    if df_train.empty:
        raise HTTPException(status_code=400, detail="Not enough data to train.")

    # Тренуємо модель
    model = train_and_save_model(df_train)

    # Preprocess train
    X_train, y_train, _ = preprocess_data(df_train)

    # Preprocess test (new_input)
    X_test, y_test, _ = preprocess_data(df_new_input)

    # Передбачення на train-для логування
    raw_preds_train = model.predict(X_train)
    pred_labels_train = ["Obese" if p == 1 else "Not Obese" for p in raw_preds_train]

    # Логування передбачень для TRAIN
    for pred, feat_id in zip(pred_labels_train, df_train["id"]):
        create_prediction(
            db=db,
            feature_id=int(feat_id),
            predicted_value=pred,
            source="train"
        )

    # Оцінка точності на TEST
    test_preds = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_preds)

    return {
        "message": "Model trained and saved successfully.",
        "train_size": len(df_train),
        "new_input_size": len(df_new_input),
        "test_accuracy": round(test_accuracy, 4),   # <-- НОВЕ ЗНАЧЕННЯ
        "model_path": "models/obesity_classifier.joblib",
        "log_entries_count": len(df_train)
    }
