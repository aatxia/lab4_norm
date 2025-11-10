import pandas as pd
from sqlalchemy.orm import Session
from app.database.models import Feature, Prediction, InferenceInput
from datetime import datetime
from typing import List, Dict, Any

def bulk_insert_features(db: Session, df: pd.DataFrame):
    """Масове додавання даних з датасету в таблицю features."""
    
    # Припускаємо, що датасет вже прочитано в df
    records = df.to_dict(orient='records')
    feature_objects = [
        Feature(
            Gender=r['Gender'],
            Age=r['Age'],
            Height=r['Height'],
            Weight=r['Weight'],
            FAVC=r['FAVC'],
            NObeyesdad=r['NObeyesdad'],
            # Додаємо timestamp, який може бути згенерований або взятий з даних
            timestamp=datetime.utcnow() 
        ) for r in records
    ]
    
    db.add_all(feature_objects)
    db.commit()

def get_all_features(db: Session) -> List[Feature]:
    """Отримати всі записи з таблиці features."""
    return db.query(Feature).all()

def create_prediction(db: Session, feature_id: int | None, predicted_value: str, source: str) -> Prediction:
    """Створює запис у таблиці predictions."""
    db_prediction = Prediction(
        feature_id=feature_id,
        predicted_value=predicted_value,
        source=source
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def create_inference_input(db: Session, prediction_id: int, input_data: Dict[str, Any]) -> InferenceInput:
    """Створює запис у таблиці inference_inputs."""
    
    # Забираємо не потрібні поля, залишаємо лише ті, що є у моделі InferenceInput
    input_fields = {
        'prediction_id': prediction_id,
        'Gender': input_data.get('Gender'),
        'Age': input_data.get('Age'),
        'Height': input_data.get('Height'),
        'Weight': input_data.get('Weight'),
        'FAVC': input_data.get('FAVC'),
    }
    
    db_input = InferenceInput(**input_fields)
    db.add(db_input)
    db.commit()
    db.refresh(db_input)
    return db_input