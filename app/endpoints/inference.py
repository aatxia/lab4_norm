from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.database.crud import create_prediction, create_inference_input
from app.services.model_service import load_model, predict_single_input
from app.schemas import InferenceInputSchema, PredictionResponseSchema

router = APIRouter()

@router.post("/predict", response_model=PredictionResponseSchema)
def predict_endpoint(
    input_data: InferenceInputSchema, 
    db: Session = Depends(get_db)
):
    """
    Робить передбачення на основі вхідних JSON-даних:
    1. Підвантажує модель.
    2. Робить передбачення.
    3. Зберігає результат у predictions (source="inference").
    4. Зберігає вхідні параметри у inference_inputs.
    """
    
    # 1. Підвантажує модель з диска
    model = load_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Trained model not found. Please run POST /train-model first."
        )
    
    # Конвертуємо Pydantic-схему в словник для роботи моделі
    input_dict = input_data.model_dump()
    
    # 2. Робить передбачення
    predicted_value = predict_single_input(model, input_dict)
    
    # 3. Зберігає результат у таблицю predictions (source="inference")
    # feature_id = None, оскільки це нові вхідні дані, а не дані з таблиці features
    db_prediction = create_prediction(
        db=db, 
        feature_id=None, 
        predicted_value=predicted_value, 
        source="inference"
    )
    
    # 4. Зберігає вхідні параметри у таблицю inference_inputs
    create_inference_input(
        db=db, 
        prediction_id=db_prediction.id, 
        input_data=input_dict
    )
    
    return PredictionResponseSchema(
        prediction=predicted_value,
        log_id=db_prediction.id,
        source="inference"
    )