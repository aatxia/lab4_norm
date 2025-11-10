from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

# Таблиця 1: features (Ознаки)
class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, index=True)
    Gender = Column(String)
    Age = Column(Float)
    Height = Column(Float)
    Weight = Column(Float)
    FAVC = Column(String) # Frequent consumption of high caloric food
    NObeyesdad = Column(String) # Target variable
    
    # *** КЛЮЧОВЕ: timestamp для кожного запису ознаки ***
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Зворотний зв'язок: один-до-багатьох до predictions
    predictions = relationship("Prediction", back_populates="feature_record")

# Таблиця 2: predictions (Передбачення)
class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    feature_id = Column(Integer, ForeignKey("features.id"), nullable=True) # ID ознаки (NULL для inference)
    predicted_value = Column(String)
    
    # *** КЛЮЧОВЕ: source ("train" або "inference") ***
    source = Column(String) 
    
    prediction_timestamp = Column(DateTime, default=datetime.utcnow)

    # Зв'язки
    feature_record = relationship("Feature", back_populates="predictions")
    inference_input_record = relationship("InferenceInput", back_populates="prediction_record", uselist=False)

# Таблиця 3: inference_inputs (Вхідні параметри для передбачення)
class InferenceInput(Base):
    __tablename__ = "inference_inputs"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True) # Зв'язок 1:1 з Prediction
    
    # Поля, що зберігають вхідні параметри
    Gender = Column(String)
    Age = Column(Float)
    Height = Column(Float)
    Weight = Column(Float)
    FAVC = Column(String) 
    
    request_timestamp = Column(DateTime, default=datetime.utcnow)
    
    prediction_record = relationship("Prediction", back_populates="inference_input_record")