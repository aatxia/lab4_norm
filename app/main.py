from fastapi import FastAPI
from app.endpoints import training, inference
from app.database.connection import init_db

# Ініціалізуємо додаток FastAPI і називаємо його 'app'
app = FastAPI(title="ML Deployment Lab 4") # <<< Ось ця змінна потрібна!

# Підключаємо роутери
app.include_router(training.router)
app.include_router(inference.router)

# Створюємо таблиці при запуску додатку
@app.on_event("startup")
def on_startup():
    init_db()
    print("Database initialized and tables created.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Deployment API. Use /train-model and /predict."}