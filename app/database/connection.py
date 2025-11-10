from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database.models import Base

# Файл бази даних
SQLALCHEMY_DATABASE_URL = "sqlite:///./lab_db.sqlite"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    # Створює всі таблиці, визначені в Base (models.py)
    Base.metadata.create_all(bind=engine)

# Dependency для отримання сесії БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()