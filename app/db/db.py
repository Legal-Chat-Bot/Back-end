
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://admin:1234@postgres:5432/service_db" #임시 url
engine = create_engine(DATABASE_URL)
connection = engine.connect()
print("PostgreSQL connected!")