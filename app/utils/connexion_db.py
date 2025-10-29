import os
from dotenv import load_dotenv
from sqlalchemy import create_engine


def connexion_db():

    env = os.getenv("ENV", "dev")
    env_file = f".env.{env}"
    load_dotenv(dotenv_path=env_file)
    
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    dbname = os.getenv("DB_NAME")

    conn = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

    return conn