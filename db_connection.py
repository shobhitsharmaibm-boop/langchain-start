
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_mysql_connection():
    """
    Creates a SQLAlchemy engine for MySQL connection using environment variables.
    """
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    database = os.getenv("DB_NAME", "school_db")
    
    # Create the connection string
    # Connection format: mysql+pymysql://user:password@host/database
    connection_string = f"mysql+pymysql://{user}:{password}@{host}/{database}"
    
    try:
        engine = create_engine(connection_string)
        print("Successfully connected to MySQL database.")
        return engine
    except Exception as e:
        print(f"Error connecting to MySQL: {e}")
        return None