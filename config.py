import os
import google.generativeai as genai
from dotenv import load_dotenv

DATA_DIR = "data"
CSV_FILENAME = "hospitals_leb.csv"
PDF_FILENAME = "WHO_Article.pdf"
CSV_FILEPATH = os.path.join(DATA_DIR, CSV_FILENAME)
PDF_FILEPATH = os.path.join(DATA_DIR, PDF_FILENAME)

EMBEDDING_MODEL_NAME = 'models/embedding-001' 
GENERATIVE_MODEL_NAME = 'gemini-2.0-flash'   

CSV_RELEVANT_COLUMNS = ['Hospital', 'Est_Date', 'Location', 'Address', 'NumBeds']
CSV_ID_COLUMN = 'Id'

TOP_K = 3

CHUNK_SIZE = 1000 # Characters per chunk
CHUNK_OVERLAP = 150 # Characters overlap between chunks

EMBEDDING_DIMENSION = 768

def configure_api():
    """Loads API key from .env and configures the Generative AI client."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    print("Gemini API configured.")