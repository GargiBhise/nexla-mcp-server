import os
import json
import faiss
import numpy as np
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
from src.metadata import extract_metadata

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model to use
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunk size and overlap in characters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
