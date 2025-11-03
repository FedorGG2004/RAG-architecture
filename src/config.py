import os
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

# Настройки модели
MODEL_NAME = "tinyllama:1.1b"
OLLAMA_HOST = "http://localhost:11434"

# Настройки векторной БД
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_COLLECTION_NAME = "diplom_rag_memory"

# Настройки поиска
TOP_K_RESULTS = 3

# Создаем необходимые директории
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "test_documents").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "knowledge_base").mkdir(parents=True, exist_ok=True)