import os
import sys
from pathlib import Path

def create_project_structure():
    # Правильный путь для Windows
    base_dir = Path(r"C:\Users\Fedor\Desktop\Diplom")
    
    print("🔄 Создание структуры проекта для Windows 10...")
    
    # Список всех директорий для создания
    directories = [
        ".vscode",
        "src",
        "data/chroma_db",
        "data/test_documents", 
        "data/knowledge_base",
        "tests",
        "docs"
    ]
    
    # Создаем директории
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Создана папка: {dir_path}")
    
    # Создаем файлы с содержимым (адаптировано для Windows)
    files_content = {
        # Конфигурация VS Code
        ".vscode/settings.json": '''{
    "python.defaultInterpreterPath": "venv\\\\Scripts\\\\python.exe",
    "python.analysis.extraPaths": ["./src"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true
    },
    "editor.formatOnSave": true,
    "python.formatting.provider": "black"
}''',
        
        ".vscode/launch.json": '''{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: RAG System",
            "type": "python",
            "request": "launch",
            "program": "src/main.py",
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}''',
        
        # Основные файлы
        "requirements.txt": '''ollama
chromadb
sentence-transformers
requests
numpy
pydantic
python-dotenv''',
        
        "src/__init__.py": "# RAG System Package",
        
        "src/config.py": '''import os
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
(DATA_DIR / "knowledge_base").mkdir(parents=True, exist_ok=True)''',
        
        "src/vector_db.py": '''import chromadb
from sentence_transformers import SentenceTransformer
import logging
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        logger.info("Инициализация векторной базы данных...")
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection = self.client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Векторная БД готова!")
    
    def add_documents(self, documents, metadata_list=None):
        try:
            if metadata_list is None:
                metadata_list = [{}] * len(documents)
            
            embeddings = self.embedder.encode(documents).tolist()
            
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata_list,
                ids=[f"doc_{i}" for i in range(len(documents))]
            )
            logger.info(f"Добавлено {len(documents)} документов")
            return True
        except Exception as e:
            logger.error(f"Ошибка при добавлении документов: {e}")
            return False
    
    def search_similar(self, query, top_k=TOP_K_RESULTS):
        try:
            query_embedding = self.embedder.encode([query]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            documents = results["documents"][0] if results["documents"] else []
            logger.info(f"Найдено {len(documents)} релевантных документов")
            return documents
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []''',
        
        "src/rag_system.py": '''import ollama
from vector_db import VectorStore
import logging
from datetime import datetime
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.vector_db = VectorStore()
        self.dialog_history = []
        logger.info(f"RAG система инициализирована с моделью: {model_name}")
    
    def build_prompt(self, query, context_documents, history):
        context = " ".join(context_documents) if context_documents else "Релевантная информация не найдена."
        recent_history = " ".join(history[-4:]) if history else "История диалога пуста."
        
        prompt = f"""Используй контекст для ответа на вопрос.

Контекст: {context}

История: {recent_history}

Вопрос: {query}

Ответ:"""
        return prompt
    
    def process_query(self, user_query):
        logger.info(f"Получен запрос: {user_query}")
        relevant_docs = self.vector_db.search_similar(user_query)
        prompt = self.build_prompt(user_query, relevant_docs, self.dialog_history)
        
        try:
            logger.info("Генерация ответа...")
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )
            answer = response['response'].strip()
            logger.info("Ответ сгенерирован")
        except Exception as e:
            error_msg = f"Ошибка: {e}"
            logger.error(error_msg)
            answer = error_msg
        
        self.dialog_history.extend([f"User: {user_query}", f"Assistant: {answer}"])
        return answer

    def add_initial_knowledge(self):
        """Добавление начальных знаний в систему"""
        initial_knowledge = [
            "Машинное обучение - это раздел искусственного интеллекта.",
            "Python популярный язык для программирования.",
            "RAG означает Retrieval-Augmented Generation.",
            "Оллaма - платформа для запуска языковых моделей."
        ]
        self.vector_db.add_documents(
            initial_knowledge,
            [{"source": "base_knowledge"}] * len(initial_knowledge)
        )''',
        
        "src/main.py": '''from rag_system import RAGSystem

def main():
    print("=" * 50)
    print("RAG System for Diplom Project")
    print("=" * 50)
    
    # Создаем систему
    rag = RAGSystem()
    
    # Добавляем начальные знания
    rag.add_initial_knowledge()
    
    print("Система готова! Введите вопрос (или 'quit' для выхода)")
    
    while True:
        user_input = input("\\nВаш вопрос: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("До свидания!")
            break
            
        if user_input:
            response = rag.process_query(user_input)
            print(f"Ответ: {response}")

if __name__ == "__main__":
    main()''',
        
        "README.md": '''# RAG System for Diplom

Простая RAG система для дипломного проекта.

## Установка
1. pip install -r requirements.txt
2. ollama serve
3. python src/main.py'''
    }
    
    # Создаем файлы
    for file_path, content in files_content.items():
        full_path = base_dir / file_path
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Создан файл: {full_path}")
        except Exception as e:
            print(f"❌ Ошибка создания {full_path}: {e}")
    
    print("\\n" + "="*50)
    print("🎉 СТРУКТУРА ПРОЕКТА СОЗДАНА!")
    print("="*50)
    print("📋 Дальнейшие шаги:")
    print("1. Откройте папку 'Diplom' в VS Code")
    print("2. В терминале VS Code выполните:")
    print("   python -m venv venv")
    print("   venv\\\\Scripts\\\\activate")
    print("   pip install -r requirements.txt")
    print("3. Убедитесь, что Ollama запущен")
    print("4. Запустите: python src/main.py")
    print("="*50)

if __name__ == "__main__":
    create_project_structure()