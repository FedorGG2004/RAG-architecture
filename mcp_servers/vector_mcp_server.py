import asyncio
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import time
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модели данных для API
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class DocumentAddRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    documents: List[str]
    count: int
    query: str

class AddResponse(BaseModel):
    success: bool
    message: str
    doc_id: Optional[str] = None

class VectorMCPServer:
    def __init__(self):
        self.app = FastAPI(
            title="Vector MCP Server",
            description="MCP-сервер для работы с векторной базой данных",
            version="1.0.0"
        )
        
        # Инициализация векторной БД
        base_dir = Path(__file__).parent.parent
        db_path = base_dir / "data" / "chroma_db"
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection("rag_memory")
        
        # Загрузка модели для эмбеддингов
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        # Регистрация маршрутов
        self.setup_routes()
        
        logger.info("✅ MCP Vector Server инициализирован")

    def setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"message": "Vector MCP Server is running!"}

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "vector_mcp_server"}

        @self.app.post("/search", response_model=SearchResponse)
        async def search_documents(request: SearchRequest):
            start_time = time.time()
            logger.info(f"🔍 Поиск документов для запроса: '{request.query}'")
            
            try:
                # Замер времени векторизации
                vector_start = time.time()
                query_embedding = self.embedder.encode([request.query]).tolist()
                vector_time = time.time() - vector_start
                logger.info(f"⏱️ Векторизация заняла: {vector_time:.3f} сек")
                
                # Замер времени поиска в БД
                search_start = time.time()
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=request.top_k
                )
                search_time = time.time() - search_start
                logger.info(f"⏱️ Поиск в БД занял: {search_time:.3f} сек")
                
                documents = results["documents"][0] if results["documents"] else []
                
                # Общее время
                total_time = time.time() - start_time
                logger.info(f"✅ Найдено {len(documents)} документов за {total_time:.3f} сек")
                
                return SearchResponse(
                    documents=documents,
                    count=len(documents),
                    query=request.query
                )
                
            except Exception as e:
                total_time = time.time() - start_time
                logger.error(f"❌ Ошибка поиска за {total_time:.3f} сек: {e}")
                raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

        @self.app.post("/add", response_model=AddResponse)
        async def add_document(request: DocumentAddRequest):
            """Добавление документа в векторную БД"""
            try:
                logger.info(f"💾 Добавление документа: {request.text[:50]}...")
                
                if request.metadata is None:
                    request.metadata = {"source": "mcp_api", "type": "fact"}
                
                # Преобразуем текст в вектор
                embedding = self.embedder.encode([request.text]).tolist()
                
                # Создаем ID документа
                doc_id = f"doc_{hash(request.text) % 1000000}"
                
                # Добавляем в базу данных
                self.collection.add(
                    embeddings=embedding,
                    documents=[request.text],
                    metadatas=[request.metadata],
                    ids=[doc_id]
                )
                
                logger.info(f"✅ Документ добавлен с ID: {doc_id}")
                
                return AddResponse(
                    success=True,
                    message="Документ успешно добавлен",
                    doc_id=doc_id
                )
                
            except Exception as e:
                logger.error(f"❌ Ошибка добавления: {e}")
                raise HTTPException(status_code=500, detail=f"Add error: {str(e)}")

        @self.app.get("/info")
        async def get_collection_info():
            """Получение информации о коллекции"""
            try:
                count = self.collection.count()
                return {
                    "document_count": count,
                    "collection_name": "rag_memory",
                    "status": "active"
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Info error: {str(e)}")

        @self.app.post("/batch_add")
        async def batch_add_documents(documents: List[DocumentAddRequest]):
            """Пакетное добавление документов"""
            try:
                texts = [doc.text for doc in documents]
                metadatas = [doc.metadata or {"source": "batch_mcp", "type": "fact"} for doc in documents]
                
                # Пакетное кодирование
                embeddings = self.embedder.encode(texts).tolist()
                
                # Генерация ID
                doc_ids = [f"batch_{hash(text) % 1000000}" for text in texts]
                
                # Пакетное добавление
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=doc_ids
                )
                
                return {
                    "success": True,
                    "message": f"Добавлено {len(documents)} документов",
                    "count": len(documents)
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Batch add error: {str(e)}")

def main():
    """Запуск MCP сервера"""
    server = VectorMCPServer()
    
    print("🚀 Запуск MCP Vector Server на http://localhost:8000")
    print("📚 Доступные эндпоинты:")
    print("   GET  /health - проверка здоровья")
    print("   POST /search - поиск документов")
    print("   POST /add    - добавление документа")
    print("   GET  /info   - информация о коллекции")
    
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()