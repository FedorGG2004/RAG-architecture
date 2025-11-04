import asyncio
import json
import uvicorn
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import ollama
from ollama import Client as OllamaClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== МОДЕЛИ ДАННЫХ ====================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class DocumentAddRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class GenerateRequest(BaseModel):
    model: str = "tinyllama:1.1b"
    prompt: str
    options: Optional[Dict[str, Any]] = None

class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str

class ChatRequest(BaseModel):
    model: str = "tinyllama:1.1b"
    messages: List[ChatMessage]

class RAGRequest(BaseModel):
    query: str
    model: str = "tinyllama:1.1b"
    top_k: int = 3

# ==================== ОСНОВНОЙ СЕРВЕР ====================

class AIMCPServer:
    def __init__(self):
        self.app = FastAPI(
            title="AI MCP Server",
            description="Универсальный MCP-сервер для векторной БД и LLM моделей",
            version="2.0.0"
        )
        
        # Инициализация векторной БД
        self._init_vector_db()
        
        # Инициализация LLM клиента
        self._init_llm_client()
        
        # Регистрация всех маршрутов
        self.setup_routes()
        
        logger.info("✅ AI MCP Server инициализирован (БД + Модели)")

    def _init_vector_db(self):
        """Инициализация векторной базы данных"""
        try:
            base_dir = Path(__file__).parent.parent
            db_path = base_dir / "data" / "chroma_db"
            self.client = chromadb.PersistentClient(path=str(db_path))
            self.collection = self.client.get_or_create_collection("rag_memory")
            self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ Векторная БД инициализирована")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации векторной БД: {e}")
            raise

    def _init_llm_client(self):
        """Инициализация клиента для работы с LLM моделями"""
        try:
            self.ollama_client = OllamaClient(host='http://localhost:11434')
            
            # Проверяем доступность Ollama
            models_response = self.ollama_client.list()
            logger.info(f"📦 Ответ от Ollama: {models_response}")
            
            # Обрабатываем разные форматы ответа
            if 'models' in models_response:
                models_list = models_response['models']
                model_count = len(models_list)
                
                # Извлекаем имена моделей безопасно
                self.available_models = []
                for model in models_list:
                    if 'name' in model:
                        self.available_models.append(model['name'])
                    elif 'model' in model:
                        self.available_models.append(model['model'])
                
                logger.info(f"✅ LLM клиент инициализирован. Доступно моделей: {model_count}")
                logger.info(f"📋 Доступные модели: {self.available_models}")
                
            else:
                # Если структура ответа неожиданная
                logger.warning("⚠️ Неожиданный формат ответа от Ollama")
                self.available_models = []
                
            # Если нет моделей, создаем пустой список
            if not hasattr(self, 'available_models'):
                self.available_models = []
                
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации LLM клиента: {e}")
            logger.warning("⚠️  Убедитесь, что Ollama запущен: ollama serve")
            
            # Создаем атрибут даже при ошибке
            self.available_models = []
            
            # Не прерываем инициализацию - сервер может работать без моделей
            logger.warning("🔄 Продолжаем инициализацию сервера без LLM моделей")
            
    def setup_routes(self):
        """Регистрация всех API эндпоинтов"""
        
        # ==================== БАЗОВЫЕ ЭНДПОИНТЫ ====================
        
        @self.app.get("/")
        async def root():
            return {
                "message": "AI MCP Server is running!",
                "version": "2.0.0",
                "services": ["vector_db", "llm_models"]
            }

        @self.app.get("/health")
        async def health_check():
            db_status = "healthy" if hasattr(self, 'collection') else "unhealthy"
            llm_status = "healthy" if hasattr(self, 'ollama_client') and self.available_models else "unhealthy"
            
            return {
                "status": "healthy",
                "services": {
                    "vector_db": db_status,
                    "llm_models": llm_status
                },
                "models_available": len(self.available_models)
            }

        # ==================== ВЕКТОРНАЯ БД ЭНДПОИНТЫ ====================
        
        @self.app.post("/search", response_model=Dict)
        async def search_documents(request: SearchRequest):
            """Поиск документов по семантическому сходству"""
            start_time = time.time()
            try:
                logger.info(f"🔍 Поиск документов: '{request.query}'")
                
                # Замер времени векторизации
                vector_start = time.time()
                query_embedding = self.embedder.encode([request.query]).tolist()
                vector_time = time.time() - vector_start
                
                # Замер времени поиска в БД
                search_start = time.time()
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=request.top_k
                )
                search_time = time.time() - search_start
                
                documents = results["documents"][0] if results["documents"] else []
                total_time = time.time() - start_time
                
                logger.info(f"✅ Найдено {len(documents)} документов за {total_time:.3f} сек")
                
                return {
                    "documents": documents,
                    "count": len(documents),
                    "query": request.query,
                    "timing": {
                        "total": round(total_time, 3),
                        "vectorization": round(vector_time, 3),
                        "search": round(search_time, 3)
                    }
                }
                
            except Exception as e:
                logger.error(f"❌ Ошибка поиска: {e}")
                raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

        @self.app.post("/add", response_model=Dict)
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
                
                return {
                    "success": True,
                    "message": "Документ успешно добавлен",
                    "doc_id": doc_id,
                    "text_length": len(request.text)
                }
                
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

        # ==================== LLM МОДЕЛИ ЭНДПОИНТЫ ====================
        
        @self.app.post("/generate", response_model=Dict)
        async def generate_text(request: GenerateRequest):
            """Генерация текста через LLM модель"""
            start_time = time.time()
            try:
                logger.info(f"🤖 Генерация текста моделью: {request.model}")
                
                if request.model not in self.available_models:
                    raise HTTPException(status_code=400, detail=f"Модель {request.model} не доступна")
                
                response = self.ollama_client.generate(
                    model=request.model,
                    prompt=request.prompt,
                    options=request.options or {}
                )
                
                generation_time = time.time() - start_time
                logger.info(f"✅ Текст сгенерирован за {generation_time:.3f} сек")
                
                return {
                    "response": response['response'],
                    "model": request.model,
                    "prompt_length": len(request.prompt),
                    "generation_time": round(generation_time, 3),
                    "total_duration": response.get('total_duration', 0)
                }
                
            except Exception as e:
                logger.error(f"❌ Ошибка генерации: {e}")
                raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

        @self.app.post("/chat", response_model=Dict)
        async def chat_completion(request: ChatRequest):
            """Чат-комплишн через LLM модель"""
            start_time = time.time()
            try:
                logger.info(f"💬 Чат-запрос к модели: {request.model}")
                
                if request.model not in self.available_models:
                    raise HTTPException(status_code=400, detail=f"Модель {request.model} не доступна")
                
                response = self.ollama_client.chat(
                    model=request.model,
                    messages=[{"role": msg.role, "content": msg.content} for msg in request.messages]
                )
                
                chat_time = time.time() - start_time
                logger.info(f"✅ Чат-ответ получен за {chat_time:.3f} сек")
                
                return {
                    "message": response['message'],
                    "model": request.model,
                    "chat_time": round(chat_time, 3)
                }
                
            except Exception as e:
                logger.error(f"❌ Ошибка чата: {e}")
                raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

        @self.app.get("/models")
        async def list_models():
            """Получение списка доступных моделей"""
            try:
                models_response = self.ollama_client.list()
                
                # Безопасно извлекаем модели
                if 'models' in models_response:
                    models_list = models_response['models']
                    # Форматируем ответ безопасно
                    formatted_models = []
                    for model in models_list:
                        model_info = {
                            'name': model.get('name', model.get('model', 'unknown')),
                            'size': model.get('size', 0),
                            'modified': model.get('modified_at', '')
                        }
                        formatted_models.append(model_info)
                        
                    return {
                        "models": formatted_models,
                        "count": len(formatted_models)
                    }
                else:
                    return {
                        "models": [],
                        "count": 0,
                        "warning": "Неожиданный формат ответа от Ollama"
                    }
                    
            except Exception as e:
                logger.error(f"❌ Ошибка получения моделей: {e}")
                return {
                    "models": [],
                    "count": 0,
                    "error": str(e)
                }

        # ==================== RAG ЭНДПОИНТЫ ====================
        
        @self.app.post("/rag", response_model=Dict)
        async def rag_query(request: RAGRequest):
            """Полный RAG pipeline: поиск + генерация"""
            start_time = time.time()
            try:
                logger.info(f"🎯 RAG запрос: '{request.query}'")
                
                # 1. Поиск в векторной БД
                search_start = time.time()
                query_embedding = self.embedder.encode([request.query]).tolist()
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=request.top_k
                )
                search_time = time.time() - search_start
                
                documents = results["documents"][0] if results["documents"] else []
                
                # 2. Формирование промпта
                context = "\n".join(documents) if documents else "Информация не найдена в базе знаний."
                
                prompt = f"""Используя приведенный контекст, ответь на вопрос пользователя. 
Если в контексте нет нужной информации, скажи что не знаешь.

Контекст: {context}

Вопрос: {request.query}

Ответ:"""
                
                # 3. Генерация ответа
                gen_start = time.time()
                response = self.ollama_client.generate(
                    model=request.model,
                    prompt=prompt
                )
                gen_time = time.time() - gen_start
                
                total_time = time.time() - start_time
                
                logger.info(f"✅ RAG ответ сгенерирован за {total_time:.3f} сек")
                
                return {
                    "answer": response['response'],
                    "documents_found": len(documents),
                    "model": request.model,
                    "query": request.query,
                    "timing": {
                        "total": round(total_time, 3),
                        "search": round(search_time, 3),
                        "generation": round(gen_time, 3)
                    }
                }
                
            except Exception as e:
                logger.error(f"❌ Ошибка RAG: {e}")
                raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

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
    """Запуск универсального MCP сервера"""
    try:
        server = AIMCPServer()
        
        print("🚀 Запуск AI MCP Server на http://localhost:8000")
        print("📚 Объединенные сервисы:")
        print("   ├── Векторная БД (ChromaDB)")
        print("   ├── LLM Модели (Ollama)") 
        print("   └── RAG Pipeline")
        print("")
        print("🌐 API документация: http://localhost:8000/docs")
        print("❤️  Проверка здоровья: http://localhost:8000/health")
        print("")
        print("⚡ Для остановки сервера нажмите Ctrl+C")
        print("-" * 50)
        
        uvicorn.run(
            server.app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Не удалось запустить сервер: {e}")
        print("🔧 Проверьте что:")
        print("   - Ollama запущен: ollama serve")
        print("   - ChromaDB может быть создана")
        print("   - Порт 8000 свободен")

if __name__ == "__main__":
    main()