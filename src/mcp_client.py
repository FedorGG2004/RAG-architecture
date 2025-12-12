import requests
import json
from typing import Dict, Any, List, Optional
import logging
from time import sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    """Универсальный клиент для работы с AI MCP сервером"""
    
    def __init__(self, server_url: str = "http://ai-dev.hpclab:8000", timeout: int = 120):
        self.server_url = server_url
        self.timeout = timeout
        self.session = requests.Session()
        
        # Заголовки для JSON API
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RAG-System/2.0'
        })
        
        logger.info(f"🔧 Универсальный MCP клиент: подключение к {server_url}")
        
        # Проверка соединения при инициализации
        self._wait_for_server()

    def _wait_for_server(self, max_retries: int = 10, retry_delay: int = 3):
        """Ожидание запуска сервера с повторными попытками"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.server_url}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    health_data = response.json()
                    logger.info("✅ AI MCP сервер доступен!")
                    
                    # Проверяем статусы сервисов
                    services = health_data.get("services", {})
                    db_status = services.get("vector_db", "unknown")
                    llm_status = services.get("llm_models", "unknown")
                    
                    logger.info(f"📊 Статусы сервисов:")
                    logger.info(f"   - Векторная БД: {db_status}")
                    logger.info(f"   - LLM Модели: {llm_status}")
                    logger.info(f"   - Моделей доступно: {health_data.get('models_available', 0)}")
                    
                    return True
            except requests.exceptions.ConnectionError:
                if attempt == 0:
                    logger.warning(f"⏳ AI MCP сервер не отвечает, попытка подключения...")
                else:
                    logger.warning(f"⏳ Попытка {attempt + 1}/{max_retries}...")
            
            if attempt < max_retries - 1:
                sleep(retry_delay)
        
        logger.error(f"❌ Не удалось подключиться к AI MCP серверу после {max_retries} попыток")
        return False

    # ==================== ВЕКТОРНАЯ БД МЕТОДЫ ====================
    
    def search_documents(self, query: str, top_k: int = 3) -> List[str]:
        """Поиск документов через MCP сервер"""
        try:
            logger.info(f"🔍 MCP клиент: поиск '{query}'")
            
            payload = {
                "query": query,
                "top_k": top_k
            }
            
            response = self.session.post(
                f"{self.server_url}/search",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                documents = result.get("documents", [])
                timing = result.get("timing", {})
                
                logger.info(f"✅ Найдено {len(documents)} документов за {timing.get('total', 0)} сек")
                return documents
            else:
                logger.error(f"❌ Ошибка поиска: {response.status_code} - {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Сетевая ошибка при поиске: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка при поиске: {e}")
            return []

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Добавление документа через MCP сервер"""
        try:
            if metadata is None:
                metadata = {"source": "rag_system", "type": "fact"}
            
            logger.info(f"💾 MCP клиент: добавление '{text[:50]}...'")
            
            payload = {
                "text": text,
                "metadata": metadata
            }
            
            response = self.session.post(
                f"{self.server_url}/add",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                if success:
                    doc_id = result.get("doc_id")
                    logger.info(f"✅ Документ добавлен (ID: {doc_id})")
                else:
                    logger.warning(f"⚠️ Документ не добавлен")
                return success
            else:
                logger.error(f"❌ Ошибка добавления: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Сетевая ошибка при добавлении: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка при добавлении: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Получение информации о коллекции"""
        try:
            response = self.session.get(
                f"{self.server_url}/info",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Ошибка получения информации: {response.status_code}")
                return {"document_count": 0, "error": "server_error"}
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения информации: {e}")
            return {"document_count": 0, "error": str(e)}

    # ==================== LLM МОДЕЛИ МЕТОДЫ ====================
    
    def generate_text(self, prompt: str, model: str = "tinyllama:1.1b", options: Optional[Dict] = None) -> str:
        """Генерация текста через MCP сервер"""
        try:
            logger.info(f"🤖 MCP клиент: генерация текста моделью {model}")
            
            payload = {
                "model": model,
                "prompt": prompt,
                "options": options or {}
            }
            
            response = self.session.post(
                f"{self.server_url}/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                gen_time = result.get("generation_time", 0)
                
                logger.info(f"✅ Текст сгенерирован за {gen_time} сек")
                return response_text
            else:
                logger.error(f"❌ Ошибка генерации: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            return ""

    def chat_completion(self, messages: List[Dict], model: str = "tinyllama:1.1b") -> str:
        """Чат-комплишн через MCP сервер"""
        try:
            logger.info(f"💬 MCP клиент: чат-запрос к модели {model}")
            
            # Преобразуем сообщения в формат Pydantic
            chat_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
            
            payload = {
                "model": model,
                "messages": chat_messages
            }
            
            response = self.session.post(
                f"{self.server_url}/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", {})
                chat_time = result.get("chat_time", 0)
                
                logger.info(f"✅ Чат-ответ получен за {chat_time} сек")
                return message.get("content", "")
            else:
                logger.error(f"❌ Ошибка чата: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Ошибка чата: {e}")
            return ""

    def list_models(self) -> List[str]:
        """Получение списка доступных моделей"""
        try:
            response = self.session.get(
                f"{self.server_url}/models",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                models = result.get("models", [])
                model_names = [model['name'] for model in models]
                
                logger.info(f"📋 Доступно моделей: {len(model_names)}")
                return model_names
            else:
                logger.error(f"❌ Ошибка получения моделей: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения моделей: {e}")
            return []

    # ==================== RAG МЕТОДЫ ====================
    
    def rag_query(self, query: str, model: str = "tinyllama:1.1b", top_k: int = 3) -> Dict[str, Any]:
        """Полный RAG pipeline через MCP сервер"""
        try:
            logger.info(f"🎯 MCP клиент: RAG запрос '{query}'")
            
            payload = {
                "query": query,
                "model": model,
                "top_k": top_k
            }
            
            response = self.session.post(
                f"{self.server_url}/rag",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                timing = result.get("timing", {})
                
                logger.info(f"✅ RAG ответ получен за {timing.get('total', 0)} сек")
                return result
            else:
                logger.error(f"❌ Ошибка RAG: {response.status_code} - {response.text}")
                return {"answer": "Ошибка при обработке запроса", "documents_found": 0}
                
        except Exception as e:
            logger.error(f"❌ Ошибка RAG: {e}")
            return {"answer": "Ошибка при обработке запроса", "documents_found": 0}

    def is_server_running(self) -> bool:
        """Проверка доступности сервера"""
        try:
            response = self.session.get(
                f"{self.server_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def get_server_info(self) -> Dict[str, Any]:
        """Полная информация о сервере"""
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}

    def __del__(self):
        """Закрытие сессии при уничтожении объекта"""
        if hasattr(self, 'session'):
            self.session.close()