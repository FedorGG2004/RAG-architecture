import requests
import json
from typing import Dict, Any, List, Optional
import logging
from time import sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    """Настоящий клиент для взаимодействия с MCP-сервером через HTTP"""
    
    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 30):
        self.server_url = server_url
        self.timeout = timeout
        self.session = requests.Session()
        
        # Заголовки для JSON API
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RAG-System/1.0'
        })
        
        logger.info(f"🔧 MCP клиент: подключение к {server_url}")
        
        # Проверка соединения при инициализации
        self._wait_for_server()

    def _wait_for_server(self, max_retries: int = 10, retry_delay: int = 2):
        """Ожидание запуска сервера с повторными попытками"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.server_url}/health",
                    timeout=5
                )
                if response.status_code == 200:
                    logger.info("✅ MCP сервер доступен!")
                    return True
            except requests.exceptions.ConnectionError:
                if attempt == 0:
                    logger.warning(f"⏳ MCP сервер не отвечает, попытка подключения...")
                else:
                    logger.warning(f"⏳ Попытка {attempt + 1}/{max_retries}...")
            
            if attempt < max_retries - 1:
                sleep(retry_delay)
        
        logger.error(f"❌ Не удалось подключиться к MCP серверу после {max_retries} попыток")
        return False

    def search_documents(self, query: str, top_k: int = 3) -> List[str]:
        """Настоящий поиск документов через MCP сервер"""
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
                logger.info(f"✅ MCP клиент: найдено {len(documents)} документов")
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
        """Настоящее добавление документа через MCP сервер"""
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
                    logger.info(f"✅ MCP клиент: документ добавлен (ID: {result.get('doc_id')})")
                else:
                    logger.warning(f"⚠️ MCP клиент: документ не добавлен")
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

    def batch_add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Пакетное добавление документов"""
        try:
            logger.info(f"💾 MCP клиент: пакетное добавление {len(documents)} документов")
            
            payload = [{"text": doc["text"], "metadata": doc.get("metadata", {})} for doc in documents]
            
            response = self.session.post(
                f"{self.server_url}/batch_add",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                if success:
                    logger.info(f"✅ MCP клиент: добавлено {result.get('count', 0)} документов")
                return success
            else:
                logger.error(f"❌ Ошибка пакетного добавления: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка пакетного добавления: {e}")
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

    def __del__(self):
        """Закрытие сессии при уничтожении объекта"""
        if hasattr(self, 'session'):
            self.session.close()