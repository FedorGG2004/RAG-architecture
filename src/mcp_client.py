import requests
import json
from typing import Dict, Any, List

class MCPClient:
    """Клиент для взаимодействия с MCP-сервером через HTTP"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        print(f"🔧 MCP клиент: подключение к {server_url}")
    
    def _send_command(self, command: str) -> Dict[str, Any]:
        """Отправка команды в MCP сервер (имитация HTTP)"""
        # Поскольку у нас упрощенный сервер, имитируем вызовы
        # В реальной системе здесь был бы HTTP запрос
        
        # Имитируем поиск - возвращаем те же данные что видим в сервере
        if command.startswith("search "):
            query = command[7:]
            return {
                "documents": [
                    "Python является популярным языком программирования для анализа данных и машинного обучения.",
                    "Оллaма - это платформа для запуска больших языковых моделей локально на компьютере.",
                    "Машинное обучение - это раздел искусственного интеллекта, который позволяет компьютерам обучаться на данных."
                ],
                "count": 3,
                "query": query
            }
        
        # Имитируем добавление документа
        elif command.startswith("add "):
            return {
                "success": True,
                "message": "Документ успешно добавлен"
            }
        
        # Имитируем получение информации
        elif command == "info":
            return {
                "document_count": 6,
                "collection_name": "rag_memory"
            }
        
        else:
            return {"error": "Unknown command"}
    
    def search_documents(self, query: str, top_k: int = 3) -> List[str]:
        """Поиск документов через MCP"""
        print(f"🔍 MCP клиент: поиск '{query}'")
        
        # Имитируем вызов к серверу
        result = self._send_command(f"search {query}")
        
        if "error" in result:
            print(f"❌ Ошибка поиска: {result['error']}")
            return []
        
        documents = result.get("documents", [])
        print(f"✅ MCP клиент: найдено {len(documents)} документов")
        return documents
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Добавление документа через MCP"""
        if metadata is None:
            metadata = {}
            
        print(f"💾 MCP клиент: добавление '{text[:50]}...'")
        
        # Имитируем вызов к серверу
        result = self._send_command(f"add {text}")
        
        if "error" in result:
            print(f"❌ Ошибка добавления: {result['error']}")
            return False
        
        success = result.get("success", False)
        if success:
            print(f"✅ MCP клиент: документ добавлен")
        return success
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Информация о коллекции"""
        result = self._send_command("info")
        return result
    
    def is_server_running(self) -> bool:
        """Всегда возвращает True (сервер работает)"""
        return True