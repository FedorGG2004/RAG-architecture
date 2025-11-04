import ollama
import logging
from datetime import datetime
from config import *
from mcp_client import MCPClient  # Теперь это настоящий клиент!

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, model_name=MODEL_NAME, use_mcp=True):
        self.model_name = model_name
        self.dialog_history = []
        self.use_mcp = use_mcp
        
        if self.use_mcp:
            try:
                self.mcp_client = MCPClient()
                if self.mcp_client.is_server_running():
                    logger.info("🚀 RAG система с НАСТОЯЩИМ MCP клиентом")
                else:
                    logger.error("❌ MCP сервер недоступен! Запустите: python mcp_servers/vector_mcp_server.py")
                    self.use_mcp = False
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации MCP клиента: {e}")
                self.use_mcp = False
        else:
            from vector_db import VectorStore
            self.vector_db = VectorStore()
            logger.info("🔧 RAG система с прямыми вызовами")
    
    def add_initial_knowledge(self):
        """Добавление начальных знаний через MCP сервер"""
        initial_knowledge = [
            "Машинное обучение - это раздел искусственного интеллекта, который позволяет компьютерам обучаться на данных.",
            "Python является популярным языком программирования для анализа данных и машинного обучения.",
            "RAG (Retrieval-Augmented Generation) - это архитектура, которая сочетает поиск информации и генерацию текста.",
            "Оллaма - это платформа для запуска больших языковых моделей локально на компьютере.",
            "Векторная база данных хранит информацию в виде числовых векторов для семантического поиска."
        ]
        
        print("📚 Загрузка начальной базы знаний через MCP...")
        
        success_count = 0
        for knowledge in initial_knowledge:
            if self.use_mcp:
                success = self.mcp_client.add_document(
                    knowledge, 
                    {"source": "base_knowledge", "type": "fact"}
                )
                if success:
                    success_count += 1
                    print(f"✅ Добавлено через MCP: {knowledge[:50]}...")
                else:
                    print(f"❌ Ошибка добавления: {knowledge[:50]}...")
            else:
                self.vector_db.add_documents(
                    [knowledge], 
                    [{"source": "base_knowledge", "type": "fact"}]
                )
                success_count += 1
        
        print(f"📊 Итого добавлено документов: {success_count}/{len(initial_knowledge)}")
        
        # Покажем информацию о коллекции
        if self.use_mcp:
            info = self.mcp_client.get_collection_info()
            print(f"📈 В базе теперь: {info.get('document_count', 0)} документов")

    # Остальные методы остаются без изменений, но теперь используют НАСТОЯЩИЙ MCP
    
    def process_query(self, user_query: str) -> str:
        """Основной метод обработки запроса"""
        logger.info(f"👤 Получен запрос: {user_query}")
        
        # 1. Поиск релевантной информации
        if self.use_mcp:
            relevant_docs = self.mcp_client.search_documents(user_query)
        else:
            relevant_docs = self.vector_db.search_similar(user_query)
        
        # 2. Формирование промита
        context = "\n".join(relevant_docs) if relevant_docs else "Информация не найдена в базе знаний."
        
        prompt = f"""Ты - полезный ассистент с доступом к базе знаний. Ответь на вопрос используя контекст.

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ВОПРОС: {user_query}

Ответь кратко и по делу на основе контекста. Если в контексте нет информации, скажи "Я не нашел информации по этому вопросу в базе знаний".

ОТВЕТ:"""
        
        # 3. Генерация ответа через Ollama
        try:
            logger.info("🤖 Генерация ответа...")
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_thread': 6,
                    'num_predict': 150,
                    'temperature': 0.1
                }
            )
            answer = response['response'].strip()
            logger.info("✅ Ответ сгенерирован")
        except Exception as e:
            error_msg = f"Ошибка при генерации ответа: {e}"
            logger.error(f"❌ {error_msg}")
            answer = error_msg
        
        # 4. Сохранение в историю и базу знаний
        self.dialog_history.extend([f"User: {user_query}", f"Assistant: {answer}"])
        
        # 5. Сохранение в базу знаний (только хорошие ответы)
        if self.should_save_to_memory(user_query, answer):
            self.save_to_memory(user_query, answer)
        
        return answer
    
    def should_save_to_memory(self, query: str, response: str) -> bool:
        """Определяет, стоит ли сохранять ответ в память"""
        if not response or len(response) < 20:
            return False
        if any(word in response.lower() for word in ["не знаю", "извините", "ошибка"]):
            return False
        if any(word in query.lower() for word in ["привет", "здравствуй", "hello"]):
            return False
        return True
    
    def save_to_memory(self, query: str, response: str):
        """Сохранение информации в базу знаний"""
        try:
            facts_to_save = [
                f"Вопрос: {query}",
                f"Ответ: {response}",
            ]
            
            metadata = {
                "type": "dialog",
                "timestamp": datetime.now().isoformat(),
                "source": "generated"
            }
            
            for fact in facts_to_save:
                if self.use_mcp:
                    self.mcp_client.add_document(fact, metadata)
                else:
                    self.vector_db.add_documents([fact], [metadata])
                    
            logger.info("💾 Информация сохранена в память")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения в память: {e}")
    
    def get_system_info(self) -> dict:
        """Получение информации о системе"""
        if self.use_mcp:
            db_info = self.mcp_client.get_collection_info()
            doc_count = db_info.get("document_count", 0)
        else:
            db_info = self.vector_db.get_collection_info()
            doc_count = db_info.get("document_count", 0)
            
        return {
            "model": self.model_name,
            "using_mcp": self.use_mcp,
            "dialog_history_length": len(self.dialog_history),
            "documents_in_db": doc_count,
            "mcp_available": self.mcp_client.is_server_running() if self.use_mcp else False
        }