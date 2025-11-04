import logging
from datetime import datetime
from config import *
from mcp_client import MCPClient

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
                    server_info = self.mcp_client.get_server_info()
                    services = server_info.get("services", {})
                    
                    logger.info("🚀 RAG система с УНИВЕРСАЛЬНЫМ MCP клиентом")
                    logger.info(f"📊 Сервисы: БД({services.get('vector_db', 'unknown')}), Модели({services.get('llm_models', 'unknown')})")
                    
                    # Покажем доступные модели
                    available_models = self.mcp_client.list_models()
                    if available_models:
                        logger.info(f"📋 Модели на сервере: {', '.join(available_models)}")
                    else:
                        logger.warning("⚠️ На сервере нет доступных моделей")
                else:
                    logger.error("❌ MCP сервер недоступен! Запустите: python mcp_servers/ai_mcp_server.py")
                    self.use_mcp = False
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации MCP клиента: {e}")
                self.use_mcp = False
        else:
            from vector_db import VectorStore
            import ollama
            self.vector_db = VectorStore()
            self.ollama_client = ollama.Client()
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
            
            # Покажем информацию о сервере
            server_info = self.mcp_client.get_server_info()
            print(f"🔧 Сервер: {server_info.get('models_available', 0)} моделей доступно")

    def process_query(self, user_query: str) -> str:
        """Основной метод обработки запроса через MCP сервер"""
        logger.info(f"👤 Получен запрос: {user_query}")
        
        if self.use_mcp:
            # Используем полный RAG pipeline через MCP сервер
            result = self.mcp_client.rag_query(
                query=user_query,
                model=self.model_name,
                top_k=3
            )
            
            answer = result.get("answer", "Ошибка при обработке запроса")
            documents_found = result.get("documents_found", 0)
            timing = result.get("timing", {})
            
            logger.info(f"✅ RAG ответ: {documents_found} док., {timing.get('total', 0)} сек")
            
        else:
            # Старая логика с прямыми вызовами
            relevant_docs = self.vector_db.search_similar(user_query)
            context = "\n".join(relevant_docs) if relevant_docs else "Информация не найдена в базе знаний."
            
            prompt = f"""Ты - полезный ассистент с доступом к базе знаний. Ответь на вопрос используя контекст.

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ВОПРОС: {user_query}

Ответь кратко и по делу на основе контекста.

ОТВЕТ:"""
            
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt
            )
            answer = response['response'].strip()
        
        # Сохранение в историю и базу знаний
        self.dialog_history.extend([f"User: {user_query}", f"Assistant: {answer}"])
        
        # Сохранение в базу знаний (только хорошие ответы)
        if self.should_save_to_memory(user_query, answer):
            self.save_to_memory(user_query, answer)
        
        return answer
    
    def should_save_to_memory(self, query: str, response: str) -> bool:
        """Определяет, стоит ли сохранять ответ в память"""
        if not response or len(response) < 10:
            return False
        
        # Не сохраняем если ответ содержит части промпта
        forbidden_phrases = [
            "разъясняя ответ", "контекст из базы знаний", 
            "отправляй наш вопрос", "неизвестно", "не знаю",
            "контекст:", "вопрос:", "ответ:"
        ]
        
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in forbidden_phrases):
            return False
            
        # Не сохраняем приветствия
        if any(word in query.lower() for word in ["привет", "здравствуй", "hello"]):
            return False
            
        # Проверяем что ответ осмысленный (содержит законченные предложения)
        if len(response.split('.')) < 1:
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
            server_info = self.mcp_client.get_server_info()
            models = self.mcp_client.list_models()
            
            doc_count = db_info.get("document_count", 0)
            models_available = server_info.get("models_available", 0)
            
        else:
            db_info = self.vector_db.get_collection_info()
            doc_count = db_info.get("document_count", 0)
            models_available = 1  # только локальная модель
            models = [self.model_name]
            
        return {
            "model": self.model_name,
            "using_mcp": self.use_mcp,
            "dialog_history_length": len(self.dialog_history),
            "documents_in_db": doc_count,
            "models_available": models_available,
            "available_models": models,
            "mcp_available": self.mcp_client.is_server_running() if self.use_mcp else False
        }
    
    def test_model_generation(self, prompt: str = "Напиши коротко о искусственном интеллекте") -> str:
        """Тестирование генерации текста через MCP"""
        if self.use_mcp:
            return self.mcp_client.generate_text(prompt, self.model_name)
        else:
            response = self.ollama_client.generate(model=self.model_name, prompt=prompt)
            return response['response']