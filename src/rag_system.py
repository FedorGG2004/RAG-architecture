import logging
import re
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
        self.user_preferences = {}  # Хранилище персональных предпочтений
        
        if self.use_mcp:
            try:
                self.mcp_client = MCPClient()
                if self.mcp_client.is_server_running():
                    server_info = self.mcp_client.get_server_info()
                    services = server_info.get("services", {})
                    
                    print("=" * 60)
                    print("🚀 RAG система с УНИВЕРСАЛЬНЫМ MCP клиентом")
                    print(f"📊 Сервисы: БД({services.get('vector_db', 'unknown')}), Модели({services.get('llm_models', 'unknown')})")
                    print(f"🌐 Интернет-поиск: {services.get('internet_search', 'unknown')}")
                    print("=" * 60)
                    
                    # Покажем доступные модели
                    available_models = self.mcp_client.list_models()
                    if available_models:
                        print(f"📋 Модели на сервере: {', '.join(available_models)}")
                    else:
                        print("⚠️ На сервере нет доступных моделей")
                else:
                    print("❌ MCP сервер недоступен! Запустите: python mcp_servers/ai_mcp_server.py")
                    self.use_mcp = False
            except Exception as e:
                print(f"❌ Ошибка инициализации MCP клиента: {e}")
                self.use_mcp = False
        else:
            from vector_db import VectorStore
            import ollama
            self.vector_db = VectorStore()
            self.ollama_client = ollama.Client()
            print("🔧 RAG система с прямыми вызовами")
    
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
        print(f"\n{'='*60}")
        print(f"👤 ПОЛЬЗОВАТЕЛЬ: {user_query}")
        print(f"{'='*60}")
        
        logger.info(f"👤 Получен запрос: {user_query}")
        
        # Проверка на персональные данные перед отправкой на сервер
        personal_fact = self.extract_personal_fact(user_query)
        if personal_fact:
            # Сохраняем персональный факт локально
            key, value = personal_fact
            self.user_preferences[key] = value
            logger.info(f"💾 Сохранено предпочтение: {key} = {value}")
            response = f"✅ Запомнил! Ваше {key}: {value}"
            print(f"[СИСТЕМА] {response}")
            return response
        
        # Проверка запроса о сохраненных предпочтениях
        preference_answer = self.check_user_preferences(user_query)
        if preference_answer:
            print(f"[СИСТЕМА] {preference_answer}")
            return preference_answer
        
        # Обычный RAG запрос через сервер
        if self.use_mcp:
            print(f"[ПОИСК] 🔍 Ищу информацию по запросу: '{user_query}'")
            result = self.mcp_client.rag_query(
                query=user_query,
                model=self.model_name,
                top_k=3
            )
            
            answer = result.get("answer", "Ошибка при обработке запроса")
            documents_found = result.get("documents_found", 0)
            internet_results = result.get("internet_results", 0)
            source = result.get("source", "unknown")
            timing = result.get("timing", {})
            
            # Подробное логирование
            print(f"[БАЗА ДАННЫХ] 📊 Найдено документов: {documents_found}")
            
            if internet_results > 0:
                print(f"[ИНТЕРНЕТ] 🌐 Найдено результатов: {internet_results}")
                print(f"[СИСТЕМА] 💾 Сохраняю новую информацию в базу знаний...")
            
            if source == "интернет" or source == "база знаний + интернет":
                print(f"[ИСТОЧНИК] 📍 Основной источник: интернет")
            elif source == "база знаний":
                print(f"[ИСТОЧНИК] 📍 Основной источник: база знаний")
            else:
                print(f"[ИСТОЧНИК] 📍 Основной источник: {source}")
            
            print(f"[ВРЕМЯ] ⏱️ Общее: {timing.get('total', 0):.2f} сек")
            print(f"[ВРЕМЯ] 🔎 Поиск: {timing.get('search', 0):.2f} сек")
            if internet_results > 0:
                print(f"[ВРЕМЯ] 🌐 Интернет-поиск: {timing.get('internet_search', 0):.2f} сек")
            print(f"[ВРЕМЯ] 🤖 Генерация: {timing.get('generation', 0):.2f} сек")
            
            logger.info(f"✅ RAG ответ: {documents_found} док., {internet_results} интернет, {timing.get('total', 0)} сек")
            
        else:
            # Режим без MCP
            print(f"[ПОИСК] 🔍 Ищу информацию в локальной базе данных...")
            relevant_docs = self.vector_db.search_similar(user_query)
            context = "\n".join(relevant_docs) if relevant_docs else "Информация не найдена в базе знаний."
            
            if not relevant_docs:
                print(f"[СИСТЕМА] 📭 Информация не найдена в базе знаний")
            
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=f"Контекст: {context}\n\nВопрос: {user_query}\n\nОтвет:"
            )
            answer = response['response'].strip()
        
        # Сохранение в историю
        self.dialog_history.extend([f"User: {user_query}", f"Assistant: {answer}"])
        
        # Сохранение в базу знаний
        if self.should_save_to_memory(user_query, answer):
            print(f"[СОХРАНЕНИЕ] 💾 Сохраняю вопрос и ответ в базу знаний...")
            self.save_to_memory(user_query, answer)
        
        return answer
    
    def extract_personal_fact(self, query: str):
        """Извлекает персональные факты из запроса"""
        query_lower = query.lower()
        
        # Улучшенные паттерны для извлечения фактов
        patterns = [
            (r"мо[ёе] любимое животное[:\s\-—]*([^.?!]+)", "любимое животное"),
            (r"любимое животное[:\s\-—]*([^.?!]+)", "любимое животное"),
            (r"запомни[,]? мо[ёе] любимое животное[:\s\-—]*([^.?!]+)", "любимое животное"),
            (r"мо[ёе] любимое животное это ([^.?!]+)", "любимое животное"),
            (r"я люблю ([^.?!]+)", "любимое животное"),
            (r"мне нравится ([^.?!]+)", "нравится"),
            (r"моя любимая еда[:\s\-—]*([^.?!]+)", "любимая еда"),
            (r"любимая еда[:\s\-—]*([^.?!]+)", "любимая еда"),
        ]
        
        for pattern, fact_type in patterns:
            match = re.search(pattern, query_lower)
            if match:
                value = match.group(1).strip()
                if value and len(value) > 1:
                    # Очищаем значение
                    value = value.replace('это', '').replace('—', '').replace('-', '').strip()
                    return (fact_type, value)
        
        # Проверяем прямые упоминания животных
        animals = ["кенгуру", "собака", "кошка", "черепаха", "тигр", "лев", "слон", "обезьяна"]
        for animal in animals:
            if animal in query_lower and "любим" in query_lower:
                return ("любимое животное", animal)
        
        return None
    
    def check_user_preferences(self, query: str) -> str:
        """Проверяет запрос о сохраненных предпочтениях пользователя"""
        query_lower = query.lower()
        
        if "мое любимое животное" in query_lower or "моё любимое животное" in query_lower:
            if "любимое животное" in self.user_preferences:
                return f"✅ Ваше любимое животное: {self.user_preferences['любимое животное']}"
            else:
                return "🤔 Я не знаю ваше любимое животное. Скажите мне, и я запомню!"
        
        if "какие у меня предпочтения" in query_lower or "что я говорил" in query_lower:
            if self.user_preferences:
                prefs = "\n".join([f"• {k}: {v}" for k, v in self.user_preferences.items()])
                return f"📋 Ваши предпочтения:\n{prefs}"
            else:
                return "📝 У вас пока нет сохраненных предпочтений."
        
        return ""
    
    def should_save_to_memory(self, query: str, response: str) -> bool:
        """Определяет, стоит ли сохранять ответ в память"""
        if not response or len(response) < 10:
            return False
        
        # Не сохраняем если ответ содержит части промпта
        forbidden_phrases = [
            "разъясняя ответ", "контекст из базы знаний", 
            "отправляй наш вопрос", "неизвестно", "не знаю",
            "контекст:", "вопрос:", "ответ:", "не нашел информации"
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
            
        # Не сохраняем короткие ответы
        if len(response) < 20:
            return False
            
        return True
    
    def save_to_memory(self, query: str, response: str):
        """Сохранение информации в базу знаний"""
        try:
            facts_to_save = [
                f"Вопрос пользователя: {query}",
                f"Ответ ассистента: {response}",
            ]
            
            metadata = {
                "type": "dialog",
                "timestamp": datetime.now().isoformat(),
                "source": "generated",
                "query": query[:50]
            }
            
            for fact in facts_to_save:
                if self.use_mcp:
                    success = self.mcp_client.add_document(fact, metadata)
                    if success:
                        print(f"[СОХРАНЕНИЕ] ✅ Информация сохранена в базу знаний")
                    else:
                        print(f"[СОХРАНЕНИЕ] ❌ Ошибка сохранения")
                else:
                    self.vector_db.add_documents([fact], [metadata])
                    
            logger.info("💾 Информация сохранена в память")
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения в память: {e}")
            print(f"[СОХРАНЕНИЕ] ❌ Ошибка: {e}")
    
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
            "user_preferences": self.user_preferences,
            "mcp_available": self.mcp_client.is_server_running() if self.use_mcp else False
        }
    
    def test_model_generation(self, prompt: str = "Напиши коротко о искусственном интеллекте") -> str:
        """Тестирование генерации текста через MCP"""
        if self.use_mcp:
            return self.mcp_client.generate_text(prompt, self.model_name)
        else:
            response = self.ollama_client.generate(model=self.model_name, prompt=prompt)
            return response['response']