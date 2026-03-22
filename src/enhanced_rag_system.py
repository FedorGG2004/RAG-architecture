import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from config import *
from mcp_client import MCPClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedRAGSystem:
    def __init__(self, model_name=MODEL_NAME, use_mcp=True, max_context_rounds=3):
        self.model_name = model_name
        self.dialog_history = []
        self.use_mcp = use_mcp
        self.max_context_rounds = max_context_rounds
        
        # Регулярные выражения для парсинга тегов
        # В __init__ замените:
        self.need_context_pattern = re.compile(r'<NEED_CONTEXT>(.*?)(?:</NEED_CONTEXT>|$)', re.DOTALL)
        self.answer_pattern = re.compile(r'<ANSWER>(.*?)(?:</ANSWER>|$)', re.DOTALL)
        
        if self.use_mcp:
            try:
                self.mcp_client = MCPClient()
                if self.mcp_client.is_server_running():
                    server_info = self.mcp_client.get_server_info()
                    services = server_info.get("services", {})
                    
                    logger.info("RAG система инициализирована с MCP клиентом")
                    logger.info(f"Статус сервисов: БД({services.get('vector_db', 'unknown')})")
                    
                    available_models = self.mcp_client.list_models()
                    if available_models:
                        logger.info(f"Доступные модели: {', '.join(available_models)}")
                else:
                    logger.error("MCP сервер недоступен")
                    self.use_mcp = False
            except Exception as e:
                logger.error(f"Ошибка инициализации MCP клиента: {e}")
                self.use_mcp = False
        else:
            from vector_db import VectorStore
            import ollama
            self.vector_db = VectorStore()
            self.ollama_client = ollama.Client()
            logger.info("RAG система с прямыми вызовами")
    
    def clear_database(self):
        """Очистка базы данных"""
        try:
            print("\nОчистка базы данных...")
            
            if self.use_mcp:
                # Получаем информацию до очистки
                info = self.mcp_client.get_collection_info()
                before_count = info.get('document_count', 0)
                
                if before_count == 0:
                    print("База данных уже пуста")
                    return
                
                # Для очистки через MCP сервер используем прямой запрос
                try:
                    # Отправляем POST запрос на эндпоинт /clear
                    response = self.mcp_client.session.post(
                        f"{self.mcp_client.server_url}/clear",
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        deleted = result.get('deleted_count', 0)
                        print(f" База данных очищена. Удалено документов: {deleted}")
                    else:
                        print(f" Ошибка очистки: {response.status_code}")
                        print("Попробуйте удалить папку data/chroma_db вручную")
                        
                except Exception as e:
                    print(f" Ошибка при очистке: {e}")
                    print("Эндпоинт /clear не доступен. Добавьте его в ai_mcp_server.py")
            else:
                # Очистка при прямом доступе (без MCP)
                import shutil
                from config import VECTOR_DB_DIR
                
                # Получаем информацию до очистки
                info = self.vector_db.get_collection_info()
                before_count = info.get('document_count', 0)
                
                if before_count == 0:
                    print("База данных уже пуста")
                    return
                
                # Закрываем текущее соединение
                del self.vector_db
                
                # Удаляем папку с данными
                shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
                print(f"Папка {VECTOR_DB_DIR} удалена")
                
                # Создаём заново
                from vector_db import VectorStore
                self.vector_db = VectorStore()
                
                print(f" База данных очищена. Удалено документов: {before_count}")
            
            # Перезагружаем начальные знания
            print("Перезагрузка начальных знаний...")
            self.add_initial_knowledge()
            
        except Exception as e:
            print(f" Ошибка при очистке базы данных: {e}")
            import traceback
            traceback.print_exc()
    
    def add_initial_knowledge(self):
        """Добавление начальных знаний"""
        initial_knowledge = [
            "Машинное обучение - это раздел искусственного интеллекта, который позволяет компьютерам обучаться на данных.",
            "Python является популярным языком программирования для анализа данных и машинного обучения.",
            "RAG (Retrieval-Augmented Generation) - это архитектура, которая сочетает поиск информации и генерацию текста.",
            "Оллaма - это платформа для запуска больших языковых моделей локально на компьютере.",
            "Векторная база данных хранит информацию в виде числовых векторов для семантического поиска."
        ]
        
        print("Загрузка начальной базы знаний...")
        
        success_count = 0
        for knowledge in initial_knowledge:
            if self.use_mcp:
                success = self.mcp_client.add_document(
                    knowledge, 
                    {"source": "base_knowledge", "type": "fact"}
                )
                if success:
                    success_count += 1
                    print(f"Добавлено: {knowledge[:50]}...")
                else:
                    print(f"Ошибка добавления: {knowledge[:50]}...")
            else:
                self.vector_db.add_documents(
                    [knowledge], 
                    [{"source": "base_knowledge", "type": "fact"}]
                )
                success_count += 1
        
        print(f"Всего добавлено документов: {success_count}/{len(initial_knowledge)}")
        
        if self.use_mcp:
            info = self.mcp_client.get_collection_info()
            print(f"Всего документов в базе: {info.get('document_count', 0)}")
    
    def build_initial_prompt(self, user_query: str) -> str:
        """Формирует начальный промпт с инструкцией по использованию тегов"""
        prompt = f"""Ты - ассистент. Отвечай на вопросы пользователя.

ПРАВИЛА:
1. Если тебе не хватает информации для ответа, используй тег <NEED_CONTEXT> с описанием того, что нужно найти.
2. После получения достаточной информации дай ответ в теге <ANSWER>.
3. Если информации в контексте недостаточно, ты можешь запросить дополнительный контекст снова.
4. Отвечай на том языке, на котором задан вопрос.

ФОРМАТ ЗАПРОСА КОНТЕКСТА:
<NEED_CONTEXT>
  "query": "текст запроса для поиска в базе знаний",
  "reason": "почему тебе нужна эта информация"
</NEED_CONTEXT>

ФОРМАТ ОТВЕТА:
<ANSWER>
Текст ответа пользователю
</ANSWER>

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}

Твой ответ (используй <NEED_CONTEXT> или <ANSWER>):
"""
        return prompt
    
    def build_context_prompt(self, original_query: str, context: str, round_num: int) -> str:
        """Формирует промпт с контекстом для модели"""
        prompt = f"""Ты - ассистент. Вот информация из базы знаний, которую ты запросил:

КОНТЕКСТ:
{context}

Исходный вопрос пользователя: {original_query}
Текущий раунд диалога: {round_num}

ПРАВИЛА:
1. Если информации в контексте достаточно, дай ответ в теге <ANSWER>.
2. Если информации НЕДОСТАТОЧНО, ты можешь запросить дополнительный контекст в теге <NEED_CONTEXT>.
3. Отвечай на том языке, на котором задан вопрос.

Твой ответ (используй <ANSWER> если информации достаточно, или <NEED_CONTEXT> если нужно больше информации):
"""
        return prompt
    
    def build_force_answer_prompt(self, original_query: str, context: str) -> str:
        """Принуждение к ответу при зацикливании"""
        prompt = f"""Ты уже несколько раз запрашивал контекст. Используй имеющуюся информацию для ответа.

КОНТЕКСТ:
{context}

ВОПРОС: {original_query}

Дай ответ в теге <ANSWER> (только сам ответ, без пояснений):
"""
        return prompt
    
    def search_context(self, query_text: str) -> str:
        """Поиск контекста в базе знаний"""
        try:
            # Извлекаем сам запрос из JSON, если он есть
            try:
                request_data = json.loads(query_text)
                search_query = request_data.get("query", query_text)
            except:
                search_query = query_text
            
            if self.use_mcp:
                documents = self.mcp_client.search_documents(search_query, top_k=3)
            else:
                documents = self.vector_db.search_similar(search_query, top_k=3)
            
            if documents:
                context = "\n---\n".join(documents)
                logger.info(f"Найден контекст ({len(documents)} документов)")
                return context
            else:
                return "Информация по данному запросу отсутствует в базе знаний."
        except Exception as e:
            logger.error(f"Ошибка при поиске контекста: {e}")
            return "Ошибка при поиске информации в базе знаний."
    
    def process_query(self, user_query: str) -> str:
        logger.info(f"Обработка запроса: {user_query}")
        
        current_prompt = self.build_initial_prompt(user_query)
        round_num = 0
        context_history = []
        context_requests = []
        
        while round_num < self.max_context_rounds:
            round_num += 1
            logger.info(f"Раунд {round_num}/{self.max_context_rounds}")
            
            # Получаем ответ от модели
            if self.use_mcp:
                response = self.mcp_client.generate_text(
                    prompt=current_prompt,
                    model=self.model_name
                )
            else:
                response = self.ollama_client.generate(
                    model=self.model_name,
                    prompt=current_prompt
                )['response']
            
            logger.info(f"Сырой ответ модели (раунд {round_num}): {response[:200]}...")
            
            # Проверяем запрос контекста
            need_context_match = self.need_context_pattern.search(response)
            if need_context_match:
                context_request = need_context_match.group(1).strip()
                logger.info(f"РАСПОЗНАН ЗАПРОС КОНТЕКСТА: {context_request[:100]}...")
                
                # Проверка на пустой запрос
                if not context_request or len(context_request) < 5:
                    logger.warning("Модель запросила пустой контекст")
                    if context_history:
                        # Уже есть контекст - просим ответить
                        current_prompt = f"""У тебя уже есть контекст. Используй его для ответа.

    КОНТЕКСТ:
    {context_history[-1]}

    Вопрос: {user_query}

    Ответь в теге <ANSWER>:
    """
                    else:
                        # Нет контекста - повторяем начальный запрос
                        current_prompt = self.build_initial_prompt(user_query)
                    continue
                
                # Защита от повторных запросов с одинаковым содержанием
                if context_request in context_requests:
                    logger.warning(f"Повторный запрос контекста: {context_request[:50]}...")
                    if context_history:
                        # Даём тот же контекст, но просим ответить
                        current_prompt = f"""Ты уже получал этот контекст. Теперь ответь.

    КОНТЕКСТ:
    {context_history[-1]}

    Вопрос: {user_query}

    Ответ в теге <ANSWER>:
    """
                    continue
                
                # Новый запрос контекста
                context_requests.append(context_request)
                
                # Ищем контекст
                context = self.search_context(context_request)
                context_history.append(context)
                
                # Формируем промпт с контекстом
                current_prompt = self.build_context_prompt(
                    user_query, 
                    context,
                    round_num
                )
                continue
            
            # Проверяем ответ в теге ANSWER
            answer_match = self.answer_pattern.search(response)
            if answer_match:
                final_answer = answer_match.group(1).strip()
                logger.info(f"РАСПОЗНАН ОТВЕТ В ТЕГЕ на раунде {round_num}")
                
                self.dialog_history.extend([f"User: {user_query}", f"Assistant: {final_answer}"])
                if self.should_save_to_memory(user_query, final_answer):
                    self.save_to_memory(user_query, final_answer, context_history)
                
                return final_answer
            
            # Если ответ без тегов, но осмысленный
            if len(response) > 20 and not any(phrase in response.lower() for phrase in ["<need_context", "запросил контекст"]):
                logger.info(f"РАСПОЗНАН ОТВЕТ БЕЗ ТЕГОВ на раунде {round_num}")
                
                clean_response = re.sub(r'<[^>]+>', '', response).strip()
                
                self.dialog_history.extend([f"User: {user_query}", f"Assistant: {clean_response}"])
                if self.should_save_to_memory(user_query, clean_response):
                    self.save_to_memory(user_query, clean_response, context_history)
                
                return clean_response
            
            logger.warning(f"НЕРАСПОЗНАННЫЙ ФОРМАТ: {response[:100]}...")
        
        error_msg = "Не удалось получить ответ за допустимое число запросов контекста"
        logger.error(error_msg)
        return error_msg

    def save_to_memory(self, query: str, response: str, context_history: List[str]):
        """Сохранение информации в базу знаний"""
        try:
            if not self.use_mcp:
                from vector_db import VectorStore
                vector_db = VectorStore()
                vector_db.add_documents(
                    [f"Вопрос: {query}", f"Ответ: {response}"],
                    [
                        {"type": "dialog", "source": "user", "timestamp": datetime.now().isoformat()},
                        {"type": "dialog", "source": "assistant", "timestamp": datetime.now().isoformat()}
                    ]
                )
                logger.info("Информация сохранена в память (прямой доступ)")
                return
            
            # Сохраняем вопрос
            self.mcp_client.add_document(
                f"Вопрос: {query}",
                {
                    "type": "dialog",
                    "timestamp": datetime.now().isoformat(),
                    "source": "user"
                }
            )
            
            # Сохраняем ответ
            self.mcp_client.add_document(
                f"Ответ: {response}",
                {
                    "type": "dialog",
                    "timestamp": datetime.now().isoformat(),
                    "source": "assistant"
                }
            )
            
            # Сохраняем использованный контекст (только если он есть и не содержит ошибок)
            if context_history and len(context_history) > 0:
                last_context = context_history[-1]
                if last_context and len(last_context) > 50 and "отсутствует" not in last_context.lower():
                    self.mcp_client.add_document(
                        f"Контекст для вопроса '{query[:50]}...': {last_context[:200]}",
                        {
                            "type": "context",
                            "timestamp": datetime.now().isoformat(),
                            "source": "retrieved"
                        }
                    )
                    
            logger.info("Информация сохранена в память")
        except Exception as e:
            logger.error(f"Ошибка сохранения в память: {e}")
    
    def get_system_info(self) -> dict:
        """Получение информации о системе"""
        if self.use_mcp:
            db_info = self.mcp_client.get_collection_info()
            models = self.mcp_client.list_models()
            doc_count = db_info.get("document_count", 0)
        else:
            from vector_db import VectorStore
            vector_db = VectorStore()
            doc_count = vector_db.get_collection_info().get("document_count", 0)
            models = [self.model_name]
            
        return {
            "model": self.model_name,
            "dialog_history_length": len(self.dialog_history),
            "documents_in_db": doc_count,
            "available_models": models,
            "max_context_rounds": self.max_context_rounds,
            "using_mcp": self.use_mcp
        }
    def should_save_to_memory(self, query: str, response: str) -> bool:
        """Определяет, стоит ли сохранять ответ в память"""
        if not response or len(response) < 10:
            return False
        
        forbidden_phrases = [
            "не знаю", "неизвестно", "не нашел информации", 
            "отсутствует в базе знаний", "не удалось получить ответ"
        ]
        
        response_lower = response.lower()
        if any(phrase in response_lower for phrase in forbidden_phrases):
            return False
            
        if any(word ыin query.lower() for word in ["привет", "здравствуй", "hello", "пока"]):
            return False
            
        return True