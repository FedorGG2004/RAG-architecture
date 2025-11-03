import asyncio
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import sys

class VectorStoreMCP:
    """Упрощенный MCP-сервер для работы с векторной БД"""
    
    def __init__(self):
        # Определяем путь к базе данных
        base_dir = Path(__file__).parent.parent
        db_path = base_dir / "data" / "chroma_db"
        
        # Инициализируем ChromaDB
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection("rag_memory")
        
        # Загружаем модель для эмбеддингов
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        print("✅ MCP Vector Store инициализирован")

    def search_documents(self, query: str, top_k: int = 3):
        """Поиск документов в векторной БД"""
        try:
            print(f"🔍 MCP: Поиск документов для запроса '{query}'")
            
            # Преобразуем запрос в вектор
            query_embedding = self.embedder.encode([query]).tolist()
            
            # Ищем в векторной БД
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            documents = results["documents"][0] if results["documents"] else []
            print(f"✅ MCP: Найдено {len(documents)} документов")
            
            return {
                "documents": documents,
                "count": len(documents),
                "query": query
            }
            
        except Exception as e:
            print(f"❌ MCP Ошибка поиска: {e}")
            return {
                "documents": [],
                "count": 0,
                "error": str(e)
            }

    def add_document(self, text: str, metadata: dict = None):
        """Добавление документа в векторную БД"""
        try:
            if metadata is None:
                metadata = {"source": "manual", "type": "fact"}
            
            print(f"💾 MCP: Добавление документа: {text[:50]}...")
            
            # Преобразуем текст в вектор
            embedding = self.embedder.encode([text]).tolist()
            
            # Создаем ID документа
            doc_id = f"doc_{hash(text) % 1000000}"
            
            # Добавляем в базу данных (исправленная версия)
            self.collection.add(
                embeddings=embedding,
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return {
                "success": True,
                "message": "Документ успешно добавлен",
                "text_length": len(text),
                "doc_id": doc_id
            }
            
        except Exception as e:
            print(f"❌ MCP Ошибка добавления: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def add_initial_knowledge(self):
        """Добавление начальных знаний в базу"""
        initial_knowledge = [
            "Машинное обучение - это раздел искусственного интеллекта, который позволяет компьютерам обучаться на данных.",
            "Python является популярным языком программирования для анализа данных и машинного обучения.",
            "RAG (Retrieval-Augmented Generation) - это архитектура, которая сочетает поиск информации и генерацию текста.",
            "Оллaма - это платформа для запуска больших языковых моделей локально на компьютере.",
            "Векторная база данных хранит информацию в виде числовых векторов для семантического поиска."
        ]
        
        print("📚 Добавление начальных знаний в базу...")
        
        for i, knowledge in enumerate(initial_knowledge):
            try:
                embedding = self.embedder.encode([knowledge]).tolist()
                
                self.collection.add(
                    embeddings=embedding,
                    documents=[knowledge],
                    metadatas=[{"source": "initial", "type": "fact", "index": i}],
                    ids=[f"initial_{i}"]
                )
                print(f"✅ Добавлено: {knowledge[:50]}...")
            except Exception as e:
                print(f"❌ Ошибка добавления начальных знаний: {e}")

    def get_collection_info(self):
        """Получение информации о коллекции"""
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": "rag_memory"
            }
        except Exception as e:
            return {
                "document_count": 0,
                "error": str(e)
            }

def main():
    """Простой MCP-сервер через HTTP"""
    print("🚀 Запуск упрощенного MCP-сервера для векторной БД")
    print("📚 Доступные инструменты:")
    print("   - search 'ваш запрос'")
    print("   - add 'ваш текст'")
    print("   - info")
    print("   - exit")
    print("   - init (добавить начальные знания)")
    print("⏳ Сервер готов к работе...")
    
    # Инициализируем хранилище
    vector_store = VectorStoreMCP()
    
    # Простой интерактивный режим для тестирования
    try:
        while True:
            print("\n" + "="*50)
            print("Тестовые команды:")
            print("1. search 'ваш запрос'")
            print("2. add 'ваш текст'")
            print("3. info")
            print("4. init (добавить начальные знания)")
            print("5. exit")
            
            command = input("\nВведите команду: ").strip()
            
            if command.startswith("search "):
                query = command[7:]  # Убираем "search "
                result = vector_store.search_documents(query)
                print("📄 Результат поиска:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                
            elif command.startswith("add "):
                text = command[4:]  # Убираем "add "
                result = vector_store.add_document(text, {"source": "manual", "type": "fact"})
                print("💾 Результат добавления:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                
            elif command == "info":
                result = vector_store.get_collection_info()
                print("📊 Информация о коллекции:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                
            elif command == "init":
                print("📚 Добавление начальных знаний...")
                vector_store.add_initial_knowledge()
                print("✅ Начальные знания добавлены!")
                
            elif command == "exit":
                print("👋 Завершение работы сервера")
                break
                
            else:
                print("❌ Неизвестная команда")
                
    except KeyboardInterrupt:
        print("\n👋 Сервер остановлен пользователем")

if __name__ == "__main__":
    main()