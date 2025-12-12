from rag_system import RAGSystem
import sys

def main():
    print("=" * 60)
    print("🎓 ДИПЛОМНЫЙ ПРОЕКТ: RAG-архитектура для долгосрочной памяти")
    print("=" * 60)
    
    try:
        # Создаем систему
        rag = RAGSystem()
        
        # Добавляем начальные знания
        rag.add_initial_knowledge()
        
        print("✅ Система успешно инициализирована!")
        print(f"📊 Модель: tinyllama:1.1b")
        
        print("\n🤖 RAG Система готова! Введите ваш вопрос")
        print("   Команды: 'quit' - выход")
        print("-" * 60)
        
        # Основной цикл взаимодействия
        while True:
            try:
                user_input = input("\n👤 Ваш вопрос: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'выход', 'q']:
                    print("\n👋 До свидания! Система завершает работу.")
                    break
                
                if not user_input:
                    continue
                
                print("🔄 Обработка запроса...")
                response = rag.process_query(user_input)
                print(f"\n🤖 Ответ: {response}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Завершение работы по запросу пользователя.")
                break
            except Exception as e:
                print(f"\n❌ Произошла ошибка: {e}")
                
    except Exception as e:
        print(f"\n❌ Ошибка инициализации системы: {e}")
        print("Проверьте, что Ollama запущен: ollama serve")
        sys.exit(1)

if __name__ == "__main__":
    main()