from enhanced_rag_system import EnhancedRAGSystem
import sys

def main():
    print("=" * 60)
    print("ДИПЛОМНЫЙ ПРОЕКТ: RAG-архитектура для долгосрочной памяти")
    print("Версия с динамическим запросом контекста")
    print("=" * 60)
    
    try:
        rag = EnhancedRAGSystem(max_context_rounds=3)
        rag.add_initial_knowledge()
        
        print("\nСистема готова к работе")
        print("Команды: 'quit' - выход, 'clear' - очистить базу данных, 'stats' - статистика")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nВаш вопрос: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'выход', 'q']:
                    print("\nЗавершение работы")
                    break
                
                if user_input.lower() == 'clear':
                    rag.clear_database()
                    continue
                    
                if user_input.lower() == 'stats':
                    info = rag.get_system_info()
                    print(f"\nСтатистика:")
                    print(f"  Документов в БД: {info['documents_in_db']}")
                    print(f"  Модель: {info['model']}")
                    print(f"  Доступные модели: {', '.join(info['available_models'])}")
                    print(f"  История диалогов: {info['dialog_history_length']} записей")
                    continue
                
                if not user_input:
                    continue
                
                print("Обработка запроса...")
                response = rag.process_query(user_input)
                print(f"\nОтвет: {response}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nЗавершение работы")
                break
            except Exception as e:
                print(f"\nОшибка: {e}")
                
    except Exception as e:
        print(f"\nОшибка инициализации системы: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()