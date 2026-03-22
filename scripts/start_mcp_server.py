#!/usr/bin/env python3
"""
Скрипт для запуска универсального AI MCP сервера
"""
import subprocess
import sys
import os
from pathlib import Path

def start_ai_mcp_server():
    """Запуск универсального AI MCP сервера"""
    server_path = Path(__file__).parent.parent / "mcp_servers" / "ai_mcp_server.py"
    
    if not server_path.exists():
        print(f"❌ Файл сервера не найден: {server_path}")
        return False
    
    print("🚀 Запуск УНИВЕРСАЛЬНОГО AI MCP Server...")
    print("📍 Сервер будет доступен по адресу: http://localhost:8000")
    print("📚 Объединенные сервисы:")
    print("   ├── Векторная БД (ChromaDB)")
    print("   ├── LLM Модели (Ollama)")
    print("   └── RAG Pipeline")
    print("")
    print("🌐 API документация: http://localhost:8000/docs")
    print("❤️  Проверка здоровья: http://localhost:8000/health")
    print("")
    print("⚡ Для остановки сервера нажмите Ctrl+C")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, str(server_path)
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Остановка сервера...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_ai_mcp_server()