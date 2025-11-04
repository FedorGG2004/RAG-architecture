#!/usr/bin/env python3
"""
Скрипт для запуска MCP сервера
"""
import subprocess
import sys
import os
from pathlib import Path

def start_mcp_server():
    """Запуск MCP сервера"""
    server_path = Path(__file__).parent.parent / "mcp_servers" / "vector_mcp_server.py"
    
    if not server_path.exists():
        print(f"❌ Файл сервера не найден: {server_path}")
        return False
    
    print("🚀 Запуск MCP Vector Server...")
    print("📍 Сервер будет доступен по адресу: http://localhost:8000")
    print("📚 API документация: http://localhost:8000/docs")
    print("\nДля остановки сервера нажмите Ctrl+C\n")
    
    try:
        # Запускаем сервер
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
    start_mcp_server()