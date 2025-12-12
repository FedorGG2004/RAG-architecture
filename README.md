# RAG System for Diplom

Простая RAG система для дипломного проекта.

## Установка
1. pip install -r requirements.txt
2. ollama serve
3. python src/main.py

загрузка на гитхаб в буферную ветку изменений локальных:

cd C:\Users\Fedos\Desktop\RAG-architecture-main

git init

git remote add origin https://github.com/FedorGG2004/RAG-architecture.git

git pull origin main

git checkout -b buffer

git add .

git commit -m "Обновление: улучшен промпт, исправлена архитектура, добавлена модель llama3.2:3b"

git push -u origin buffer


подгрузка изменений на сервере:
#!/bin/bash
cd ~/fedor/RAG-architecture

echo "💾 Сохраняю текущие изменения..."
git stash

echo "📥 Получаю ветку buffer с GitHub..."
git fetch origin buffer

echo "🔀 Переключаюсь на ветку buffer..."
git checkout -b buffer --track origin/buffer

echo "↩️ Возвращаю изменения..."
git stash pop

echo "🚫 Добавляю .gitignore для игнорирования служебных файлов..."
cat > .gitignore << 'EOF'
.venv/
data/chroma_db/
__pycache__/
*.pyc
chroma_db/
.DS_Store
EOF

echo "✅ Готово! Код обновлен из ветки buffer"
