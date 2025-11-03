import chromadb
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

# Настройки (на случай если config не импортируется)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_COLLECTION_NAME = "diplom_rag_memory"
TOP_K_RESULTS = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        logger.info("🔄 Инициализация векторной базы данных...")
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection = self.client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("✅ Векторная БД готова!")
    
    def add_documents(self, documents, metadata_list=None):
        try:
            if metadata_list is None:
                metadata_list = [{}] * len(documents)
            
            embeddings = self.embedder.encode(documents).tolist()
            
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata_list,
                ids=[f"doc_{i}" for i in range(len(documents))]
            )
            logger.info(f"✅ Добавлено {len(documents)} документов")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка при добавлении документов: {e}")
            return False
    
    def search_similar(self, query, top_k=TOP_K_RESULTS):
        try:
            query_embedding = self.embedder.encode([query]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            
            documents = results["documents"][0] if results["documents"] else []
            logger.info(f"🔍 Найдено {len(documents)} релевантных документов")
            return documents
        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            return []
    
    def get_collection_info(self):
        try:
            count = self.collection.count()
            return {"document_count": count}
        except Exception as e:
            logger.error(f"❌ Ошибка получения информации: {e}")
            return {"document_count": 0}