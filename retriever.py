import chromadb
from sentence_transformers import SentenceTransformer
import sys
import os

# Obter o diretório do script atual
script_dir = os.path.dirname(os.path.abspath(__file__))
chroma_db_path = os.path.join(script_dir, "chroma_db")

class Retriever:
    def __init__(self, db_path=chroma_db_path, collection_name=""):
        """
        Inicializa o sistema de query RAG.
        
        Args:
            db_path (str): Caminho para o banco ChromaDB
            collection_name (str): Nome da coleção no ChromaDB
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.modelo = None
        
        self._initialize()
    
    def _initialize(self):
        """Inicializa o cliente ChromaDB e carrega o modelo."""
        try:
            # Conectar ao ChromaDB
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_collection(name=self.collection_name)
            
            # Carregar modelo de embeddings
            print("Carregando modelo de embeddings...")
            self.modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            print(f"✅ Conectado à coleção '{self.collection_name}'")
            print(f"📊 Total de documentos: {self.collection.count()}")
            
        except Exception as e:
            print(f"❌ Erro ao inicializar: {e}")
            print("Certifique-se de que o banco ChromaDB foi criado executando rag_classic.py primeiro.")
            sys.exit(1)
    
    def search(self, query_text, n_results=5, show_metadata=False):
        """
        Busca documentos similares à query.
        
        Args:
            query_text (str): Texto da consulta
            n_results (int): Número de resultados a retornar
            show_metadata (bool): Se deve mostrar metadados dos resultados
            
        Returns:
            dict: Resultados da busca
        """
        try:
            # Gerar embedding da query
            query_embedding = self.modelo.encode([query_text])
            
            # Buscar no ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['documents', 'distances', 'metadatas']
            )
            
            
            res = results['documents'][0]
            
            return res
            
        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return None
    


