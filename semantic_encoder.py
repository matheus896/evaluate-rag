# arquivo semantic_encoder.py

import os
from chunks import Chunks
from read_files import ReadFiles
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from typing import Any, Dict, List, Optional, cast


class SemanticEncoder:
    """
    Constrói a base vetorial e popula o ChromaDB a partir de documentos em um diretório.

    Parâmetros:
    - docs_dir (str): diretório onde estão os documentos
    - chunk_size (int): tamanho de cada chunk
    - overlap_size (int): tamanho da sobreposição entre chunks
    - db_path (str): caminho do banco ChromaDB (default: "./chroma_db")
    - collection_name (str): nome da coleção no ChromaDB (default: "documentos_rag")
    """

    def __init__(
        self,
        docs_dir: str,
        chunk_size: int,
        overlap_size: int,
        db_path: str = "./chroma_db",
        collection_name: str = "documentos_rag",
    ) -> None:
        self.docs_dir = docs_dir
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.db_path = db_path
        self.collection_name = collection_name

        # Dependências
        self.rf = ReadFiles()
        self.chunker = Chunks(chunk_size=self.chunk_size, overlap_size=self.overlap_size)
        self.modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = None

    def build(self, reset_collection: bool = True, collection_name: Optional[str] = None) -> dict:
        """
        Lê os documentos do diretório, cria chunks, gera embeddings e salva no ChromaDB.

        Args:
            reset_collection (bool): Se verdadeiro, apaga a coleção antes de recriá-la.
        Returns:
            dict: Estatísticas do processo (número de chunks salvos e total de documentos na coleção).
        """
        collection_name = collection_name or self.collection_name
        self.collection_name = collection_name

        # 1) Ler documentos e consolidar em markdown
        mds = self.rf.docs_to_markdown(self.docs_dir)

        # 2) Criar chunks
        text_chunks = self.chunker.create_chunks(mds)

        if not text_chunks:
            print("⚠️ Nenhum chunk foi gerado. Verifique o conteúdo em 'docs'.")
            try:
                existing_collection = self.client.get_collection(name=collection_name)
                total_docs = existing_collection.count()
            except Exception:
                total_docs = 0
            return {
                "chunks_salvos": 0,
                "colecao": collection_name,
                "total_documentos": total_docs,
            }

        # 3) Gerar embeddings
        base_vetorial_documentos = self.modelo.encode(text_chunks)
        embeddings = base_vetorial_documentos.tolist()  # ChromaDB requer lista

        # 4) (Re)criar/obter coleção
        if reset_collection:
            try:
                self.client.delete_collection(name=collection_name)
                print(f"Coleção '{collection_name}' existente foi deletada.")
            except Exception:
                pass

        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Coleção de chunks de documentos com embeddings"},
            )

        # 5) Inserir dados
        ids = [str(uuid.uuid4()) for _ in range(len(text_chunks))]
        metadatas: List[Dict[str, Any]] = [
            {
                "chunk_id": i,
                "chunk_size": len(chunk),
                "source": self.docs_dir,
            }
            for i, chunk in enumerate(text_chunks)
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=text_chunks,
            metadatas=cast(Any, metadatas),
        )

        print(f"✅ Salvos {len(text_chunks)} chunks no ChromaDB!")
        print(
            f"📊 Coleção '{self.collection_name}' agora possui {self.collection.count()} documentos"
        )

        return {
            "chunks_salvos": len(text_chunks),
            "colecao": self.collection_name,
            "total_documentos": self.collection.count(),
        }



if __name__ == "__main__":
    import os
    import traceback

    # Obter o diretório do script atual (garante que os caminhos relativos funcionem independente do cwd)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    docs_base = os.path.join(script_dir, "docs")
    chroma_db_path = os.path.join(script_dir, "chroma_db")

    # Lista de datasets para processar (nome da coleção -> subpasta em docs)
    datasets = [
        {"name": "synthetic_dataset_papers", "subdir": "synthetic_dataset_papers"},
        {"name": "direito_constitucional", "subdir": "direito_constitucional"},
    ]

    results = {}

    for ds in datasets:
        docs_path = os.path.join(docs_base, ds["subdir"])

        if not os.path.exists(docs_path):
            print(f"⚠️ Diretório de documentos não encontrado para '{ds['name']}': {docs_path}. Pulando.")
            results[ds["name"]] = {"error": "docs_not_found", "path": docs_path}
            continue

        try:
            print(f"\n🔁 Construindo coleção '{ds['name']}' a partir de: {docs_path}")
            retriever = SemanticEncoder(
                docs_dir=docs_path,
                chunk_size=2000,
                overlap_size=500,
                db_path=chroma_db_path,
                collection_name=ds["name"],
            )

            stats = retriever.build(collection_name=ds["name"])
            results[ds["name"]] = stats

        except Exception as e:
            print(f"❌ Erro ao construir coleção '{ds['name']}': {e}")
            traceback.print_exc()
            results[ds["name"]] = {"error": str(e)}

    # Resumo final
    print("\n📌 Resumo das operações:")
    for name, value in results.items():
        print(f"- {name}: {value}")