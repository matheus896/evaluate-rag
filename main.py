from semantic_encoder import SemanticEncoder
import os

# Obter o diretório do script atual
script_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(script_dir, "docs", "synthetic_dataset_papers")
chroma_db_path = os.path.join(script_dir, "chroma_db")

encoder = SemanticEncoder(
    docs_dir=docs_path, #diretório dos documentos
    chunk_size=5000, #tamanho do chunk
    overlap_size=500, #tamanho da sobreposição
    db_path=chroma_db_path, #caminho do banco ChromaDB
)

# Construir base vetorial
stats = encoder.build(collection_name="synthetic_dataset_papers")

# Imprimir estatísticas
print(stats)



