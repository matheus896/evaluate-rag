# file_search_rag.py

import os
import time
import logging
import unicodedata
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Configuração básica de logging para observarmos o processo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar variáveis de ambiente
load_dotenv()

# Configurar o cliente com a API key
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
)

def normalize_filename(filename: str) -> str:
    """
    Normaliza o nome do arquivo removendo caracteres especiais que podem causar problemas no upload.
    Remove acentos e caracteres não-ASCII, mantendo apenas caracteres seguros.
    """
    # Normaliza o unicode para decompor acentos
    nfd = unicodedata.normalize('NFD', filename)
    # Remove marcas diacríticas (acentos)
    ascii_name = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    # Remove caracteres não-ASCII restantes
    safe_name = ascii_name.encode('ascii', 'ignore').decode('ascii')
    # Substitui múltiplos espaços por underscore e limpa caracteres especiais problemáticos
    safe_name = safe_name.replace(' ', '_').replace('+', '_').replace('[', '').replace(']', '')
    return safe_name

def find_store_by_display_name(display_name: str):
    """Busca um FileSearchStore pelo display_name."""
    try:
        for store in client.file_search_stores.list():
            if store.display_name == display_name:
                logging.info(f"FileSearchStore encontrado: {store.name} (display_name: {display_name})")
                return store
        return None
    except Exception as e:
        logging.warning(f"Erro ao listar stores: {e}")
        return None

def create_or_get_store(store_name: str):
    """Cria um novo FileSearchStore se não existir, ou retorna o existente pelo display_name."""
    # Primeiro tenta buscar pelo display_name
    store = find_store_by_display_name(store_name)
    
    if store:
        logging.info(f"FileSearchStore '{store_name}' já existe. Utilizando: {store.name}")
        return store
    
    # Se não encontrou, cria um novo
    logging.info(f"Criando um novo FileSearchStore com o nome: {store_name}")
    store = client.file_search_stores.create(config={'display_name': store_name})
    logging.info(f"FileSearchStore criado: {store.name} (display_name: {store_name})")
    return store

def upload_files_to_store(store_name: str, files_directory: str):
    """Faz o upload de todos os arquivos de um diretório para o FileSearchStore."""
    store = create_or_get_store(store_name)
    
    if not store.name:
        raise ValueError(f"Store '{store_name}' não tem um nome válido.")
    
    for filename in os.listdir(files_directory):
        file_path = os.path.join(files_directory, filename)
        if os.path.isfile(file_path):
            # Normalizar o nome do arquivo para evitar problemas com caracteres especiais
            safe_display_name = normalize_filename(filename)
            
            logging.info(f"Iniciando upload do arquivo: {filename} (como '{safe_display_name}') para o store '{store_name}'...")
            
            # O upload e a importação são feitos em uma única chamada.
            operation = client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=store.name,
                config={'display_name': safe_display_name}
            )
            
            # O processo é assíncrono, então precisamos monitorar a conclusão.
            while not operation.done:
                logging.info(f"Processando arquivo '{filename}'... Aguardando conclusão...")
                time.sleep(5)
                operation = client.operations.get(operation)
            
            # Verificar se houve erro
            if hasattr(operation, 'error') and operation.error:
                logging.error(f"Falha ao importar o arquivo '{filename}'. Erro: {operation.error}")
            else:
                logging.info(f"Arquivo '{filename}' importado com sucesso.")
    
    logging.info("Processo de upload de arquivos concluído.")
    return store

def ask_question(query: str, store_name: str):
    """Envia uma pergunta para o modelo Gemini usando o FileSearchStore como ferramenta."""
    logging.info(f"Executando query: '{query}' no store '{store_name}'")
    
    # Obter o store completo para passar o nome correto
    # Se store_name não contém '/', assume que é apenas o display_name e busca o store
    store = create_or_get_store(store_name)
    
    if not store.name:
        raise ValueError(f"Store '{store_name}' não tem um nome válido.")
    
    # Medida de latência TTFT
    start_time = time.time()
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",  # Usando o modelo mais recente
        contents=query,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store.name]
                    )
                )
            ]
        )
    )
    
    first_token_time = time.time()
    ttft = first_token_time - start_time
    
    response_text = response.text if response.text else ""
    total_time = time.time() - start_time
    
    # Medida de latência TPOT
    output_token_count = len(response_text.split()) if response_text else 1
    tpot = (total_time - ttft) / output_token_count if output_token_count > 0 else 0

    logging.info(f"Resposta recebida. TTFT: {ttft:.2f}s, TPOT: {tpot:.4f}s/token")
    
    # Extraindo os metadados de grounding (os chunks recuperados)
    grounding_metadata = None
    if response.candidates and len(response.candidates) > 0:
        grounding_metadata = response.candidates[0].grounding_metadata
    
    return {
        "answer": response_text,
        "grounding_metadata": grounding_metadata,
        "latency": {"ttft": ttft, "tpot": tpot}
    }

# --- Bloco de Teste ---
if __name__ == "__main__":
    # Usaremos o dataset de direito constitucional para o teste
    DATASET_NAME = "direito_constitucional"
    DOCS_PATH = os.path.join(os.path.dirname(__file__), "docs", DATASET_NAME)
    
    logging.info(f"Iniciando teste de upload para o dataset: {DATASET_NAME}")
    logging.info(f"Diretório dos documentos: {DOCS_PATH}")
    
    # Executa o upload
    # Opcional: Comente a linha abaixo após a primeira execução para não refazer o upload sempre
    upload_files_to_store(store_name=DATASET_NAME, files_directory=DOCS_PATH)
    
    # Teste de Query
    test_query = "Qual a distinção entre eutanásia, ortotanásia e distanásia, conforme apresentado no artigo sobre o tema à luz do direito constitucional?"
    result = ask_question(query=test_query, store_name=DATASET_NAME)
    
    print("\n--- RESULTADO DA BUSCA ---")
    print(f"Pergunta: {test_query}")
    print(f"\nResposta: {result['answer']}")
    print("\n--- CONTEXTO RECUPERADO (Grounding Metadata) ---")
    print(result['grounding_metadata'])
    print("\n--- LATÊNCIA ---")
    print(result['latency'])
