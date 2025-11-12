# evaluate_light.py
"""
Script de avaliação SIMPLIFICADO para testar uma pergunta por vez.
Ideal para API Keys gratuitas com limitações de rate limit.
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Importar os sistemas RAG
from file_search_rag import ask_question as file_search_ask
from retriever import Retriever
from augmentation import Augmentation
from generation import Generation

# Importar módulos de refatoração
from report_generator import EvaluationReportGenerator
from console_presenter import ConsolePresenter

load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class LightRAGEvaluator:
    """Avaliador leve para testar uma pergunta por vez."""
    
    def __init__(self, config_path: str = "test_config.json"):
        """Inicializa o avaliador."""
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        
        # Inicializar módulos de apresentação e relatórios
        self.presenter = ConsolePresenter()
        self.report_generator = EvaluationReportGenerator()
        
        
        encodings_to_try = ['utf-8']
        config_loaded = False
        
        for encoding in encodings_to_try:
            try:
                with open(config_path, 'r', encoding=encoding) as f:
                    self.config = json.load(f)
                    config_loaded = True
                    logging.info(f"✅ Config carregado com encoding: {encoding}")
                    break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        
        if not config_loaded:
            raise ValueError(f"❌ Não foi possível carregar {config_path}")
        
        # Inicializar cliente AI Judge
        self.judge_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Inicializar RAG Manual
        self.retriever = Retriever(collection_name=self.config['dataset'])
        self.augmentation = Augmentation()
        self.generation = Generation(model="gemini-2.5-flash")
        
        logging.info(f"✅ Avaliador inicializado com {len(self.config['questions'])} perguntas disponíveis")
    
    def list_questions(self):
        """Lista todas as perguntas disponíveis."""
        print("\n" + "="*80)
        print("📝 PERGUNTAS DISPONÍVEIS PARA TESTE")
        print("="*80 + "\n")
        
        for q in self.config['questions']:
            print(f"[{q['id']}] Categoria: {q['category']}")
            print(f"    Pergunta: {q['question']}")
            print()
    
    def run_manual_rag(self, query: str) -> Dict[str, Any]:
        """Executa o RAG manual."""
        logging.info("🔧 Executando RAG Manual...")
        start_time = time.time()
        
        chunks = self.retriever.search(query, n_results=5, show_metadata=False)
        prompt = self.augmentation.generate_prompt(query, chunks)
        response = self.generation.generate(prompt)
        
        total_time = time.time() - start_time
        chunks_list = chunks if chunks else []
        
        return {
            "answer": response,
            "context": chunks_list,
            "latency": total_time,
            "num_chunks": len(chunks_list)
        }
    
    def run_file_search_rag(self, query: str) -> Dict[str, Any]:
        """Executa o File Search RAG."""
        logging.info("🔍 Executando File Search RAG...")
        result = file_search_ask(query, self.config['dataset'])
        
        chunks_text = []
        grounding_info = None
        
        if result['grounding_metadata'] and result['grounding_metadata'].grounding_chunks:
            for chunk in result['grounding_metadata'].grounding_chunks:
                if chunk.retrieved_context:
                    chunks_text.append(chunk.retrieved_context.text)
            
            # Extrair informações serializáveis do grounding_metadata
            grounding_info = {
                "num_chunks": len(result['grounding_metadata'].grounding_chunks),
                "has_grounding": True
            }
        
        return {
            "answer": result['answer'],
            "context": chunks_text,
            "latency": result['latency']['ttft'],
            "num_chunks": len(chunks_text),
            "grounding_info": grounding_info  
        }
    
    def create_judge_prompt(self, question: str, context: str, answer: str) -> str:
        """Cria prompt para AI Judge."""
        return f"""Você é um avaliador especialista em sistemas RAG no domínio jurídico.

Avalie a qualidade da resposta segundo os critérios abaixo, usando escala de 1 a 5:

**PERGUNTA:**
{question}

**CONTEXTO RECUPERADO (5 CHUNKS):**
{context}

**RESPOSTA GERADA:**
{answer}

**CRITÉRIOS (1-5):**
1. **Consistência Factual:** Resposta baseada no contexto? Há alucinações?
2. **Seguir Instruções:** Responde o que foi pedido?
3. **Conhecimento Jurídico:** Usa terminologia correta?
4. **Precisão do Contexto:** Contexto é relevante?
5. **Cobertura do Contexto:** Contexto tem todas as informações necessárias?

**RESPONDA EM JSON:**
{{
  "factual_consistency": {{"score": <1-5>, "justification": "..."}},
  "instruction_following": {{"score": <1-5>, "justification": "..."}},
  "domain_knowledge": {{"score": <1-5>, "justification": "..."}},
  "context_precision": {{"score": <1-5>, "justification": "..."}},
  "context_recall": {{"score": <1-5>, "justification": "..."}},
  "overall_assessment": "resumo geral em 1-2 frases"
}}"""
    
    def judge_response(self, question: str, context: str, answer: str, system_name: str) -> Dict[str, Any]:
        """Avalia resposta com AI Judge (com retry em caso de erro 503)."""
        logging.info(f"⚖️  Avaliando {system_name} com AI Judge...")
        
        prompt = self.create_judge_prompt(question, context, answer)
        
        max_retries = 3
        retry_delay = 5  # segundos
        
        for attempt in range(max_retries):
            try:
                response = self.judge_client.models.generate_content(
                    model=self.config['ai_judge_model'],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.config['ai_judge_temperature']
                    )
                )
                
                response_text = (response.text or "").strip()
                
                # Limpar markdown
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                evaluation = json.loads(response_text)
                logging.info(f"✅ Avaliação de {system_name} concluída")
                return evaluation
            
            except Exception as e:
                error_msg = str(e)
                
                if '503' in error_msg or 'overloaded' in error_msg.lower():
                    if attempt < max_retries - 1:
                        logging.warning(f"⚠️  Modelo sobrecarregado. Tentativa {attempt + 1}/{max_retries}. Aguardando {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2 
                        continue
                    else:
                        logging.error(f"❌ Modelo sobrecarregado após {max_retries} tentativas")
                else:
                    logging.error(f"❌ Erro ao avaliar: {error_msg}")
        
        # Se chegou aqui, todas as tentativas falharam
        return {
            "error": "Falha após todas as tentativas",
            "factual_consistency": {"score": 0, "justification": "Erro na avaliação"},
            "instruction_following": {"score": 0, "justification": "Erro na avaliação"},
            "domain_knowledge": {"score": 0, "justification": "Erro na avaliação"},
            "context_precision": {"score": 0, "justification": "Erro na avaliação"},
            "context_recall": {"score": 0, "justification": "Erro na avaliação"}
        }
    
    def evaluate_single_question(self, question_id: int, skip_judge: bool = False):
        """Avalia uma única pergunta."""
        
        # Encontrar a pergunta
        question_data = next((q for q in self.config['questions'] if q['id'] == question_id), None)
        
        if not question_data:
            print(f"❌ Pergunta {question_id} não encontrada!")
            return None
        
        question = question_data['question']
        
        # Exibir cabeçalho
        self.presenter.print_question_header(question_id, question_data['category'], question)
        
        # ============== RAG MANUAL ==============
        self.presenter.print_rag_header("RAG Manual")
        try:
            manual_result = self.run_manual_rag(question)
            self.presenter.print_rag_result(manual_result, "RAG Manual")
            
            # Análise dos Chunks do RAG Manual
            self.presenter.print_chunk_analysis(manual_result['context'], "RAG Manual")
            
            if not skip_judge:
                manual_context_str = "\n\n--- CHUNK SEPARATOR ---\n\n".join(manual_result['context'][:5])
                self.presenter.print_judge_info(manual_context_str)
                
                manual_evaluation = self.judge_response(
                    question, 
                    manual_context_str, 
                    manual_result['answer'],
                    "RAG Manual"
                )
                manual_result['evaluation'] = manual_evaluation
                
                self.presenter.print_judge_evaluation(manual_evaluation, "RAG Manual")
        
        except Exception as e:
            self.presenter.print_error(str(e), "RAG Manual")
            manual_result = None
        
        # Delay entre chamadas para evitar rate limit
        self.presenter.print_wait_message(3, "antes de executar File Search RAG")
        time.sleep(3)
        
        # ============== FILE SEARCH RAG ==============
        self.presenter.print_rag_header("File Search RAG")
        try:
            file_search_result = self.run_file_search_rag(question)
            self.presenter.print_rag_result(file_search_result, "File Search RAG")
            
            # Análise dos Chunks do File Search RAG
            self.presenter.print_chunk_analysis(file_search_result['context'], "File Search RAG")
            
            if not skip_judge:
                # Delay antes do AI Judge
                self.presenter.print_wait_message(5, "antes de chamar AI Judge")
                time.sleep(5)
                
                file_search_context_str = "\n\n--- CHUNK SEPARATOR ---\n\n".join(file_search_result['context'][:5])
                self.presenter.print_judge_info(file_search_context_str)
                
                file_search_evaluation = self.judge_response(
                    question,
                    file_search_context_str,
                    file_search_result['answer'],
                    "File Search RAG"
                )
                file_search_result['evaluation'] = file_search_evaluation
                
                self.presenter.print_judge_evaluation(file_search_evaluation, "File Search RAG")
        
        except Exception as e:
            self.presenter.print_error(str(e), "File Search RAG")
            file_search_result = None
        
        # ============== COMPARAÇÃO ==============
        if manual_result and file_search_result and not skip_judge:
            self.presenter.print_comparison_header()
            self.presenter.print_score_comparison(manual_result, file_search_result)
            self.presenter.print_chunk_comparison(manual_result, file_search_result)
        
        # Salvar resultado
        result = {
            "question_id": question_id,
            "question": question,
            "category": question_data['category'],
            "timestamp": datetime.now().isoformat(),
            "manual_rag": manual_result,
            "file_search_rag": file_search_result
        }
        
        self.report_generator.save_result(result)
        
        self.presenter.print_completion_message()
        return result


def main():
    """Interface de linha de comando."""
    evaluator = LightRAGEvaluator()
    
    print("\n" + "="*80)
    print("🚀 LIGHT RAG EVALUATOR - Teste Individual de Perguntas")
    print("="*80)
    
    while True:
        print("\n📋 OPÇÕES:")
        print("  [L] Listar todas as perguntas disponíveis")
        print("  [1-10] Avaliar pergunta específica (com AI Judge)")
        print("  [Q] Avaliar pergunta SEM AI Judge (apenas respostas)")
        print("  [S] Sair")
        
        choice = input("\n👉 Escolha uma opção: ").strip().upper()
        
        if choice == 'S':
            print("\n👋 Até logo!")
            break
        
        elif choice == 'L':
            evaluator.list_questions()
        
        elif choice == 'Q':
            q_id = input("Digite o ID da pergunta (1-10): ").strip()
            try:
                q_id = int(q_id)
                evaluator.evaluate_single_question(q_id, skip_judge=True)
            except ValueError:
                print("❌ ID inválido!")
        
        elif choice.isdigit():
            q_id = int(choice)
            if 1 <= q_id <= 10:
                evaluator.evaluate_single_question(q_id, skip_judge=False)
            else:
                print("❌ ID deve estar entre 1 e 10!")
        
        else:
            print("❌ Opção inválida!")


if __name__ == "__main__":
    main()
