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

# Importar os sistemas RAG
from file_search_rag import ask_question as file_search_ask
from retriever import Retriever
from augmentation import Augmentation
from generation import Generation

# Importar módulos de refatoração
from report_generator import EvaluationReportGenerator
from console_presenter import ConsolePresenter

# Importar AI Judge com LiteLLM
from ai_judge import AIJudge

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
        
        # Inicializar AI Judge com LiteLLM (agnóstico de provider)
        self.judge = AIJudge(config_path=config_path)
        
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
                
                manual_evaluation = self.judge.judge_response(
                    question=question, 
                    context=manual_context_str, 
                    answer=manual_result['answer'],
                    system_name="RAG Manual"
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
                
                file_search_evaluation = self.judge.judge_response(
                    question=question,
                    context=file_search_context_str,
                    answer=file_search_result['answer'],
                    system_name="File Search RAG"
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
