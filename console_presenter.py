# console_presenter.py
"""
Módulo responsável pela formatação e apresentação de resultados no console.
Centraliza toda a lógica de output visual do sistema de avaliação.
"""

from typing import Dict, List, Any, Optional


class ConsolePresenter:
    """Formata e exibe resultados de avaliação RAG no console."""
    
    @staticmethod
    def print_question_header(question_id: int, category: str, question: str):
        """
        Exibe cabeçalho da pergunta sendo avaliada.
        
        Args:
            question_id: ID da pergunta
            category: Categoria da pergunta
            question: Texto da pergunta
        """
        print("\n" + "="*80)
        print(f"🎯 AVALIANDO PERGUNTA {question_id}")
        print("="*80)
        print(f"Categoria: {category}")
        print(f"Pergunta: {question}")
        print("="*80 + "\n")
    
    @staticmethod
    def print_rag_header(system_name: str):
        """
        Exibe cabeçalho de execução RAG.
        
        Args:
            system_name: Nome do sistema RAG
        """
        icon = "🔧" if "Manual" in system_name else "🔍"
        print(f"\n{icon} {system_name.upper()}")
        print("-" * 80)
    
    @staticmethod
    def print_rag_result(result: Dict[str, Any], system_name: str):
        """
        Exibe resultado de execução RAG.
        
        Args:
            result: Dicionário com resultado do RAG
            system_name: Nome do sistema RAG
        """
        latency = result.get('latency', 0)
        num_chunks = result.get('num_chunks', 0)
        answer = result.get('answer', 'N/A')
        
        latency_label = "TTFT" if "File Search" in system_name else ""
        
        print(f"✅ Resposta gerada em {latency:.2f}s{' (' + latency_label + ')' if latency_label else ''}")
        print(f"📦 Chunks recuperados: {num_chunks}")
        print(f"\n📝 Resposta:\n{answer}\n")
    
    @staticmethod
    def print_chunk_analysis(chunks: List[str], system_name: str):
        """
        Exibe análise detalhada dos chunks recuperados.
        
        Args:
            chunks: Lista de chunks (strings)
            system_name: Nome do sistema RAG
        """
        print(f"\n🔍 ANÁLISE - CHUNKS {system_name.upper()}:")
        print("-" * 80)
        
        for i, chunk in enumerate(chunks[:5], 1):
            chunk_size = len(chunk)
            chunk_preview = chunk[:200].replace('\n', ' ')
            print(f"Chunk {i}:")
            print(f"  • Tamanho: {chunk_size} caracteres")
            print(f"  • Preview: {chunk_preview}...")
            print(f"  • Tipo: {type(chunk)}")
            print()
    
    @staticmethod
    def print_judge_info(context_str: str):
        """
        Exibe informações sobre dados enviados ao AI Judge.
        
        Args:
            context_str: String concatenada de contexto
        """
        num_chunks = context_str.count("--- CHUNK SEPARATOR ---") + 1
        print(f"📊 Total de caracteres enviados ao AI Judge: {len(context_str)}")
        print(f"📊 Número de chunks enviados: {num_chunks}\n")
    
    @staticmethod
    def print_judge_evaluation(evaluation: Dict[str, Any], system_name: str):
        """
        Exibe avaliação do AI Judge.
        
        Args:
            evaluation: Dicionário com avaliação
            system_name: Nome do sistema RAG
        """
        if 'error' in evaluation:
            print(f"\n❌ Erro na avaliação do {system_name}")
            return
        
        print(f"\n⚖️  Avaliação AI Judge ({system_name}):")
        for criterion, data in evaluation.items():
            if isinstance(data, dict) and 'score' in data:
                score = data['score']
                justification = data['justification']
                print(f"  • {criterion}: {score}/5 - {justification}")
    
    @staticmethod
    def print_wait_message(seconds: int, reason: str):
        """
        Exibe mensagem de espera.
        
        Args:
            seconds: Tempo de espera em segundos
            reason: Motivo da espera
        """
        print(f"\n⏳ Aguardando {seconds}s {reason}...")
    
    @staticmethod
    def print_comparison_header():
        """Exibe cabeçalho de comparação."""
        print("\n" + "="*80)
        print("📊 COMPARAÇÃO DE SCORES")
        print("="*80)
    
    @staticmethod
    def print_score_comparison(manual_result: Dict[str, Any], file_search_result: Dict[str, Any]):
        """
        Exibe tabela de comparação de scores.
        
        Args:
            manual_result: Resultado do RAG Manual
            file_search_result: Resultado do File Search RAG
        """
        criteria = ['factual_consistency', 'instruction_following', 'domain_knowledge', 
                   'context_precision', 'context_recall']
        
        print(f"\n{'Critério':<25} {'Manual':<10} {'File Search':<15} {'Diferença'}")
        print("-" * 80)
        
        manual_eval = manual_result.get('evaluation', {})
        file_eval = file_search_result.get('evaluation', {})
        
        for criterion in criteria:
            if (criterion in manual_eval and 'score' in manual_eval[criterion] and
                criterion in file_eval and 'score' in file_eval[criterion]):
                
                m_score = manual_eval[criterion]['score']
                f_score = file_eval[criterion]['score']
                diff = f_score - m_score
                diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
                
                print(f"{criterion:<25} {m_score}/5{' '*5} {f_score}/5{' '*10} {diff_str}")
    
    @staticmethod
    def print_chunk_comparison(manual_result: Dict[str, Any], file_search_result: Dict[str, Any]):
        """
        Exibe comparação de chunks entre sistemas.
        
        Args:
            manual_result: Resultado do RAG Manual
            file_search_result: Resultado do File Search RAG
        """
        print("\n" + "="*80)
        print("�� COMPARAÇÃO DE CHUNKS (Análise de Consistência)")
        print("="*80)
        
        manual_chunks = manual_result.get('context', [])[:5]
        file_search_chunks = file_search_result.get('context', [])[:5]
        
        # RAG Manual
        print(f"\nRAG Manual:")
        print(f"  • Chunks enviados ao Judge: {len(manual_chunks)}")
        print(f"  • Total de caracteres: {sum(len(c) for c in manual_chunks)}")
        
        manual_avg = sum(len(c) for c in manual_chunks) // max(len(manual_chunks), 1)
        print(f"  • Tamanho médio por chunk: {manual_avg} chars")
        
        # File Search RAG
        print(f"\nFile Search RAG:")
        print(f"  • Chunks enviados ao Judge: {len(file_search_chunks)}")
        print(f"  • Total de caracteres: {sum(len(c) for c in file_search_chunks)}")
        
        file_avg = sum(len(c) for c in file_search_chunks) // max(len(file_search_chunks), 1)
        print(f"  • Tamanho médio por chunk: {file_avg} chars")
        
        # Diferenças
        print(f"\nDiferença:")
        diff_chunks = len(file_search_chunks) - len(manual_chunks)
        diff_chars = sum(len(c) for c in file_search_chunks) - sum(len(c) for c in manual_chunks)
        print(f"  • Diferença em chunks: {diff_chunks:+d}")
        print(f"  • Diferença em caracteres: {diff_chars:+d}")
        
        consistency = "✅ CONSISTENTE" if abs(diff_chunks) == 0 else "⚠️ INCONSISTENTE"
        print(f"  • Conclusão: {consistency}")
    
    @staticmethod
    def print_completion_message():
        """Exibe mensagem de conclusão."""
        print("\n✅ Avaliação concluída!")
    
    @staticmethod
    def print_error(message: str, system_name: Optional[str] = None):
        """
        Exibe mensagem de erro.
        
        Args:
            message: Mensagem de erro
            system_name: Nome do sistema (opcional)
        """
        prefix = f"❌ Erro no {system_name}: " if system_name else "❌ "
        print(f"{prefix}{message}")
