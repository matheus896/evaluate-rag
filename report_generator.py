# report_generator.py
"""
Módulo responsável pela geração e persistência de relatórios de avaliação.
Suporta exportação em JSON e Markdown.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any


class EvaluationReportGenerator:
    """Gera e salva relatórios de avaliação RAG em JSON e Markdown."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Inicializa o gerador de relatórios.
        
        Args:
            output_dir: Diretório onde os relatórios serão salvos
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Garante que o diretório de saída existe."""
        if not os.path.isabs(self.output_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_dir = os.path.join(script_dir, self.output_dir)
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_result(self, result: Dict[str, Any]) -> Dict[str, str]:
        """
        Salva resultado completo em JSON e Markdown.
        
        Args:
            result: Dicionário com dados da avaliação
            
        Returns:
            Dict com caminhos dos arquivos salvos
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        question_id = result.get('question_id', 'unknown')
        
        # Salvar JSON
        json_path = self._save_json(result, question_id, timestamp)
        
        # Salvar Markdown
        md_path = self._save_markdown(result, question_id, timestamp)
        
        logging.info(f"💾 Relatório salvo: JSON={json_path} | MD={md_path}")
        
        return {
            'json': json_path,
            'markdown': md_path
        }
    
    def _save_json(self, result: Dict[str, Any], question_id: int, timestamp: str) -> str:
        """Salva resultado em formato JSON."""
        filename = f"evaluation_single_q{question_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logging.info(f"💾 JSON salvo em: {filepath}")
        return filepath
    
    def _save_markdown(self, result: Dict[str, Any], question_id: int, timestamp: str) -> str:
        """Salva resultado em formato Markdown."""
        filename = f"evaluation_single_q{question_id}_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        content = self._generate_markdown_content(result)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logging.info(f"📄 Markdown salvo em: {filepath}")
        return filepath
    
    def _generate_markdown_content(self, result: Dict[str, Any]) -> str:
        """Gera conteúdo do relatório em Markdown."""
        sections = []
        
        # Cabeçalho
        sections.append(self._build_header(result))
        
        # Pergunta
        sections.append(self._build_question_section(result))
        
        # RAG Manual
        manual = result.get('manual_rag')
        if manual:
            sections.append(self._build_rag_section(manual, "🔧 RAG Manual", "manual"))
        
        # File Search RAG
        file_search = result.get('file_search_rag')
        if file_search:
            sections.append(self._build_rag_section(file_search, "🔍 File Search RAG", "file_search"))
        
        # Comparação
        if manual and file_search:
            sections.append(self._build_comparison_section(manual, file_search))
        
        # Rodapé
        sections.append(self._build_footer())
        
        return "\n".join(sections)
    
    def _build_header(self, result: Dict[str, Any]) -> str:
        """Constrói cabeçalho do relatório."""
        timestamp_str = datetime.fromisoformat(result['timestamp']).strftime('%d/%m/%Y %H:%M:%S')
        
        return f"""# Avaliação: Pergunta {result['question_id']}

**Data:** {timestamp_str}

**Categoria:** {result['category']}
"""
    
    def _build_question_section(self, result: Dict[str, Any]) -> str:
        """Constrói seção da pergunta."""
        return f"""## 📝 Pergunta

{result['question']}

---
"""
    
    def _build_rag_section(self, rag_result: Dict[str, Any], title: str, rag_type: str) -> str:
        """Constrói seção de resultado RAG."""
        sections = [f"## {title}\n"]
        
        # Métricas
        latency = rag_result.get('latency', 0)
        latency_label = "Latência (TTFT)" if rag_type == "file_search" else "Latência"
        sections.append(f"**{latency_label}:** {latency:.2f}s\n")
        sections.append(f"**Chunks Recuperados:** {rag_result.get('num_chunks', 0)}\n")
        
        # Resposta
        sections.append("### Resposta\n")
        sections.append(f"{rag_result.get('answer', 'N/A')}\n")
        
        # Avaliação AI Judge
        if 'evaluation' in rag_result and 'error' not in rag_result['evaluation']:
            sections.append(self._build_judge_evaluation_table(rag_result['evaluation']))
        
        sections.append("---\n")
        
        return "\n".join(sections)
    
    def _build_judge_evaluation_table(self, evaluation: Dict[str, Any]) -> str:
        """Constrói tabela de avaliação do AI Judge."""
        lines = ["### ⚖️ Avaliação AI Judge\n"]
        lines.append("| Critério | Score | Justificativa |")
        lines.append("|----------|-------|---------------|")
        
        criteria_map = {
            'factual_consistency': 'Consistência Factual',
            'instruction_following': 'Seguir Instruções',
            'domain_knowledge': 'Conhecimento Jurídico',
            'context_precision': 'Precisão do Contexto',
            'context_recall': 'Cobertura do Contexto'
        }
        
        for key, label in criteria_map.items():
            if key in evaluation and isinstance(evaluation[key], dict):
                score = evaluation[key].get('score', 'N/A')
                justification = evaluation[key].get('justification', 'N/A')
                lines.append(f"| {label} | {score}/5 | {justification} |")
        
        if 'overall_assessment' in evaluation:
            lines.append(f"\n**Avaliação Geral:** {evaluation['overall_assessment']}\n")
        
        return "\n".join(lines)
    
    def _build_comparison_section(self, manual: Dict[str, Any], file_search: Dict[str, Any]) -> str:
        """Constrói seção de comparação entre os sistemas."""
        sections = ["## 📊 Comparação de Scores\n"]
        
        manual_eval = manual.get('evaluation', {})
        file_eval = file_search.get('evaluation', {})
        
        # Verificar se ambos têm avaliações válidas
        if 'error' in manual_eval or 'error' in file_eval:
            return ""
        
        # Tabela de comparação
        sections.append("| Critério | RAG Manual | File Search RAG | Diferença |")
        sections.append("|----------|------------|-----------------|----------|")
        
        criteria = ['factual_consistency', 'instruction_following', 'domain_knowledge', 
                   'context_precision', 'context_recall']
        
        criteria_labels = {
            'factual_consistency': 'Consistência Factual',
            'instruction_following': 'Seguir Instruções',
            'domain_knowledge': 'Conhecimento Jurídico',
            'context_precision': 'Precisão do Contexto',
            'context_recall': 'Cobertura do Contexto'
        }
        
        for criterion in criteria:
            if (criterion in manual_eval and isinstance(manual_eval[criterion], dict) and
                criterion in file_eval and isinstance(file_eval[criterion], dict)):
                
                m_score = manual_eval[criterion].get('score', 0)
                f_score = file_eval[criterion].get('score', 0)
                diff = f_score - m_score
                diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
                
                label = criteria_labels.get(criterion, criterion)
                sections.append(f"| {label} | {m_score}/5 | {f_score}/5 | {diff_str} |")
        
        # Vencedores
        sections.append(self._build_winner_summary(manual_eval, file_eval, criteria, criteria_labels))
        
        # Análise de chunks
        sections.append(self._build_chunk_analysis(manual, file_search))
        
        return "\n".join(sections)
    
    def _build_winner_summary(self, manual_eval: Dict, file_eval: Dict, 
                             criteria: list, criteria_labels: Dict) -> str:
        """Constrói resumo de vencedores por critério."""
        lines = ["\n### 🏆 Vencedor por Critério\n"]
        
        manual_wins = 0
        file_search_wins = 0
        ties = 0
        
        for criterion in criteria:
            if (criterion in manual_eval and isinstance(manual_eval[criterion], dict) and
                criterion in file_eval and isinstance(file_eval[criterion], dict)):
                
                m_score = manual_eval[criterion].get('score', 0)
                f_score = file_eval[criterion].get('score', 0)
                
                if m_score > f_score:
                    manual_wins += 1
                    winner = "🔧 RAG Manual"
                elif f_score > m_score:
                    file_search_wins += 1
                    winner = "🔍 File Search RAG"
                else:
                    ties += 1
                    winner = "🤝 Empate"
                
                label = criteria_labels.get(criterion, criterion)
                lines.append(f"- **{label}:** {winner}")
        
        lines.append(f"\n**Resumo:** RAG Manual ({manual_wins}) | File Search RAG ({file_search_wins}) | Empates ({ties})\n")
        
        return "\n".join(lines)
    
    def _build_chunk_analysis(self, manual: Dict[str, Any], file_search: Dict[str, Any]) -> str:
        """Constrói análise de chunks."""
        lines = ["---\n", "## 🔬 Análise de Chunks\n"]
        
        manual_chunks = manual.get('context', [])[:5]
        file_search_chunks = file_search.get('context', [])[:5]
        
        lines.append("| Métrica | RAG Manual | File Search RAG |")
        lines.append("|---------|------------|----------------|")
        lines.append(f"| Chunks enviados | {len(manual_chunks)} | {len(file_search_chunks)} |")
        lines.append(f"| Total de caracteres | {sum(len(c) for c in manual_chunks)} | {sum(len(c) for c in file_search_chunks)} |")
        
        manual_avg = sum(len(c) for c in manual_chunks) // max(len(manual_chunks), 1)
        file_avg = sum(len(c) for c in file_search_chunks) // max(len(file_search_chunks), 1)
        
        lines.append(f"| Tamanho médio/chunk | {manual_avg} chars | {file_avg} chars |")
        
        diff_chars = sum(len(c) for c in file_search_chunks) - sum(len(c) for c in manual_chunks)
        lines.append(f"\n**Diferença de caracteres:** {diff_chars:+d}\n")
        
        return "\n".join(lines)
    
    def _build_footer(self) -> str:
        """Constrói rodapé do relatório."""
        timestamp = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        return f"""---

*Relatório gerado automaticamente em {timestamp}*
"""
